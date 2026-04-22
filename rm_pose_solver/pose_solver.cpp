/**
 * rm_pose_solver: Yaw optimization via reprojection.
 *
 * Assumes images are pre-undistorted — no distortion coefficients needed.
 *
 * Strategy: split [yaw_lo, yaw_hi] at the midpoint. Two std::threads each
 * run Brent's method (boost::math::tools::brent_find_minima) on their half.
 * The lower half seeds near 1/4 of the range, the upper near 3/4 — one
 * thread per basin covers both U-shaped and W-shaped cost landscapes.
 * ~26 cost evaluations per thread vs ~180 for the old 1°-step sweep.
 *
 * Boost.Math is header-only: no extra link step, just include dirs.
 *
 * Exposes one function to Python via pybind11:
 *   solve_yaw(...) — dual-seeded Brent minimization
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <opencv2/opencv.hpp>
#include <boost/math/tools/minima.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>
#include <tuple>
#include "utils.hpp"
#include <condition_variable>

namespace py = pybind11;

#if defined(__clang__)
#define RM_UNROLL_4 _Pragma("clang loop unroll_count(4)")
#elif defined(__GNUC__)
#define RM_UNROLL_4 _Pragma("GCC unroll 4")
#else
#define RM_UNROLL_4
#endif

// ---------------------------------------------------------------------------
// Input bundle — cache-friendly, aligned, no heap after construction.
// ---------------------------------------------------------------------------

struct SolveInputs {
    alignas(64) std::array<double, 8>  observed_xy; // [x0,y0, x1,y1, x2,y2, x3,y3]
    double tx, ty, tz;
    alignas(64) std::array<double, 12> obj_pts;     // row-major [pt0_x, pt0_y, pt0_z, ...]
    double fx, fy, cx, cy;
    double sin_pitch, cos_pitch, sin_roll, cos_roll;
};

// ---------------------------------------------------------------------------
// Unpack Python arrays into SolveInputs.
// ---------------------------------------------------------------------------

static SolveInputs unpack_inputs(
    py::array_t<double> image_points,
    py::array_t<double> position_xyz,
    py::array_t<double> plate_matrix,
    py::array_t<double> camera_matrix,
    double fixed_pitch,
    double fixed_roll)
{
    auto ip  = image_points.unchecked<2>();  // (4,2)
    auto pos = position_xyz.unchecked<1>();  // (3,)
    auto pm  = plate_matrix.unchecked<2>();  // (4,3)
    auto km  = camera_matrix.unchecked<2>(); // (3,3)

    if (ip.shape(0) != 4 || ip.shape(1) != 2)
        throw std::invalid_argument("image_points must be shape (4, 2)");
    if (pos.shape(0) != 3)
        throw std::invalid_argument("position_xyz must be shape (3,)");
    if (pm.shape(0) != 4 || pm.shape(1) != 3)
        throw std::invalid_argument("plate_matrix must be shape (4, 3)");
    if (km.shape(0) != 3 || km.shape(1) != 3)
        throw std::invalid_argument("camera_matrix must be shape (3, 3)");

    SolveInputs inp{};

    RM_UNROLL_4
    for (int i = 0; i < 4; ++i) {
        inp.observed_xy[2*i]   = ip(i, 0);
        inp.observed_xy[2*i+1] = ip(i, 1);
    }

    inp.tx = pos(0); inp.ty = pos(1); inp.tz = pos(2);

    for (int i = 0; i < 4; ++i) {
        inp.obj_pts[3*i]   = pm(i, 0);
        inp.obj_pts[3*i+1] = pm(i, 1);
        inp.obj_pts[3*i+2] = pm(i, 2);
    }

    inp.fx = km(0, 0); inp.fy = km(1, 1);
    inp.cx = km(0, 2); inp.cy = km(1, 2);

    inp.sin_pitch = std::sin(fixed_pitch); inp.cos_pitch = std::cos(fixed_pitch);
    inp.sin_roll  = std::sin(fixed_roll);  inp.cos_roll  = std::cos(fixed_roll);

    return inp;
}

// ---------------------------------------------------------------------------
// Cost: squared reprojection error at a given yaw.
// Inline pinhole projection — no distortion, no heap.
// ---------------------------------------------------------------------------

static inline double squared_reprojection_error(
    const SolveInputs& inp, double yaw) noexcept
{
    const cv::Vec3d rv_cv2 = standard_rvec_to_cv2(
        euler_to_rvec_fast(yaw,
            inp.sin_pitch, inp.cos_pitch,
            inp.sin_roll,  inp.cos_roll));

    cv::Matx33d R;
    cv::Rodrigues(rv_cv2, R);

    const double r00=R(0,0), r01=R(0,1), r02=R(0,2);
    const double r10=R(1,0), r11=R(1,1), r12=R(1,2);
    const double r20=R(2,0), r21=R(2,1), r22=R(2,2);
    const double fx=inp.fx, fy=inp.fy, cx=inp.cx, cy=inp.cy;
    const double tx=inp.tx, ty=inp.ty, tz=inp.tz;

    double sq = 0.0;

    RM_UNROLL_4
    for (int i = 0; i < 4; ++i) {
        const double px=inp.obj_pts[3*i], py=inp.obj_pts[3*i+1], pz=inp.obj_pts[3*i+2];
        const double X = r00*px + r01*py + r02*pz + tx;
        const double Y = r10*px + r11*py + r12*pz + ty;
        const double Z = r20*px + r21*py + r22*pz + tz;
        const double inv_Z = 1.0 / Z;
        const double dx = fx*(X*inv_Z) + cx - inp.observed_xy[2*i];
        const double dy = fy*(Y*inv_Z) + cy - inp.observed_xy[2*i+1];
        sq += dx*dx + dy*dy;
    }
    return sq;
}

// ---------------------------------------------------------------------------
// Persistent two-thread pool for Brent searches.
// Threads are created once (on first call) and reused forever.
// ---------------------------------------------------------------------------

struct BrentPool {
    struct Job {
        const SolveInputs* inputs = nullptr;
        double lo = 0, hi = 0;
        std::pair<double,double> result;
        bool ready = false;   // input ready for worker
        bool done  = false;   // worker finished
    };

    std::array<Job, 2>              jobs;
    std::array<std::thread, 2>      threads;
    std::array<std::mutex, 2>       mtx;
    std::array<std::condition_variable, 2> cv;
    bool shutdown = false;

    BrentPool() {
        for (int id = 0; id < 2; ++id) {
            threads[id] = std::thread([this, id] {
                while (true) {
                    std::unique_lock lk(mtx[id]);
                    cv[id].wait(lk, [&] { return jobs[id].ready || shutdown; });
                    if (shutdown) return;

                    auto& job = jobs[id];
                    boost::uintmax_t max_iter = 200;
                    int bits = static_cast<int>(std::ceil(
                        std::log2((job.hi - job.lo) / kYawTolRad)));

                    job.result = boost::math::tools::brent_find_minima(
                        [&](double y) {
                            return squared_reprojection_error(*job.inputs, y);
                        },
                        job.lo, job.hi, bits, max_iter);

                    job.ready = false;
                    job.done  = true;
                    lk.unlock();
                    cv[id].notify_one();
                }
            });
        }
    }

    ~BrentPool() {
        shutdown = true;
        for (int id = 0; id < 2; ++id) {
            cv[id].notify_one();
            threads[id].join();
        }
    }

    // Submit both jobs and block until both finish.
    void run(const SolveInputs& inp,
             double lo1, double hi1,
             double lo2, double hi2)
    {
        // Set up jobs
        for (int id = 0; id < 2; ++id) {
            std::lock_guard lk(mtx[id]);
            jobs[id].inputs = &inp;
            jobs[id].lo     = (id == 0) ? lo1 : lo2;
            jobs[id].hi     = (id == 0) ? hi1 : hi2;
            jobs[id].done   = false;
            jobs[id].ready  = true;
        }
        // Wake both workers
        cv[0].notify_one();
        cv[1].notify_one();

        // Wait for both to finish
        for (int id = 0; id < 2; ++id) {
            std::unique_lock lk(mtx[id]);
            cv[id].wait(lk, [&] { return jobs[id].done; });
        }
    }

    static constexpr double kYawTolRad = 0.1 * (M_PI / 180.0);
};

static BrentPool g_pool; 

// ---------------------------------------------------------------------------
// solve_yaw — public entry point
// ---------------------------------------------------------------------------

/**
 * Minimize plate keypoint reprojection error over yaw using Brent's method.
 * The search range is split at its midpoint; two threads work in parallel,
 * one on each half, so both basins of a W-shaped cost surface are covered.
 * Images must be pre-undistorted; no distortion coefficients are accepted.
 *
 * Parameters:
 *   image_points  : (4,2) float64  – observed pixel keypoints [BL,BR,TR,TL]
 *   position_xyz  : (3,)  float64  – tvec in camera frame (metres)
 *   plate_matrix  : (4,3) float64  – 3-D object points of plate corners
 *   camera_matrix : (3,3) float64  – intrinsic K (fx,fy,cx,cy only used)
 *   fixed_pitch   : double          – pitch in radians (from odometry/IMU)
 *   fixed_roll    : double          – roll  in radians (from odometry/IMU)
 *   yaw_lo        : double          – search lower bound  (default -π/2)
 *   yaw_hi        : double          – search upper bound  (default +π/2)
 *
 * Returns: (optimized_yaw: double, reprojection_rms_pixels: double)
 */
static std::tuple<double, double> solve_yaw(
    py::array_t<double> image_points,
    py::array_t<double> position_xyz,
    py::array_t<double> plate_matrix,
    py::array_t<double> camera_matrix,
    double fixed_pitch,
    double fixed_roll,
    double yaw_lo,
    double yaw_hi)
{
    if (!std::isfinite(yaw_lo) || !std::isfinite(yaw_hi))
        throw std::invalid_argument("yaw bounds must be finite");
    if (yaw_hi < yaw_lo)
        throw std::invalid_argument("yaw_hi must be >= yaw_lo");

    const SolveInputs inputs = unpack_inputs(
        image_points, position_xyz, plate_matrix, camera_matrix,
        fixed_pitch, fixed_roll);

    // Split at midpoint. Thread 1 probes near lo+span/4 (≈1/4 of range),
    // thread 2 near lo+3*span/4 (≈3/4). Covers both U and W cost shapes.
    const double mid = 0.5 * (yaw_lo + yaw_hi);

    std::pair<double,double> res1, res2;

    {
        py::gil_scoped_release release;
        g_pool.run(inputs, yaw_lo, mid, mid, yaw_hi);
        res1 = g_pool.jobs[0].result;
        res2 = g_pool.jobs[1].result;
    }

    // Pick whichever half found the lower cost.
    const auto& best = (res1.second <= res2.second) ? res1 : res2;
    const double rms = std::sqrt(0.25 * best.second);
    return {best.first, rms};
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(_rm_pose_solver, m)
{
    m.doc() = "RoboMaster pose solver — dual Brent minimization via Boost.Math";

    m.def("solve_yaw", &solve_yaw,
        py::arg("image_points"),
        py::arg("position_xyz"),
        py::arg("plate_matrix"),
        py::arg("camera_matrix"),
        py::arg("fixed_pitch"),
        py::arg("fixed_roll"),
        py::arg("yaw_lo") = -M_PI / 2.0,
        py::arg("yaw_hi") =  M_PI / 2.0,
        R"doc(
Minimize plate keypoint reprojection error over yaw using Brent's method.
Two threads search the lower and upper halves of [yaw_lo, yaw_hi] in parallel.
Images must be pre-undistorted (no distortion coefficients accepted).

Args:
    image_points  (np.ndarray[float64, (4,2)]): observed pixel keypoints [BL,BR,TR,TL]
    position_xyz  (np.ndarray[float64, (3,)]): tvec in camera frame (meters)
    plate_matrix  (np.ndarray[float64, (4,3)]): 3-D plate corner object points
    camera_matrix (np.ndarray[float64, (3,3)]): intrinsic K matrix
    fixed_pitch   (float): pitch in radians
    fixed_roll    (float): roll in radians
    yaw_lo        (float): search lower bound, default -pi/2
    yaw_hi        (float): search upper bound, default +pi/2

Returns:
    tuple[float, float]: (best_yaw_rad, reprojection_error_pixels_rms)
)doc");
}
