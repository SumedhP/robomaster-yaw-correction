/**
 * rm_pose_solver: Yaw optimization via reprojection.
 *
 * Assumes images are pre-undistorted — no distortion coefficients needed.
 * Uses a persistent thread pool to eliminate per-call thread spawn overhead.
 * Projects points inline with pinhole math instead of cv::projectPoints.
 *
 * Exposes one function to Python via pybind11:
 *   1. solve_yaw(...) - bounded brute-force search over yaw
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <opencv2/opencv.hpp>
#include <cmath>
#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <limits>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>
#include <stdexcept>
#include "utils.hpp"

namespace py = pybind11;

#if defined(__clang__)
#define RM_UNROLL_4 _Pragma("clang loop unroll_count(4)")
#elif defined(__GNUC__)
#define RM_UNROLL_4 _Pragma("GCC unroll 4")
#else
#define RM_UNROLL_4
#endif

// ---------------------------------------------------------------------------
// Input bundle — no dist_coeffs, camera intrinsics stored as plain doubles,
// object points stored as a flat cache-friendly array.
// ---------------------------------------------------------------------------

struct SolveInputs {
    // Observed pixel coordinates, 4 corners [BL,BR,TR,TL]
    alignas(64) std::array<double, 8> observed_xy; // [x0,y0, x1,y1, x2,y2, x3,y3]

    // Camera-frame translation vector
    double tx, ty, tz;

    // 3-D object points — row-major [pt0_x, pt0_y, pt0_z, pt1_x, ...]
    alignas(64) std::array<double, 12> obj_pts;

    // Pinhole intrinsics (no distortion — image is pre-undistorted)
    double fx, fy, cx, cy;

    // Precomputed pitch/roll trig terms
    double sin_pitch, cos_pitch, sin_roll, cos_roll;
};

struct SweepResult {
    double yaw;
    double sq;
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr double kYawIncrementRad = M_PI / 180.0;
static constexpr double kYawTailEpsilon  = 1e-12;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static inline bool is_better_result(double candidate_sq,
                                    double candidate_yaw,
                                    const SweepResult& best) noexcept
{
    if (candidate_sq < best.sq) return true;
    if (candidate_sq > best.sq) return false;
    return candidate_yaw < best.yaw;
}

// ---------------------------------------------------------------------------
// Unpack Python arrays into SolveInputs.
// dist_coeffs are gone entirely — caller must undistort before passing in.
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
        inp.observed_xy[2 * i]     = ip(i, 0);
        inp.observed_xy[2 * i + 1] = ip(i, 1);
    }

    inp.tx = pos(0);
    inp.ty = pos(1);
    inp.tz = pos(2);

    for (int i = 0; i < 4; ++i) {
        inp.obj_pts[3 * i]     = pm(i, 0);
        inp.obj_pts[3 * i + 1] = pm(i, 1);
        inp.obj_pts[3 * i + 2] = pm(i, 2);
    }

    // Extract the four scalars we actually need from K.
    // K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    inp.fx = km(0, 0);
    inp.fy = km(1, 1);
    inp.cx = km(0, 2);
    inp.cy = km(1, 2);

    inp.sin_pitch = std::sin(fixed_pitch);
    inp.cos_pitch = std::cos(fixed_pitch);
    inp.sin_roll  = std::sin(fixed_roll);
    inp.cos_roll  = std::cos(fixed_roll);

    return inp;
}

// ---------------------------------------------------------------------------
// Core per-sample cost: inline pinhole projection, no distortion.
//
// We call euler_to_rvec_fast + standard_rvec_to_cv2 to obtain the rvec
// (preserving whatever convention utils.hpp encodes), then use cv::Rodrigues
// to get the rotation matrix as a Matx33d (stack-allocated, no heap).
// The projection itself is fully inlined: R*p + t → divide by Z → apply K.
// ---------------------------------------------------------------------------

static inline double squared_reprojection_error(
    const SolveInputs& inp,
    double yaw) noexcept
{
    // Build rotation matrix from Euler angles via existing utils convention.
    const cv::Vec3d rv_std = euler_to_rvec_fast(
        yaw,
        inp.sin_pitch, inp.cos_pitch,
        inp.sin_roll,  inp.cos_roll);
    const cv::Vec3d rv_cv2 = standard_rvec_to_cv2(rv_std);

    cv::Matx33d R;
    cv::Rodrigues(rv_cv2, R); // Stack-allocated 3x3, no heap.

    const double r00 = R(0,0), r01 = R(0,1), r02 = R(0,2);
    const double r10 = R(1,0), r11 = R(1,1), r12 = R(1,2);
    const double r20 = R(2,0), r21 = R(2,1), r22 = R(2,2);

    const double fx = inp.fx, fy = inp.fy, cx = inp.cx, cy = inp.cy;
    const double tx = inp.tx, ty = inp.ty, tz = inp.tz;

    double sq = 0.0;

    RM_UNROLL_4
    for (int i = 0; i < 4; ++i) {
        const double px = inp.obj_pts[3 * i];
        const double py = inp.obj_pts[3 * i + 1];
        const double pz = inp.obj_pts[3 * i + 2];

        // Rotate + translate into camera frame.
        const double X = r00*px + r01*py + r02*pz + tx;
        const double Y = r10*px + r11*py + r12*pz + ty;
        const double Z = r20*px + r21*py + r22*pz + tz;

        // Pinhole project (undistorted image — no distortion step).
        const double inv_Z = 1.0 / Z;
        const double u = fx * (X * inv_Z) + cx;
        const double v = fy * (Y * inv_Z) + cy;

        const double dx = u - inp.observed_xy[2 * i];
        const double dy = v - inp.observed_xy[2 * i + 1];
        sq += dx*dx + dy*dy;
    }

    return sq;
}

// ---------------------------------------------------------------------------
// Persistent thread pool — initialized once at module load, lives until
// module unload. Eliminates all thread spawn/join overhead from solve_yaw.
//
// Design:
//   Workers sleep on cv_dispatch_. Main thread sets up task fields and
//   bumps generation_ to wake them. Workers grab indices via an atomic
//   counter, accumulate a local best, store it, then signal cv_done_.
//   False-sharing between per-worker results is avoided by padding.
// ---------------------------------------------------------------------------

struct alignas(64) WorkerSlot {
    SweepResult result{ 0.0, std::numeric_limits<double>::infinity() };
    char _pad[64 - sizeof(SweepResult)]; // keep each slot on its own cache line
};

class SweepThreadPool {
public:
    explicit SweepThreadPool(int n_threads)
        : n_workers_(n_threads),
          worker_slots_(n_threads)
    {
        workers_.reserve(n_threads);
        for (int id = 0; id < n_threads; ++id)
            workers_.emplace_back([this, id] { worker_loop(id); });
    }

    ~SweepThreadPool()
    {
        {
            std::lock_guard<std::mutex> lk(mu_);
            shutdown_ = true;
            ++generation_;
        }
        cv_dispatch_.notify_all();
        for (auto& w : workers_) w.join();
    }

    // Not copyable or movable — owned by the module singleton.
    SweepThreadPool(const SweepThreadPool&)            = delete;
    SweepThreadPool& operator=(const SweepThreadPool&) = delete;

    // Run a sweep and return the global best result.
    SweepResult run(const SolveInputs& inputs,
                    double yaw_lo,
                    double yaw_hi,
                    int    sample_count,
                    bool   include_tail)
    {
        // Publish task; bump generation to wake workers.
        {
            std::lock_guard<std::mutex> lk(mu_);
            task_inputs_       = &inputs;
            task_yaw_lo_       = yaw_lo;
            task_yaw_hi_       = yaw_hi;
            task_sample_count_ = sample_count;
            task_include_tail_ = include_tail;
            next_index_.store(0, std::memory_order_relaxed);
            done_count_.store(0, std::memory_order_relaxed);
            ++generation_;
        }
        cv_dispatch_.notify_all();

        // Wait until every worker has stored its local best.
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_done_.wait(lk, [&] {
                return done_count_.load(std::memory_order_acquire) == n_workers_;
            });
        }

        // Reduce across worker slots.
        SweepResult global = worker_slots_[0].result;
        for (int i = 1; i < n_workers_; ++i) {
            const SweepResult& r = worker_slots_[i].result;
            if (is_better_result(r.sq, r.yaw, global))
                global = r;
        }
        return global;
    }

    int worker_count() const noexcept { return n_workers_; }

private:
    void worker_loop(int id)
    {
        int last_gen = 0;

        // thread_local projected buffer: allocated once per thread, reused forever.
        // (With inlined projection we don't need a projected[] vector anymore,
        // but keeping the slot here for any future cv::projectPoints fallback.)

        while (true) {
            // Sleep until a new task (generation changes) or shutdown.
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_dispatch_.wait(lk, [&] {
                    return generation_ != last_gen || shutdown_;
                });
                if (shutdown_) return;
                last_gen = generation_;
            }

            // Grab indices atomically — no fixed stride assignment, so fast
            // workers naturally pick up slack from slower ones.
            SweepResult local{ task_yaw_lo_, std::numeric_limits<double>::infinity() };

            for (int idx = next_index_.fetch_add(1, std::memory_order_relaxed);
                 idx < task_sample_count_;
                 idx = next_index_.fetch_add(1, std::memory_order_relaxed))
            {
                const bool is_tail = task_include_tail_ && (idx == task_sample_count_ - 1);
                const double yaw   = is_tail
                    ? task_yaw_hi_
                    : task_yaw_lo_ + kYawIncrementRad * static_cast<double>(idx);

                const double sq = squared_reprojection_error(*task_inputs_, yaw);
                if (!std::isfinite(sq)) continue;

                if (is_better_result(sq, yaw, local)) {
                    local.yaw = yaw;
                    local.sq  = sq;
                }
            }

            worker_slots_[id].result = local;

            // Signal completion. The last worker to arrive wakes the main thread.
            const int finished = done_count_.fetch_add(1, std::memory_order_acq_rel) + 1;
            if (finished == n_workers_) {
                // Notify under the lock so the main thread's predicate check
                // and cv_done_.wait() can't race against the notify.
                std::lock_guard<std::mutex> lk(mu_);
                cv_done_.notify_one();
            }
        }
    }

    // --- Shared task description (written by main, read by workers) ----------
    // Protected by mu_ during setup; workers read after waking without the lock
    // because generation_ acts as a publication barrier via the condition variable.
    const SolveInputs* task_inputs_       = nullptr;
    double             task_yaw_lo_       = 0.0;
    double             task_yaw_hi_       = 0.0;
    int                task_sample_count_ = 0;
    bool               task_include_tail_ = false;

    std::atomic<int> next_index_{ 0 }; // work-stealing index
    std::atomic<int> done_count_{ 0 }; // workers completed this round

    std::mutex              mu_;
    std::condition_variable cv_dispatch_; // main → workers
    std::condition_variable cv_done_;     // workers → main

    int  generation_ = 0;
    bool shutdown_   = false;

    int                    n_workers_;
    std::vector<WorkerSlot> worker_slots_; // padded to avoid false sharing
    std::vector<std::thread> workers_;
};

// Module-level singleton — created in PYBIND11_MODULE, destroyed at unload.
static std::unique_ptr<SweepThreadPool> g_pool;

// ---------------------------------------------------------------------------
// solve_yaw — public entry point
// ---------------------------------------------------------------------------

/**
 * Brute-force yaw sweep in a fixed range.
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

    // Compute sample count (mirrors original logic exactly).
    const int    full_steps          = static_cast<int>(std::floor((yaw_hi - yaw_lo) / kYawIncrementRad));
    const double last_regular_yaw    = yaw_lo + kYawIncrementRad * static_cast<double>(full_steps);
    const bool   include_tail        = last_regular_yaw < (yaw_hi - kYawTailEpsilon);
    const int    sample_count        = full_steps + 1 + (include_tail ? 1 : 0);

    SweepResult best;
    {
        py::gil_scoped_release release;
        best = g_pool->run(inputs, yaw_lo, yaw_hi, sample_count, include_tail);
    }

    const double rms = std::sqrt(0.25 * best.sq);
    return { best.yaw, rms };
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(_rm_pose_solver, m)
{
    m.doc() = "RoboMaster pose solver — persistent thread pool, inlined undistorted projection";

    // Build the pool once using all hardware threads.
    // num_threads in solve_yaw() is accepted for API compatibility but ignored.
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    g_pool = std::make_unique<SweepThreadPool>(std::max(1, hw));

    m.def("solve_yaw", &solve_yaw,
        py::arg("image_points"),
        py::arg("position_xyz"),
        py::arg("plate_matrix"),
        py::arg("camera_matrix"),
        py::arg("fixed_pitch"),
        py::arg("fixed_roll"),
        py::arg("yaw_lo")      = -M_PI / 2.0,
        py::arg("yaw_hi")      =  M_PI / 2.0,
        R"doc(
Brute-force camera-frame yaw to minimize plate keypoint reprojection error.
Images must be pre-undistorted (no distortion coefficients accepted).

Args:
    image_points  (np.ndarray[float64, (4,2)]): observed pixel keypoints [BL,BR,TR,TL]
    position_xyz  (np.ndarray[float64, (3,)]): tvec in camera frame (meters)
    plate_matrix  (np.ndarray[float64, (4,3)]): 3-D plate corner object points
    camera_matrix (np.ndarray[float64, (3,3)]): intrinsic K matrix
    fixed_pitch   (float): pitch in radians (from odometry/IMU)
    fixed_roll    (float): roll in radians (from odometry/IMU)
    yaw_lo        (float): search lower bound, default -pi/2
    yaw_hi        (float): search upper bound, default +pi/2

Returns:
    tuple[float, float]: (best_yaw_rad, reprojection_error_pixels_rms)
)doc");
}