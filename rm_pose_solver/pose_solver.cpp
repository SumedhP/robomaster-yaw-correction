#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <boost/math/tools/minima.hpp>

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr double kYawTolRad = 0.1 * (M_PI / 180.0);
static constexpr int kBrentBits = static_cast<int>(
    std::ceil(std::log2(M_PI / kYawTolRad))); // worst-case bits for [-π/2, π/2]

// ---------------------------------------------------------------------------
// Input bundle
// ---------------------------------------------------------------------------

struct SolveInputs
{
    // 4 object points as columns: [p0 | p1 | p2 | p3], shape 3×4
    Eigen::Matrix<double, 3, 4> obj_pts;

    // Observed pixel coords, shape 2×4
    Eigen::Matrix<double, 2, 4> observed;

    // Camera intrinsics
    double fx, fy, cx, cy;

    // Translation vector
    Eigen::Vector3d t;

    // Matrices for calculating rotation
    Eigen::Matrix3d R_world_to_cam;
    Eigen::Matrix3d R_plate_yx;
};

// ---------------------------------------------------------------------------
// Build Ry(a) * Rx(b) — the yaw-free part of ZYX rotation.
// ---------------------------------------------------------------------------


static inline Eigen::Matrix3d build_yx_rotation(double pitch, double roll) noexcept
{
    const double cp = std::cos(pitch), sp = std::sin(pitch);
    const double cr = std::cos(roll),  sr = std::sin(roll);

    Eigen::Matrix3d R;
    R << cp,      sp * sr,  sp * cr,
         0,       cr,      -sr,
        -sp,      cp * sr,  cp * cr;
    return R;
}

// ---------------------------------------------------------------------------
// Unpack Python arrays into SolveInputs — runs once per solve call.
// Computes R_rel = R_cam^T * R_plate and decomposes to pitch/roll in cam frame.
// ---------------------------------------------------------------------------

static SolveInputs unpack_inputs(
    py::array_t<double> image_points,
    py::array_t<double> position_xyz,
    py::array_t<double> plate_matrix,
    py::array_t<double> camera_matrix,
    double plate_pitch,
    double plate_roll,
    double cam_pitch,
    double cam_roll)
{
    auto ip  = image_points.unchecked<2>();  // (4,2)
    auto pos = position_xyz.unchecked<1>(); // (3,)
    auto pm  = plate_matrix.unchecked<2>(); // (4,3)
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

    for (int i = 0; i < 4; ++i)
    {
        inp.observed(0, i) = ip(i, 0);
        inp.observed(1, i) = ip(i, 1);
        inp.obj_pts(0, i)  = pm(i, 0);
        inp.obj_pts(1, i)  = pm(i, 1);
        inp.obj_pts(2, i)  = pm(i, 2);
    }

    inp.t  = {pos(0), pos(1), pos(2)};
    inp.fx = km(0, 0);
    inp.fy = km(1, 1);
    inp.cx = km(0, 2);
    inp.cy = km(1, 2);

    // Compute plate orientation in camera frame.
    const Eigen::Matrix3d R_cam   = build_yx_rotation(cam_pitch,   cam_roll);
    const Eigen::Matrix3d R_plate_yx = build_yx_rotation(plate_pitch, plate_roll);

    inp.R_world_to_cam = R_cam.transpose();
    inp.R_plate_yx = R_plate_yx;

    return inp;
}

// ---------------------------------------------------------------------------
// Build ZYX rotation matrix R = R_world_to_cam * Rz(yaw) * R_plate_yx
// ---------------------------------------------------------------------------

static inline Eigen::Matrix3d build_rotation(
    const SolveInputs &inp, double yaw) noexcept
{
    const double cy = std::cos(yaw), sy = std::sin(yaw);

    Eigen::Matrix3d Rz;
    Rz << cy, -sy, 0,
          sy,  cy, 0,
          0,   0,  1;

    return inp.R_world_to_cam * Rz * inp.R_plate_yx;
}

// ---------------------------------------------------------------------------
// Cost: squared reprojection error over all 4 points.
//
// Camera convention: X=forward (depth), Y=left, Z=up.
// Projection: u = fx * (-Y/X) + cx,  v = fy * (-Z/X) + cy
// ---------------------------------------------------------------------------

static inline double squared_reprojection_error(
    const SolveInputs &inp, double yaw) noexcept
{
    const Eigen::Matrix<double, 3, 4> cam =
        build_rotation(inp, yaw) * inp.obj_pts + inp.t.replicate<1, 4>();

    const Eigen::Array<double, 1, 4> inv_X = cam.row(0).array().inverse();

    const Eigen::Array<double, 1, 4> u =
        inp.fx * (-cam.row(1).array() * inv_X) + inp.cx;
    const Eigen::Array<double, 1, 4> v =
        inp.fy * (-cam.row(2).array() * inv_X) + inp.cy;

    const Eigen::Array<double, 1, 4> du = u - inp.observed.row(0).array();
    const Eigen::Array<double, 1, 4> dv = v - inp.observed.row(1).array();

    return (du * du + dv * dv).sum();
}

// ---------------------------------------------------------------------------
// Persistent two-thread pool for parallel Brent searches.
// Created once on first solve call via a local static.
// ---------------------------------------------------------------------------

class BrentPool
{
public:
    BrentPool()
    {
        for (int id = 0; id < 2; ++id)
            workers_[id] = std::thread([this, id] { worker_loop(id); });
    }

    ~BrentPool()
    {
        shutdown_ = true;
        for (int id = 0; id < 2; ++id)
        {
            cv_[id].notify_one();
            workers_[id].join();
        }
    }

    std::array<std::pair<double, double>, 2> run(
        const SolveInputs &inp,
        double lo, double mid, double hi)
    {
        bounds_[0] = {lo, mid};
        bounds_[1] = {mid, hi};

        for (int id = 0; id < 2; ++id)
        {
            std::lock_guard lk(mtx_[id]);
            inputs_[id] = &inp;
            done_[id]   = false;
            ready_[id]  = true;
        }
        cv_[0].notify_one();
        cv_[1].notify_one();

        for (int id = 0; id < 2; ++id)
        {
            std::unique_lock lk(mtx_[id]);
            cv_[id].wait(lk, [&] { return done_[id]; });
        }

        return {results_[0], results_[1]};
    }

private:
    void worker_loop(int id)
    {
        while (true)
        {
            std::unique_lock lk(mtx_[id]);
            cv_[id].wait(lk, [&] { return ready_[id] || shutdown_; });
            if (shutdown_) return;

            boost::uintmax_t max_iter = 200;
            results_[id] = boost::math::tools::brent_find_minima(
                [&](double y) { return squared_reprojection_error(*inputs_[id], y); },
                bounds_[id].first, bounds_[id].second,
                kBrentBits, max_iter);

            ready_[id] = false;
            done_[id]  = true;
            lk.unlock();
            cv_[id].notify_one();
        }
    }

    std::array<std::thread, 2>              workers_;
    std::array<std::mutex, 2>               mtx_;
    std::array<std::condition_variable, 2>  cv_;
    std::array<const SolveInputs *, 2>      inputs_{};
    std::array<std::pair<double, double>, 2> bounds_{};
    std::array<std::pair<double, double>, 2> results_{};
    std::array<bool, 2>                     ready_    = {false, false};
    std::array<bool, 2>                     done_     = {false, false};
    bool                                    shutdown_ = false;
};

// ---------------------------------------------------------------------------
// solve_yaw — public entry point
// ---------------------------------------------------------------------------

/**
 * Minimize plate keypoint reprojection error over yaw using Brent's method.
 * The search range is split at its midpoint and searched in parallel across
 * two threads, covering both basins of a W-shaped cost surface.
 * Images must be pre-undistorted; no distortion coefficients are accepted.
 *
 * Args:
 *   image_points  (4,2) float64 — observed pixel keypoints [BL,TL,TR,BR]
 *   position_xyz  (3,)  float64 — tvec in camera frame (metres)
 *   plate_matrix  (4,3) float64 — 3-D plate corner object points
 *   camera_matrix (3,3) float64 — intrinsic K (fx,fy,cx,cy only used)
 *   plate_pitch   float — plate pitch in world frame (radians)
 *   plate_roll    float — plate roll in world frame (radians)
 *   cam_pitch     float — camera pitch in world frame (radians)
 *   cam_roll      float — camera roll in world frame (radians)
 *   yaw_lo        float — search lower bound (default -π/2)
 *   yaw_hi        float — search upper bound (default +π/2)
 *
 * Returns:
 *   (yaw1, rms1, yaw2, rms2): best yaw and per-point RMS from each half
 */
static std::tuple<double, double, double, double> solve_yaw(
    py::array_t<double> image_points,
    py::array_t<double> position_xyz,
    py::array_t<double> plate_matrix,
    py::array_t<double> camera_matrix,
    double plate_pitch,
    double plate_roll,
    double cam_pitch,
    double cam_roll,
    double yaw_lo,
    double yaw_hi)
{
    if (!std::isfinite(yaw_lo) || !std::isfinite(yaw_hi))
        throw std::invalid_argument("yaw bounds must be finite");
    if (yaw_hi < yaw_lo)
        throw std::invalid_argument("yaw_hi must be >= yaw_lo");

    const SolveInputs inputs = unpack_inputs(
        image_points, position_xyz, plate_matrix, camera_matrix,
        plate_pitch, plate_roll, cam_pitch, cam_roll);

    double mid = 0.5 * (yaw_lo + yaw_hi);

    static BrentPool pool;

    std::array<std::pair<double, double>, 2> res;
    {
        py::gil_scoped_release release;

        bool prev_cost_set = false;
        double prev_cost   = 0.0;
        for (int i = 1; i < 20; ++i)
        {
            double yaw  = yaw_lo + i * (yaw_hi - yaw_lo) / 20;
            double cost = squared_reprojection_error(inputs, yaw);
            if (prev_cost_set && cost > prev_cost)
            {
                double next_cost = squared_reprojection_error(
                    inputs, yaw_lo + (i + 1) * (yaw_hi - yaw_lo) / 20);
                if (next_cost < cost)
                {
                    mid = yaw;
                    break;
                }
            }
            prev_cost     = cost;
            prev_cost_set = true;
        }

        res = pool.run(inputs, yaw_lo, mid, yaw_hi);
    }

    return {
        res[0].first, std::sqrt(0.25 * res[0].second),
        res[1].first, std::sqrt(0.25 * res[1].second)};
}

static std::array<double, 100> getReprojErr(
    py::array_t<double> image_points,
    py::array_t<double> position_xyz,
    py::array_t<double> plate_matrix,
    py::array_t<double> camera_matrix,
    double plate_pitch,
    double plate_roll,
    double cam_pitch,
    double cam_roll,
    double yaw_lo,
    double yaw_hi)
{
    const SolveInputs inputs = unpack_inputs(
        image_points, position_xyz, plate_matrix, camera_matrix,
        plate_pitch, plate_roll, cam_pitch, cam_roll);

    std::array<double, 100> errors;
    for (int i = 0; i < 100; ++i)
    {
        double yaw   = yaw_lo + i * (yaw_hi - yaw_lo) / 99;
        errors[i]    = std::sqrt(0.25 * squared_reprojection_error(inputs, yaw));
    }
    return errors;
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(_rm_pose_solver, m)
{
    m.doc() = "RoboMaster pose solver — dual Brent minimization via Boost.Math + Eigen";

    m.def("solve_yaw", &solve_yaw,
          py::arg("image_points"),
          py::arg("position_xyz"),
          py::arg("plate_matrix"),
          py::arg("camera_matrix"),
          py::arg("plate_pitch"),
          py::arg("plate_roll"),
          py::arg("cam_pitch"),
          py::arg("cam_roll"),
          py::arg("yaw_lo") = -M_PI / 2.0,
          py::arg("yaw_hi") = M_PI / 2.0);

    m.def("get_reproj_err", &getReprojErr,
          py::arg("image_points"),
          py::arg("position_xyz"),
          py::arg("plate_matrix"),
          py::arg("camera_matrix"),
          py::arg("plate_pitch"),
          py::arg("plate_roll"),
          py::arg("cam_pitch"),
          py::arg("cam_roll"),
          py::arg("yaw_lo") = -M_PI / 2.0,
          py::arg("yaw_hi") = M_PI / 2.0);
}