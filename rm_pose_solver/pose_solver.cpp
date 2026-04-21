/**
 * rm_pose_solver: Yaw optimization via reprojection.
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

struct SolveInputs {
    std::array<cv::Point2d, 4> observed;
    cv::Vec3d tvec;
    cv::Mat object_points;
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    double sin_pitch;
    double cos_pitch;
    double sin_roll;
    double cos_roll;
};

struct SweepResult {
    double yaw;
    double sq;
};

static constexpr double kYawIncrementRad = M_PI / 180.0;
static constexpr double kYawTailEpsilon = 1e-12;

static inline bool is_better_result(double candidate_sq, double candidate_yaw, const SweepResult& current_best)
{
    if (candidate_sq < current_best.sq)
        return true;
    if (candidate_sq > current_best.sq)
        return false;
    return candidate_yaw < current_best.yaw;
}

static SolveInputs unpack_inputs(
    py::array_t<double> image_points,
    py::array_t<double> position_xyz,
    py::array_t<double> plate_matrix,
    py::array_t<double> camera_matrix,
    py::array_t<double> dist_coeffs,
    double fixed_pitch,
    double fixed_roll)
{
    auto ip  = image_points.unchecked<2>();   // (4,2)
    auto pos = position_xyz.unchecked<1>();   // (3,)
    auto pm  = plate_matrix.unchecked<2>();   // (4,3)
    auto km  = camera_matrix.unchecked<2>();  // (3,3)
    auto dc  = dist_coeffs.unchecked<1>();    // (N,)

    if (ip.shape(0) != 4 || ip.shape(1) != 2)
        throw std::invalid_argument("image_points must be shape (4, 2)");
    if (pos.shape(0) != 3)
        throw std::invalid_argument("position_xyz must be shape (3,)");
    if (pm.shape(0) != 4 || pm.shape(1) != 3)
        throw std::invalid_argument("plate_matrix must be shape (4, 3)");
    if (km.shape(0) != 3 || km.shape(1) != 3)
        throw std::invalid_argument("camera_matrix must be shape (3, 3)");
    if (dc.shape(0) < 4)
        throw std::invalid_argument("dist_coeffs must contain at least 4 values");

    SolveInputs inputs{};

    RM_UNROLL_4
    for (int i = 0; i < 4; ++i)
        inputs.observed[i] = {ip(i, 0), ip(i, 1)};

    inputs.tvec = cv::Vec3d(pos(0), pos(1), pos(2));

    inputs.object_points = cv::Mat(4, 3, CV_64F);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j)
            inputs.object_points.at<double>(i, j) = pm(i, j);
    }

    inputs.camera_matrix = cv::Mat(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            inputs.camera_matrix.at<double>(i, j) = km(i, j);
    }

    const int n_dist = static_cast<int>(dc.shape(0));
    inputs.dist_coeffs = cv::Mat(1, n_dist, CV_64F);
    for (int i = 0; i < n_dist; ++i) {
        inputs.dist_coeffs.at<double>(0, i) = dc(i);
    }

    inputs.sin_pitch = std::sin(fixed_pitch);
    inputs.cos_pitch = std::cos(fixed_pitch);
    inputs.sin_roll = std::sin(fixed_roll);
    inputs.cos_roll = std::cos(fixed_roll);

    return inputs;
}


static inline double squared_reprojection_error(
    const SolveInputs& inputs,
    double yaw,
    std::vector<cv::Point2d>& projected)
{
    cv::Vec3d rv_std = euler_to_rvec_fast(
        yaw,
        inputs.sin_pitch,
        inputs.cos_pitch,
        inputs.sin_roll,
        inputs.cos_roll);
    cv::Vec3d rv_cv2 = standard_rvec_to_cv2(rv_std);

    cv::projectPoints(
        inputs.object_points,
        rv_cv2,
        inputs.tvec,
        inputs.camera_matrix,
        inputs.dist_coeffs,
        projected);

    double sq = 0.0;
    RM_UNROLL_4
    for (int i = 0; i < 4; ++i) {
        const double dx = projected[i].x - inputs.observed[i].x;
        const double dy = projected[i].y - inputs.observed[i].y;
        sq += dx * dx + dy * dy;
    }
    return sq;
}

static SweepResult brute_force_sweep_threaded(
    const SolveInputs& inputs,
    double yaw_lo,
    double yaw_hi,
    int requested_threads)
{
    const int full_steps = static_cast<int>(std::floor((yaw_hi - yaw_lo) / kYawIncrementRad));
    const double last_regular_yaw = yaw_lo + kYawIncrementRad * static_cast<double>(full_steps);
    const bool include_yaw_hi_tail = last_regular_yaw < (yaw_hi - kYawTailEpsilon);
    const int sample_count = full_steps + 1 + (include_yaw_hi_tail ? 1 : 0);

    const unsigned hw_threads = std::thread::hardware_concurrency();
    const int hw_cap = hw_threads > 0 ? static_cast<int>(hw_threads) : requested_threads;
    const int worker_count = std::max(1, std::min(std::min(requested_threads, sample_count), hw_cap));

    std::vector<SweepResult> local_results(
        worker_count,
        SweepResult{yaw_lo, std::numeric_limits<double>::infinity()});

    std::vector<std::thread> workers;
    workers.reserve(worker_count);

    std::exception_ptr thread_error = nullptr;
    std::mutex error_mutex;

    for (int worker_id = 0; worker_id < worker_count; ++worker_id) {
        workers.emplace_back([&, worker_id]() {
            try {
                SweepResult local_best{yaw_lo, std::numeric_limits<double>::infinity()};
                std::vector<cv::Point2d> projected(4);

                for (int sample_index = worker_id; sample_index < sample_count; sample_index += worker_count) {
                    const bool is_tail = include_yaw_hi_tail && sample_index == (sample_count - 1);
                    const double yaw = is_tail
                        ? yaw_hi
                        : (yaw_lo + kYawIncrementRad * static_cast<double>(sample_index));

                    const double sq = squared_reprojection_error(inputs, yaw, projected);
                    if (!std::isfinite(sq))
                        continue;

                    if (is_better_result(sq, yaw, local_best)) {
                        local_best.yaw = yaw;
                        local_best.sq = sq;
                    }
                }

                local_results[worker_id] = local_best;
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!thread_error)
                    thread_error = std::current_exception();
            }
        });
    }

    for (auto& worker : workers)
        worker.join();

    if (thread_error)
        std::rethrow_exception(thread_error);

    SweepResult global_best = local_results[0];
    for (int worker_id = 1; worker_id < worker_count; ++worker_id) {
        const SweepResult candidate = local_results[worker_id];
        if (is_better_result(candidate.sq, candidate.yaw, global_best))
            global_best = candidate;
    }

    return global_best;
}

// ---------------------------------------------------------------------------
// 1. solve_yaw
// ---------------------------------------------------------------------------

/**
 * Brute-force yaw sweep in fixed range
 *
 * Parameters (all numpy arrays / scalars):
 *   image_points  : (4,2) float64  – observed pixel keypoints [BL,BR,TR,TL]
 *   position_xyz  : (3,)  float64  – tvec in camera frame (x,y,z metres)
 *   plate_matrix  : (4,3) float64  – 3-D object points of plate corners
 *   camera_matrix : (3,3) float64  – intrinsic K
 *   dist_coeffs   : (N,)  float64  – distortion coefficients
 *   fixed_pitch   : double          – pitch in radians (from odometry)
 *   fixed_roll    : double          – roll  in radians (from odometry)
 *   yaw_lo        : double          – search lower bound
 *   yaw_hi        : double          – search upper bound
 *   num_threads   : int             – number of worker threads
 *
 * Returns: (optimized_yaw: double, reprojection_error: double)
 */
static std::tuple<double, double> solve_yaw(
    py::array_t<double> image_points,
    py::array_t<double> position_xyz,
    py::array_t<double> plate_matrix,
    py::array_t<double> camera_matrix,
    py::array_t<double> dist_coeffs,
    double fixed_pitch,
    double fixed_roll,
    double yaw_lo,
    double yaw_hi,
    int num_threads)
{
    if (!std::isfinite(yaw_lo) || !std::isfinite(yaw_hi))
        throw std::invalid_argument("yaw bounds must be finite");
    if (yaw_hi < yaw_lo)
        throw std::invalid_argument("yaw_hi must be greater than or equal to yaw_lo");
    if (num_threads < 1)
        throw std::invalid_argument("num_threads must be >= 1");

    const SolveInputs inputs = unpack_inputs(
        image_points,
        position_xyz,
        plate_matrix,
        camera_matrix,
        dist_coeffs,
        fixed_pitch,
        fixed_roll);

    SweepResult best;
    {
        py::gil_scoped_release release;
        best = brute_force_sweep_threaded(inputs, yaw_lo, yaw_hi, num_threads);
    }

    const double final_rms = std::sqrt(0.25 * best.sq);
    return {best.yaw, final_rms};
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(_rm_pose_solver, m)
{
    m.doc() = "RoboMaster pose solver: threaded brute-force yaw optimisation";

    m.def("solve_yaw", &solve_yaw,
        py::arg("image_points"),
        py::arg("position_xyz"),
        py::arg("plate_matrix"),
        py::arg("camera_matrix"),
        py::arg("dist_coeffs"),
        py::arg("fixed_pitch"),
        py::arg("fixed_roll"),
        py::arg("yaw_lo") = -M_PI / 2.0,
        py::arg("yaw_hi") = M_PI / 2.0,
        py::arg("num_threads") = 4,
        R"doc(
Brute-force camera-frame yaw to minimize plate keypoint reprojection error.

Args:
    image_points  (np.ndarray[float64, (4,2)]): observed pixel keypoints [BL,BR,TR,TL]
    position_xyz  (np.ndarray[float64, (3,)]): tvec in camera frame (meters)
    plate_matrix  (np.ndarray[float64, (4,3)]): 3-D plate corner object points
    camera_matrix (np.ndarray[float64, (3,3)]): intrinsic K matrix
    dist_coeffs   (np.ndarray[float64, (N,)]): distortion coefficients
    fixed_pitch   (float): pitch in radians (from odometry/IMU)
    fixed_roll    (float): roll in radians (from odometry/IMU)
    yaw_lo        (float): search lower bound, default -pi/2
    yaw_hi        (float): search upper bound, default +pi/2
                  sampled every 1 degree, with yaw_hi always included
    num_threads   (int): number of sweep worker threads, default 4

Returns:
    tuple[float, float]: (best_yaw_rad, reprojection_error_pixels_rms)
)doc");
}
