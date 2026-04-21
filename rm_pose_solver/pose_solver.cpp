/**
 * rm_pose_solver: Yaw optimization via reprojection.
 *
 * Exposes two functions to Python via pybind11:
 *   1. solve_yaw(...)             - bounded 1-D search over yaw
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <opencv2/opencv.hpp>
#include <cmath>
#include <array>
#include <limits>
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
    std::array<cv::Vec3d, 4> object_points4;
    cv::Vec3d tvec;
    cv::Mat object_points;
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    std::array<double, 14> dist_coeff_values;
    int n_dist;
    double fx;
    double fy;
    double cx;
    double cy;
    double skew;
    double sin_pitch;
    double cos_pitch;
    double sin_roll;
    double cos_roll;
};

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
        inputs.object_points4[i] = cv::Vec3d(pm(i, 0), pm(i, 1), pm(i, 2));
    }

    inputs.camera_matrix = cv::Mat(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            inputs.camera_matrix.at<double>(i, j) = km(i, j);
    }

    inputs.fx = km(0, 0);
    inputs.fy = km(1, 1);
    inputs.cx = km(0, 2);
    inputs.cy = km(1, 2);
    inputs.skew = km(0, 1);

    inputs.dist_coeff_values.fill(0.0);

    inputs.n_dist = static_cast<int>(dc.shape(0));
    inputs.dist_coeffs = cv::Mat(1, inputs.n_dist, CV_64F);
    for (int i = 0; i < inputs.n_dist; ++i) {
        inputs.dist_coeffs.at<double>(0, i) = dc(i);
        if (i < static_cast<int>(inputs.dist_coeff_values.size()))
            inputs.dist_coeff_values[i] = dc(i);
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
 *   yaw_lo        : double          – search lower bound, default -80 deg
 *   yaw_hi        : double          – search upper bound, default +80 deg
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
    float yaw_lo,
    float yaw_hi)
{

    const SolveInputs inputs = unpack_inputs(
        image_points,
        position_xyz,
        plate_matrix,
        camera_matrix,
        dist_coeffs,
        fixed_pitch,
        fixed_roll);

    double best_yaw = yaw_lo;
    double best_sq = std::numeric_limits<double>::infinity();

    double one_deg_in_radians = M_PI / 180.0;

    std::vector<cv::Point2d> projected(4);
    for(double curr_yaw = yaw_lo; curr_yaw <= yaw_hi; curr_yaw += one_deg_in_radians) {
        const double sq = squared_reprojection_error(inputs, curr_yaw, projected);
        if (sq < best_sq) {
            best_sq = sq;
            best_yaw = curr_yaw;
        }
    }

    const double final_rms = std::sqrt(0.25 * best_sq);
    return {best_yaw, final_rms};
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(_rm_pose_solver, m)
{
    m.doc() = "RoboMaster pose solver: yaw optimisation";

    m.def("solve_yaw", &solve_yaw,
        py::arg("image_points"),
        py::arg("position_xyz"),
        py::arg("plate_matrix"),
        py::arg("camera_matrix"),
        py::arg("dist_coeffs"),
        py::arg("fixed_pitch"),
        py::arg("fixed_roll"),
        py::arg("yaw_lo")   = -M_PI / 2.0,
        py::arg("yaw_hi")   =  M_PI / 2.0,
        R"doc(
Optimise camera-frame yaw to minimise plate keypoint reprojection error.

Args:
    image_points  (np.ndarray[float64, (4,2)]): observed pixel keypoints [BL,BR,TR,TL]
    position_xyz  (np.ndarray[float64, (3,)]): tvec in camera frame (metres)
    plate_matrix  (np.ndarray[float64, (4,3)]): 3-D plate corner object points
    camera_matrix (np.ndarray[float64, (3,3)]): intrinsic K matrix
    dist_coeffs   (np.ndarray[float64, (N,)]): distortion coefficients
    fixed_pitch   (float): pitch in radians (from odometry/IMU)
    fixed_roll    (float): roll  in radians (from odometry/IMU)
    yaw_lo        (float): search lower bound, default -π/2
    yaw_hi        (float): search upper bound, default  π/2

Returns:
    tuple[float, float]: (optimized_yaw_rad, reprojection_error_pixels)
)doc");
}
