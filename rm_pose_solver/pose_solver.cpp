/**
 * rm_pose_solver: Yaw optimization via reprojection.
 *
 * Exposes one function to Python via pybind11:
 *   1. solve_yaw(...)  - bounded 1-D search over yaw
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <opencv2/opencv.hpp>
#include <cmath>
#include <array>
#include <tuple>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

#if defined(__clang__)
#define RM_UNROLL_4 _Pragma("clang loop unroll_count(4)")
#elif defined(__GNUC__)
#define RM_UNROLL_4 _Pragma("GCC unroll 4")
#else
#define RM_UNROLL_4
#endif

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Rodrigues rotation vector from ZYX Euler angles (yaw, pitch, roll) in radians.
/// Convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
static inline cv::Vec3d euler_to_rvec_fast(
    double yaw,
    double sin_pitch,
    double cos_pitch,
    double sin_roll,
    double cos_roll)
{
    const double cy = std::cos(yaw);
    const double sy = std::sin(yaw);

    const cv::Matx33d R(
        cy * cos_pitch,
        cy * sin_pitch * sin_roll - sy * cos_roll,
        cy * sin_pitch * cos_roll + sy * sin_roll,

        sy * cos_pitch,
        sy * sin_pitch * sin_roll + cy * cos_roll,
        sy * sin_pitch * cos_roll - cy * sin_roll,

        -sin_pitch,
        cos_pitch * sin_roll,
        cos_pitch * cos_roll);

    cv::Vec3d rvec;
    cv::Rodrigues(R, rvec);
    return rvec;
}

/// Convert an axis-angle vector from Otto standard coordinates to OpenCV coordinates.
/// Matches Orientation.to_cv2_coords(): (-y, -z, x)
static inline cv::Vec3d standard_rvec_to_cv2(const cv::Vec3d& rvec_std)
{
    return cv::Vec3d(-rvec_std[1], -rvec_std[2], rvec_std[0]);
}

// ---------------------------------------------------------------------------
// 1. solve_yaw
// ---------------------------------------------------------------------------

/**
 * Optimise yaw (camera-frame) to minimise reprojection error.
 *
 * Parameters (all numpy arrays / scalars):
 *   image_points  : (4,2) float64  – observed pixel keypoints [BL,BR,TR,TL]
 *   position_xyz  : (3,)  float64  – tvec in camera frame (x,y,z metres)
 *   plate_matrix  : (4,3) float64  – 3-D object points of plate corners
 *   camera_matrix : (3,3) float64  – intrinsic K
 *   dist_coeffs   : (N,)  float64  – distortion coefficients
 *   fixed_pitch   : double          – pitch in radians (from odometry)
 *   fixed_roll    : double          – roll  in radians (from odometry)
 *   yaw_lo        : double          – search lower bound (rad), default -π
 *   yaw_hi        : double          – search upper bound (rad), default  π
 *   max_iter      : int             – max Brent iterations, default 50
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
    double yaw_lo = -M_PI,
    double yaw_hi =  M_PI,
    int    max_iter = 50)
{
    // --- unpack numpy → cv::Mat / std::vector ---
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
    if (!(yaw_lo < yaw_hi))
        throw std::invalid_argument("yaw_lo must be less than yaw_hi");
    if (max_iter <= 0)
        throw std::invalid_argument("max_iter must be positive");

    std::array<cv::Point2d, 4> observed{};
    RM_UNROLL_4
    for (int i = 0; i < 4; ++i)
        observed[i] = {ip(i, 0), ip(i, 1)};

    cv::Vec3d tvec(pos(0), pos(1), pos(2));

    cv::Mat obj(4, 3, CV_64F);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 3; ++j)
            obj.at<double>(i, j) = pm(i, j);

    cv::Mat K(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            K.at<double>(i, j) = km(i, j);

    int n_dist = (int)dc.shape(0);
    cv::Mat dist(1, n_dist, CV_64F);
    for (int i = 0; i < n_dist; ++i)
        dist.at<double>(0, i) = dc(i);

    const double sin_pitch = std::sin(fixed_pitch);
    const double cos_pitch = std::cos(fixed_pitch);
    const double sin_roll = std::sin(fixed_roll);
    const double cos_roll = std::cos(fixed_roll);

    std::vector<cv::Point2d> projected(4);

    // --- loss: squared reprojection error (no sqrt) ---
    auto loss = [&](double y) -> double {
        cv::Vec3d rv_std = euler_to_rvec_fast(y, sin_pitch, cos_pitch, sin_roll, cos_roll);
        cv::Vec3d rv_cv2 = standard_rvec_to_cv2(rv_std);
        cv::projectPoints(obj, rv_cv2, tvec, K, dist, projected);

        double sq = 0.0;
        RM_UNROLL_4
        for (int i = 0; i < 4; ++i) {
            double dx = projected[i].x - observed[i].x;
            double dy = projected[i].y - observed[i].y;
            sq += dx * dx + dy * dy;
        }
        return sq;
    };

    // -----------------------------------------------------------------------
    // Brent’s method (bounded)
    // Based on classic implementation (Numerical Recipes style)
    // -----------------------------------------------------------------------

    const double tol = 1e-7;
    const double golden = 0.3819660; // (3 - sqrt(5)) / 2

    double a = yaw_lo;
    double b = yaw_hi;

    double x = a + 0.5 * (b - a);
    double w = x;
    double v = x;

    double fx = loss(x);
    double fw = fx;
    double fv = fx;

    double d = 0.0;
    double e = 0.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        double m = 0.5 * (a + b);
        double tol1 = tol * std::abs(x) + 1e-12;
        double tol2 = 2.0 * tol1;

        if (std::abs(x - m) <= (tol2 - 0.5 * (b - a)))
            break;

        double p = 0.0, q = 0.0, r = 0.0;

        if (std::abs(e) > tol1) {
            // Attempt parabolic fit
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);

            if (q > 0) p = -p;
            q = std::abs(q);

            if (std::abs(p) < std::abs(0.5 * q * e) &&
                p > q * (a - x) &&
                p < q * (b - x)) {
                // Parabolic step
                d = p / q;
                double u = x + d;

                if (u - a < tol2 || b - u < tol2)
                    d = (x < m) ? tol1 : -tol1;
            } else {
                // Golden section
                e = (x < m) ? (b - x) : (a - x);
                d = golden * e;
            }
        } else {
            // Golden section
            e = (x < m) ? (b - x) : (a - x);
            d = golden * e;
        }

        double u = (std::abs(d) >= tol1) ? x + d : x + (d > 0 ? tol1 : -tol1);
        double fu = loss(u);

        if (fu <= fx) {
            if (u < x) b = x; else a = x;
            v = w; fv = fw;
            w = x; fw = fx;
            x = u; fx = fu;
        } else {
            if (u < x) a = u; else b = u;
            if (fu <= fw || w == x) {
                v = w; fv = fw;
                w = u; fw = fu;
            } else if (fu <= fv || v == x || v == w) {
                v = u; fv = fu;
            }
        }

        e = d;
    }

    double best_yaw = x;

    // Return RMS for consistency with your Python side (optional)
    double final_sq = fx;
    double final_rms = std::sqrt(0.25 * final_sq);

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
        py::arg("max_iter") = 50,
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
    max_iter      (int):   maximum optimiser iterations, default 50

Returns:
    tuple[float, float]: (optimized_yaw_rad, reprojection_error_pixels)
)doc");

}
