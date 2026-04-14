/**
 * rm_pose_solver: Yaw optimization via reprojection + light bar refinement.
 *
 * Exposes two functions to Python via pybind11:
 *   1. solve_yaw(...)  - L-BFGS-B over yaw to minimize reprojection error
 *   2. refine_lights(...) - Otsu threshold on R/B channel, find closest contour
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
#include <limits>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Rodrigues rotation vector from ZYX Euler angles (yaw, pitch, roll) in radians.
/// Convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
static cv::Vec3d euler_to_rvec(double yaw, double pitch, double roll)
{
    cv::Mat Rx = (cv::Mat_<double>(3, 3) <<
        1,          0,           0,
        0,  std::cos(roll), -std::sin(roll),
        0,  std::sin(roll),  std::cos(roll));

    cv::Mat Ry = (cv::Mat_<double>(3, 3) <<
         std::cos(pitch), 0, std::sin(pitch),
         0,               1, 0,
        -std::sin(pitch), 0, std::cos(pitch));

    cv::Mat Rz = (cv::Mat_<double>(3, 3) <<
        std::cos(yaw), -std::sin(yaw), 0,
        std::sin(yaw),  std::cos(yaw), 0,
        0,              0,             1);

    cv::Mat R = Rz * Ry * Rx;
    cv::Mat rvec;
    cv::Rodrigues(R, rvec);
    return cv::Vec3d(rvec);
}

/// Reproject 4 object points and return RMS pixel error.
static double reprojection_rms(
    const cv::Mat& object_pts,    // (4,3) CV_64F
    const cv::Vec3d& rvec,
    const cv::Vec3d& tvec,
    const cv::Mat& K,
    const cv::Mat& dist,
    const std::vector<cv::Point2d>& observed)
{
    std::vector<cv::Point2d> projected;
    cv::projectPoints(object_pts, rvec, tvec, K, dist, projected);
    double sq = 0.0;
    for (int i = 0; i < 4; ++i) {
        double dx = projected[i].x - observed[i].x;
        double dy = projected[i].y - observed[i].y;
        sq += dx * dx + dy * dy;
    }
    return std::sqrt(0.25 * sq);
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
 *   max_iter      : int             – L-BFGS-B iterations, default 50
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
        throw std::invalid_argument("image_points must be (4,2)");
    if (pos.shape(0) != 3)
        throw std::invalid_argument("position_xyz must be (3,)");
    if (pm.shape(0) != 4 || pm.shape(1) != 3)
        throw std::invalid_argument("plate_matrix must be (4,3)");

    std::vector<cv::Point2d> observed(4);
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
    for (int i = 0; i < n_dist; ++i) dist.at<double>(0, i) = dc(i);

    // --- golden-section search (exact 1-D, no external optimizer needed) ---
    // Faster and dependency-free compared to L-BFGS-B for a 1-D bounded problem.
    const double phi = (std::sqrt(5.0) - 1.0) / 2.0;
    double a = yaw_lo, b = yaw_hi;
    double c = b - phi * (b - a);
    double d = a + phi * (b - a);

    auto loss = [&](double y) -> double {
        cv::Vec3d rv = euler_to_rvec(y, fixed_pitch, fixed_roll);
        return reprojection_rms(obj, rv, tvec, K, dist, observed);
    };

    int iters = 0;
    while (std::abs(b - a) > 1e-7 && iters < max_iter * 10) {
        if (loss(c) < loss(d)) {
            b = d; d = c;
            c = b - phi * (b - a);
        } else {
            a = c; c = d;
            d = a + phi * (b - a);
        }
        ++iters;
    }

    double best_yaw = (a + b) / 2.0;
    cv::Vec3d best_rv = euler_to_rvec(best_yaw, fixed_pitch, fixed_roll);
    double error = reprojection_rms(obj, best_rv, tvec, K, dist, observed);

    return {best_yaw, error};
}

// ---------------------------------------------------------------------------
// 2. refine_lights
// ---------------------------------------------------------------------------

/**
 * Refine a pair of light-bar centre estimates using Otsu threshold on the
 * colour channel corresponding to the enemy colour.
 *
 * Parameters:
 *   bgr_image     : (H,W,3) uint8  – full colour frame (BGR)
 *   guess_pts     : (N,2)   float64 – initial guesses for light centres [px]
 *   enemy_is_blue : bool            – True → threshold blue channel; else red
 *   search_radius : int             – pixel radius around each guess to crop
 *
 * Returns: list of (x, y) float pairs – refined centres (or original if
 *          no valid contour found within search_radius).
 */
static std::vector<std::array<double, 2>> refine_lights(
    py::array_t<uint8_t> bgr_image,
    py::array_t<double>  guess_pts,
    bool enemy_is_blue = true,
    int  search_radius = 30)
{
    auto img_buf = bgr_image.unchecked<3>();  // (H,W,3)
    auto gp      = guess_pts.unchecked<2>(); // (N,2)

    int H = (int)img_buf.shape(0);
    int W = (int)img_buf.shape(1);
    int N = (int)gp.shape(0);

    // Wrap numpy buffer into cv::Mat (no copy)
    cv::Mat bgr(H, W, CV_8UC3, (void*)bgr_image.data());

    // Select channel: blue=0, red=2
    int ch = enemy_is_blue ? 0 : 2;

    std::vector<std::array<double, 2>> results(N);

    for (int idx = 0; idx < N; ++idx) {
        double gx = gp(idx, 0);
        double gy = gp(idx, 1);
        results[idx] = {gx, gy};  // default: return guess

        // Crop ROI
        int x0 = std::max(0, (int)(gx - search_radius));
        int y0 = std::max(0, (int)(gy - search_radius));
        int x1 = std::min(W, (int)(gx + search_radius));
        int y1 = std::min(H, (int)(gy + search_radius));
        if (x1 <= x0 || y1 <= y0) continue;

        cv::Rect roi_rect(x0, y0, x1 - x0, y1 - y0);
        cv::Mat roi = bgr(roi_rect);

        // Extract single channel
        std::vector<cv::Mat> channels(3);
        cv::split(roi, channels);
        cv::Mat gray = channels[ch];

        // Otsu threshold
        cv::Mat binary;
        double thresh = cv::threshold(gray, binary, 0, 255,
                                      cv::THRESH_BINARY | cv::THRESH_OTSU);
        if (thresh < 1.0) continue; // degenerate

        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL,
                         cv::CHAIN_APPROX_SIMPLE);

        // Pick contour whose centre is closest to the guess
        double best_dist = std::numeric_limits<double>::max();
        cv::Point2f best_center;
        bool found = false;

        cv::Point2f local_guess((float)(gx - x0), (float)(gy - y0));

        for (const auto& cnt : contours) {
            if (cnt.size() < 5) continue;  // need enough points

            cv::Moments m = cv::moments(cnt);
            if (m.m00 < 1.0) continue;

            cv::Point2f c((float)(m.m10 / m.m00), (float)(m.m01 / m.m00));
            double dist = cv::norm(c - local_guess);

            if (dist < best_dist) {
                best_dist = dist;
                best_center = c;
                found = true;
            }
        }

        if (found && best_dist < (double)search_radius) {
            results[idx] = {best_center.x + x0, best_center.y + y0};
        }
    }

    return results;
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(_rm_pose_solver, m)
{
    m.doc() = "RoboMaster pose solver: yaw optimisation + light-bar refinement.";

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

    m.def("refine_lights", &refine_lights,
        py::arg("bgr_image"),
        py::arg("guess_pts"),
        py::arg("enemy_is_blue") = true,
        py::arg("search_radius") = 30,
        R"doc(
Refine light-bar centre estimates using Otsu threshold on the target colour channel.

Crops a square ROI of side 2*search_radius around each guess, applies Otsu
thresholding on the blue (enemy_is_blue=True) or red channel, finds contours,
and returns the centroid of the closest valid contour.

Args:
    bgr_image     (np.ndarray[uint8, (H,W,3)]): full BGR frame
    guess_pts     (np.ndarray[float64, (N,2)]): initial light-centre guesses in pixels
    enemy_is_blue (bool): True → threshold blue channel, False → red channel
    search_radius (int):  pixel radius of search window around each guess

Returns:
    list[tuple[float, float]]: refined (x, y) centres; falls back to guess if
                               no suitable contour is found.
)doc");
}
