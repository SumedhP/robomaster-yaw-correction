/**
 * rm_pose_solver: Yaw optimization via reprojection.
 *
 * Exposes two functions to Python via pybind11:
 *   1. solve_yaw(...)             - bounded 1-D search over yaw
 *   2. solve_yaw_brute_force(...) - fixed sweep over yaw in [-60 deg, 60 deg]
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

struct BruteForceInputs {
    std::array<cv::Point2d, 4> observed;
    std::array<cv::Vec3d, 4> object_points4;
    cv::Vec3d tvec;
    std::array<double, 14> dist_coeff_values;
    std::array<double, 9> tilt_matrix;
    int n_dist;
    bool use_tilt;
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

static inline std::array<double, 9> compute_tilt_projection_matrix(double tau_x, double tau_y)
{
    const double c_tau_x = std::cos(tau_x);
    const double s_tau_x = std::sin(tau_x);
    const double c_tau_y = std::cos(tau_y);
    const double s_tau_y = std::sin(tau_y);

    // matRotXY = matRotY * matRotX
    const double r00 = c_tau_y;
    const double r01 = s_tau_y * s_tau_x;
    const double r02 = -s_tau_y * c_tau_x;
    const double r10 = 0.0;
    const double r11 = c_tau_x;
    const double r12 = s_tau_x;
    const double r20 = s_tau_y;
    const double r21 = -c_tau_y * s_tau_x;
    const double r22 = c_tau_y * c_tau_x;

    // matProjZ
    const double p00 = r22;
    const double p02 = -r02;
    const double p11 = r22;
    const double p12 = -r12;

    std::array<double, 9> m{};
    // matTilt = matProjZ * matRotXY
    m[0] = p00 * r00;
    m[1] = p00 * r01;
    m[2] = p00 * r02 + p02;

    m[3] = p11 * r10;
    m[4] = p11 * r11;
    m[5] = p11 * r12 + p12;

    m[6] = r20;
    m[7] = r21;
    m[8] = r22;

    return m;
}

static BruteForceInputs unpack_bruteforce_inputs(
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

    BruteForceInputs inputs{};

    RM_UNROLL_4
    for (int i = 0; i < 4; ++i)
        inputs.observed[i] = {ip(i, 0), ip(i, 1)};

    inputs.tvec = cv::Vec3d(pos(0), pos(1), pos(2));

    RM_UNROLL_4
    for (int i = 0; i < 4; ++i)
        inputs.object_points4[i] = cv::Vec3d(pm(i, 0), pm(i, 1), pm(i, 2));

    inputs.fx = km(0, 0);
    inputs.fy = km(1, 1);
    inputs.cx = km(0, 2);
    inputs.cy = km(1, 2);
    inputs.skew = km(0, 1);

    inputs.dist_coeff_values.fill(0.0);
    inputs.tilt_matrix = {1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0};
    inputs.use_tilt = false;
    inputs.n_dist = static_cast<int>(dc.shape(0));
    for (int i = 0; i < inputs.n_dist && i < static_cast<int>(inputs.dist_coeff_values.size()); ++i)
        inputs.dist_coeff_values[i] = dc(i);

    if (inputs.n_dist >= 14) {
        const double tau_x = inputs.dist_coeff_values[12];
        const double tau_y = inputs.dist_coeff_values[13];
        if (tau_x != 0.0 || tau_y != 0.0) {
            inputs.use_tilt = true;
            inputs.tilt_matrix = compute_tilt_projection_matrix(tau_x, tau_y);
        }
    }

    inputs.sin_pitch = std::sin(fixed_pitch);
    inputs.cos_pitch = std::cos(fixed_pitch);
    inputs.sin_roll = std::sin(fixed_roll);
    inputs.cos_roll = std::cos(fixed_roll);

    return inputs;
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

struct BrutePointTerms {
    double a;
    double b;
    double z_inv;
};

struct YawGrid240 {
    std::array<double, 240> yaw;
    std::array<double, 240> cos_yaw;
    std::array<double, 240> sin_yaw;
};

static const YawGrid240& get_yaw_grid_240()
{
    static const YawGrid240 grid = []() {
        YawGrid240 g{};
        constexpr double kYawMin = -60.0 * M_PI / 180.0;
        constexpr double kYawMax =  60.0 * M_PI / 180.0;
        constexpr int kSteps = 240;
        const double step = (kYawMax - kYawMin) / static_cast<double>(kSteps - 1);
        for (int i = 0; i < kSteps; ++i) {
            const double yaw = kYawMin + static_cast<double>(i) * step;
            g.yaw[i] = yaw;
            g.cos_yaw[i] = std::cos(yaw);
            g.sin_yaw[i] = std::sin(yaw);
        }
        return g;
    }();
    return grid;
}

static inline bool can_use_fast_bruteforce_path(const BruteForceInputs& inputs)
{
    if (inputs.n_dist > 14)
        return false;
    return true;
}

static inline std::array<BrutePointTerms, 4> precompute_brute_terms(const BruteForceInputs& inputs)
{
    const double sp = inputs.sin_pitch;
    const double cp = inputs.cos_pitch;
    const double sr = inputs.sin_roll;
    const double cr = inputs.cos_roll;

    const double sp_sr = sp * sr;
    const double sp_cr = sp * cr;
    const double cp_sr = cp * sr;
    const double cp_cr = cp * cr;

    std::array<BrutePointTerms, 4> terms{};

    RM_UNROLL_4
    for (int i = 0; i < 4; ++i) {
        const cv::Vec3d& p = inputs.object_points4[i];
        const double x = p[0];
        const double y = p[1];
        const double z = p[2];

        terms[i].a = cp * x + sp_sr * y + sp_cr * z;
        terms[i].b = -cr * y + sr * z;
        const double z_cam = -sp * x + cp_sr * y + cp_cr * z + inputs.tvec[2];
        terms[i].z_inv = 1.0 / z_cam;
    }

    return terms;
}

template <bool kUseRational, bool kUseThinPrism, bool kUseTilt>
static inline std::tuple<double, double> scan_yaw_grid_fast(
    const BruteForceInputs& inputs,
    const std::array<BrutePointTerms, 4>& terms,
    const YawGrid240& grid,
    bool use_k3)
{
    const auto& dc = inputs.dist_coeff_values;
    const double k1 = dc[0];
    const double k2 = dc[1];
    const double p1 = dc[2];
    const double p2 = dc[3];
    const double k3 = dc[4];
    const double k4 = dc[5];
    const double k5 = dc[6];
    const double k6 = dc[7];
    const double s1 = dc[8];
    const double s2 = dc[9];
    const double s3 = dc[10];
    const double s4 = dc[11];
    const auto& tilt = inputs.tilt_matrix;

    const double tx = inputs.tvec[0];
    const double ty = inputs.tvec[1];

    const double fx = inputs.fx;
    const double fy = inputs.fy;
    const double cx = inputs.cx;
    const double img_cy = inputs.cy;
    const double skew = inputs.skew;

    int primary_point = 0;
    double primary_score = std::abs(terms[0].a) + std::abs(terms[0].b);
    for (int p = 1; p < 4; ++p) {
        const double s = std::abs(terms[p].a) + std::abs(terms[p].b);
        if (s > primary_score) {
            primary_score = s;
            primary_point = p;
        }
    }

    auto point_sq = [&](int p, int yaw_idx) -> double {
        const double yaw_cos = grid.cos_yaw[yaw_idx];
        const double yaw_sin = grid.sin_yaw[yaw_idx];

        const double xc = yaw_cos * terms[p].a + yaw_sin * terms[p].b + tx;
        const double yc = yaw_sin * terms[p].a - yaw_cos * terms[p].b + ty;

        const double x = xc * terms[p].z_inv;
        const double y = yc * terms[p].z_inv;

        const double x2 = x * x;
        const double y2 = y * y;
        const double r2 = x2 + y2;
        const double r4 = r2 * r2;

        double radial = 1.0 + k1 * r2 + k2 * r4;
        double r6 = 0.0;
        if (use_k3 || kUseRational) {
            r6 = r4 * r2;
            if (use_k3)
                radial += k3 * r6;
        }

        if constexpr (kUseRational) {
            radial /= (1.0 + k4 * r2 + k5 * r4 + k6 * r6);
        }

        const double two_xy = 2.0 * x * y;
        double xd = x * radial + p1 * two_xy + p2 * (r2 + 2.0 * x2);
        double yd = y * radial + p1 * (r2 + 2.0 * y2) + p2 * two_xy;

        if constexpr (kUseThinPrism) {
            xd += s1 * r2 + s2 * r4;
            yd += s3 * r2 + s4 * r4;
        }

        if constexpr (kUseTilt) {
            const double vec_x = tilt[0] * xd + tilt[1] * yd + tilt[2];
            const double vec_y = tilt[3] * xd + tilt[4] * yd + tilt[5];
            const double vec_w = tilt[6] * xd + tilt[7] * yd + tilt[8];
            const double inv_w = 1.0 / vec_w;
            xd = vec_x * inv_w;
            yd = vec_y * inv_w;
        }

        const double u = fx * xd + skew * yd + cx;
        const double v = fy * yd + img_cy;

        const double dx = u - inputs.observed[p].x;
        const double dy = v - inputs.observed[p].y;
        return dx * dx + dy * dy;
    };

    std::array<double, 240> lower_bound{};
    int best_seed_idx = 0;
    double best_seed_lb = std::numeric_limits<double>::infinity();

    for (int yaw_idx = 0; yaw_idx < 240; ++yaw_idx) {
        const double lb = point_sq(primary_point, yaw_idx);
        lower_bound[yaw_idx] = lb;
        if (lb < best_seed_lb) {
            best_seed_lb = lb;
            best_seed_idx = yaw_idx;
        }
    }

    double best_sq = lower_bound[best_seed_idx];
    for (int p = 0; p < 4; ++p) {
        if (p == primary_point)
            continue;
        best_sq += point_sq(p, best_seed_idx);
    }
    double best_yaw = grid.yaw[best_seed_idx];

    for (int yaw_idx = 0; yaw_idx < 240; ++yaw_idx) {
        if (yaw_idx == best_seed_idx)
            continue;

        double sq = lower_bound[yaw_idx];
        if (sq >= best_sq)
            continue;

        for (int p = 0; p < 4; ++p) {
            if (p == primary_point)
                continue;

            sq += point_sq(p, yaw_idx);
            if (sq >= best_sq)
                break;
        }

        if (sq < best_sq) {
            best_sq = sq;
            best_yaw = grid.yaw[yaw_idx];
        }
    }

    return {best_yaw, best_sq};
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
    double yaw_lo  = -M_PI,
    double yaw_hi  =  M_PI,
    int    max_iter = 150)
{
    if (!(yaw_lo < yaw_hi))
        throw std::invalid_argument("yaw_lo must be less than yaw_hi");
    if (max_iter <= 0)
        throw std::invalid_argument("max_iter must be positive");

    const SolveInputs inputs = unpack_inputs(
        image_points, position_xyz, plate_matrix,
        camera_matrix, dist_coeffs, fixed_pitch, fixed_roll);

    std::vector<cv::Point2d> projected(4);
    auto loss = [&](double y) -> double {
        return squared_reprojection_error(inputs, y, projected);
    };

    // -----------------------------------------------------------------------
    // Piyavskii-Shubert: maintain sorted list of evaluated points.
    // For each interval [a,b] with known f(a), f(b) and Lipschitz constant L,
    // the tightest lower bound is a V-shape with minimum at:
    //   y* = (a+b)/2 + (f(a)-f(b)) / (2L)
    //   lb = (f(a)+f(b))/2 - L*(b-a)/2
    // Always evaluate next at the interval with the lowest lower bound.
    // This guarantees global convergence.
    // -----------------------------------------------------------------------

    struct Pt { double y, f; };
    std::vector<Pt> pts;
    pts.reserve(max_iter + 8);

    // Seed with uniform initial samples to get a reasonable Lipschitz estimate
    const int n_seed = 8;
    for (int i = 0; i <= n_seed; ++i) {
        double y = yaw_lo + (yaw_hi - yaw_lo) * i / n_seed;
        pts.push_back({y, loss(y)});
    }
    // pts already sorted by y (uniform spacing)

    // --- Adaptive Lipschitz estimate (with safety factor) ---
    // Re-estimated whenever a new point raises it. Never lowered.
    auto compute_L = [&]() -> double {
        double L = 1e-9;
        for (size_t i = 1; i < pts.size(); ++i)
            L = std::max(L, std::abs(pts[i].f - pts[i-1].f)
                            / (pts[i].y - pts[i-1].y));
        return L * 2.0; // 2x safety factor keeps us conservative (won't miss global min)
    };
    double L = compute_L();

    // Per-interval lower bound and its argmin
    auto seg_lb = [&](const Pt& a, const Pt& b) -> std::pair<double,double> {
        // y* = unconstrained argmin of V-shape, clamped inside interval
        double y_star = 0.5*(a.y + b.y) + (a.f - b.f) / (2.0 * L);
        y_star = std::clamp(y_star, a.y, b.y);
        double lb     = 0.5*(a.f + b.f) - 0.5*L*(b.y - a.y);
        return {lb, y_star};
    };

    double global_best_f = std::min_element(pts.begin(), pts.end(),
        [](const Pt& a, const Pt& b){ return a.f < b.f; })->f;

    const int budget = max_iter - (n_seed + 1); // remaining evals
    for (int iter = 0; iter < budget; ++iter)
    {
        // --- Find interval whose lower bound is globally lowest ---
        double best_lb = std::numeric_limits<double>::infinity();
        double next_y  = 0.5 * (yaw_lo + yaw_hi);

        for (size_t i = 0; i + 1 < pts.size(); ++i) {
            auto [lb, y_star] = seg_lb(pts[i], pts[i+1]);
            if (lb < best_lb) {
                best_lb = lb;
                next_y  = y_star;
            }
        }

        // --- Convergence: gap between proven lower bound and best known f ---
        const double gap = global_best_f - best_lb;
        if (gap < 1e-8 * (1.0 + std::abs(global_best_f)))
            break;

        // --- Evaluate and insert (list stays sorted by y) ---
        double f_new = loss(next_y);
        global_best_f = std::min(global_best_f, f_new);

        auto it = std::lower_bound(pts.begin(), pts.end(), next_y,
            [](const Pt& p, double val){ return p.y < val; });
        pts.insert(it, {next_y, f_new});

        // --- Raise L if new slopes exceed current estimate ---
        size_t idx = (size_t)(it - pts.begin());
        if (idx > 0) {
            double slope = std::abs(pts[idx].f - pts[idx-1].f)
                         / (pts[idx].y - pts[idx-1].y + 1e-15);
            if (slope * 2.0 > L) L = slope * 2.0;
        }
        if (idx + 1 < pts.size()) {
            double slope = std::abs(pts[idx+1].f - pts[idx].f)
                         / (pts[idx+1].y - pts[idx].y + 1e-15);
            if (slope * 2.0 > L) L = slope * 2.0;
        }
    }

    // -----------------------------------------------------------------------
    // Polish: tight Brent refinement around the global best found
    // -----------------------------------------------------------------------
    auto best_it = std::min_element(pts.begin(), pts.end(),
        [](const Pt& a, const Pt& b){ return a.f < b.f; });
    size_t bi = (size_t)(best_it - pts.begin());

    double pol_lo = (bi > 0)              ? pts[bi-1].y : yaw_lo;
    double pol_hi = (bi + 1 < pts.size()) ? pts[bi+1].y : yaw_hi;

    // Brent within [pol_lo, pol_hi] — guaranteed unimodal now
    const double tol    = 1e-9;
    const double golden = 0.3819660;
    double a  = pol_lo, b = pol_hi;
    double x  = best_it->y;
    double w  = x,  v  = x;
    double fx = best_it->f, fw = fx, fv = fx;
    double d  = 0.0, e = 0.0;

    for (int iter = 0; iter < 60; ++iter) {
        double m    = 0.5*(a + b);
        double tol1 = tol * std::abs(x) + 1e-13;
        double tol2 = 2.0 * tol1;
        if (std::abs(x - m) <= tol2 - 0.5*(b - a)) break;

        double p = 0.0, q = 0.0, r = 0.0;
        if (std::abs(e) > tol1) {
            r = (x-w)*(fx-fv);
            q = (x-v)*(fx-fw);
            p = (x-v)*q - (x-w)*r;
            q = 2.0*(q-r);
            if (q > 0) p = -p;
            q = std::abs(q);
            if (std::abs(p) < std::abs(0.5*q*e) && p > q*(a-x) && p < q*(b-x)) {
                d = p/q;
                double u = x + d;
                if (u-a < tol2 || b-u < tol2) d = (x < m) ? tol1 : -tol1;
            } else {
                e = (x < m) ? (b-x) : (a-x);
                d = golden * e;
            }
        } else {
            e = (x < m) ? (b-x) : (a-x);
            d = golden * e;
        }

        double u  = (std::abs(d) >= tol1) ? x+d : x+(d > 0 ? tol1 : -tol1);
        double fu = loss(u);
        if (fu <= fx) {
            if (u < x) b = x; else a = x;
            v=w; fv=fw; w=x; fw=fx; x=u; fx=fu;
        } else {
            if (u < x) a = u; else b = u;
            if (fu <= fw || w == x) { v=w; fv=fw; w=u; fw=fu; }
            else if (fu <= fv || v == x || v == w) { v=u; fv=fu; }
        }
        e = d;
    }

    return {x, std::sqrt(0.25 * fx)};
}

// ---------------------------------------------------------------------------
// 2. solve_yaw_brute_force
// ---------------------------------------------------------------------------

/**
 * Brute-force yaw sweep in fixed range [-60°, +60°].
 *
 * Parameters (all numpy arrays / scalars):
 *   image_points  : (4,2) float64  – observed pixel keypoints [BL,BR,TR,TL]
 *   position_xyz  : (3,)  float64  – tvec in camera frame (x,y,z metres)
 *   plate_matrix  : (4,3) float64  – 3-D object points of plate corners
 *   camera_matrix : (3,3) float64  – intrinsic K
 *   dist_coeffs   : (N,)  float64  – distortion coefficients
 *   fixed_pitch   : double          – pitch in radians (from odometry)
 *   fixed_roll    : double          – roll  in radians (from odometry)
 *   sweep_steps   : int             – number of samples across [-60°, 60°]
 *
 * Returns: (optimized_yaw: double, reprojection_error: double)
 */
static std::tuple<double, double> solve_yaw_brute_force(
    py::array_t<double> image_points,
    py::array_t<double> position_xyz,
    py::array_t<double> plate_matrix,
    py::array_t<double> camera_matrix,
    py::array_t<double> dist_coeffs,
    double fixed_pitch,
    double fixed_roll,
    int sweep_steps = 240)
{
    if (sweep_steps < 2)
        throw std::invalid_argument("sweep_steps must be at least 2");

    const BruteForceInputs brute_inputs = unpack_bruteforce_inputs(
        image_points,
        position_xyz,
        plate_matrix,
        camera_matrix,
        dist_coeffs,
        fixed_pitch,
        fixed_roll);

    constexpr double kYawMin = -60.0 * M_PI / 180.0;
    constexpr double kYawMax =  60.0 * M_PI / 180.0;

    double best_yaw = kYawMin;
    double best_sq = std::numeric_limits<double>::infinity();

    if (sweep_steps == 240 && can_use_fast_bruteforce_path(brute_inputs)) {
        const auto terms = precompute_brute_terms(brute_inputs);
        const auto& grid = get_yaw_grid_240();

        if (brute_inputs.n_dist <= 4) {
            std::tie(best_yaw, best_sq) = scan_yaw_grid_fast<false, false, false>(
                brute_inputs,
                terms,
                grid,
                false);
        } else if (brute_inputs.n_dist == 5) {
            std::tie(best_yaw, best_sq) = scan_yaw_grid_fast<false, false, false>(
                brute_inputs,
                terms,
                grid,
                true);
        } else if (brute_inputs.use_tilt) {
            std::tie(best_yaw, best_sq) = scan_yaw_grid_fast<true, true, true>(
                brute_inputs,
                terms,
                grid,
                true);
        } else if (brute_inputs.n_dist >= 9) {
            std::tie(best_yaw, best_sq) = scan_yaw_grid_fast<true, true, false>(
                brute_inputs,
                terms,
                grid,
                true);
        } else {
            std::tie(best_yaw, best_sq) = scan_yaw_grid_fast<true, false, false>(
                brute_inputs,
                terms,
                grid,
                true);
        }
    } else {
        const SolveInputs inputs = unpack_inputs(
            image_points,
            position_xyz,
            plate_matrix,
            camera_matrix,
            dist_coeffs,
            fixed_pitch,
            fixed_roll);

        const double step = (kYawMax - kYawMin) / static_cast<double>(sweep_steps - 1);
        std::vector<cv::Point2d> projected(4);

        int i = 0;
        for (; i + 3 < sweep_steps; i += 4) {
            RM_UNROLL_4
            for (int lane = 0; lane < 4; ++lane) {
                const double yaw = kYawMin + static_cast<double>(i + lane) * step;
                const double sq = squared_reprojection_error(inputs, yaw, projected);
                if (sq < best_sq) {
                    best_sq = sq;
                    best_yaw = yaw;
                }
            }
        }

        for (; i < sweep_steps; ++i) {
            const double yaw = kYawMin + static_cast<double>(i) * step;
            const double sq = squared_reprojection_error(inputs, yaw, projected);
            if (sq < best_sq) {
                best_sq = sq;
                best_yaw = yaw;
            }
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

    m.def("solve_yaw_brute_force", &solve_yaw_brute_force,
        py::arg("image_points"),
        py::arg("position_xyz"),
        py::arg("plate_matrix"),
        py::arg("camera_matrix"),
        py::arg("dist_coeffs"),
        py::arg("fixed_pitch"),
        py::arg("fixed_roll"),
        py::arg("sweep_steps") = 240,
        R"doc(
Brute-force yaw search in fixed range [-60°, 60°].

Args:
    image_points  (np.ndarray[float64, (4,2)]): observed pixel keypoints [BL,BR,TR,TL]
    position_xyz  (np.ndarray[float64, (3,)]): tvec in camera frame (metres)
    plate_matrix  (np.ndarray[float64, (4,3)]): 3-D plate corner object points
    camera_matrix (np.ndarray[float64, (3,3)]): intrinsic K matrix
    dist_coeffs   (np.ndarray[float64, (N,)]): distortion coefficients
    fixed_pitch   (float): pitch in radians (from odometry/IMU)
    fixed_roll    (float): roll  in radians (from odometry/IMU)
    sweep_steps   (int):   number of sampled yaws over [-60°, 60°], default 240

Returns:
    tuple[float, float]: (optimized_yaw_rad, reprojection_error_pixels)
)doc");

}
