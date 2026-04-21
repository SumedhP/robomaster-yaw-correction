#include <opencv2/opencv.hpp>


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
/// From X-forward, Y-right, Z-down to X-right, Y-down, Z-forward.
/// Matches Orientation.to_cv2_coords(): (-y, -z, x)
static inline cv::Vec3d standard_rvec_to_cv2(const cv::Vec3d& rvec_std)
{
    return cv::Vec3d(-rvec_std[1], -rvec_std[2], rvec_std[0]);
}