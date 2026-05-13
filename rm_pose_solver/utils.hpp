#include <opencv2/opencv.hpp>

/// Rodrigues rotation vector from ZYX Euler angles in OpenCV coordinates.
/// OpenCV convention: X-right, Y-down, Z-forward.
/// Rotation order: R = Rz(yaw) * Ry(pitch) * Rx(roll)
/// where yaw rotates about Z (forward), pitch about Y (down), roll about X (right).
/// Caller is responsible for converting angles from their own frame beforehand.
static inline cv::Vec3d euler_to_rvec(
    double yaw,
    double sin_pitch,
    double cos_pitch,
    double sin_roll,
    double cos_roll)
{
    const double cy = std::cos(yaw);
    const double sy = std::sin(yaw);

    // R = Rz(yaw) * Ry(pitch) * Rx(roll), fully expanded.
    // Identical algebraic form to before — the frame is now baked into
    // what the caller passes in, not handled here.
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