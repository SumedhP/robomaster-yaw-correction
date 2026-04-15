"""Smoke tests for rm_pose_solver."""

import math

import numpy as np
import pytest


@pytest.fixture
def plate_setup():
    """Small plate (115 mm × 65 mm), centred at origin, corners [BL,BR,TR,TL]."""
    w, h = 0.115 / 2, 0.065 / 2
    plate_matrix = np.array([
        [-w, -h, 0.0],
        [ w, -h, 0.0],
        [ w,  h, 0.0],
        [-w,  h, 0.0],
    ], dtype=np.float64)

    camera_matrix = np.array([
        [600.0,   0.0, 320.0],
        [  0.0, 600.0, 240.0],
        [  0.0,   0.0,   1.0],
    ], dtype=np.float64)

    dist_coeffs = np.zeros(5, dtype=np.float64)

    return plate_matrix, camera_matrix, dist_coeffs


def _project(plate_matrix, rvec, tvec, K, dist):
    import cv2
    pts, _ = cv2.projectPoints(plate_matrix, rvec, tvec, K, dist)
    return pts.reshape(-1, 2)


def _otto_euler_to_cv2_rvec(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Mirror project_otto orientation conversion: standard Euler -> OpenCV axis-angle."""
    import cv2

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ], dtype=np.float64)

    rvec_std, _ = cv2.Rodrigues(R)
    rvec_std = rvec_std.reshape(3)
    return np.array([-rvec_std[1], -rvec_std[2], rvec_std[0]], dtype=np.float64)


def test_solve_yaw_zero(plate_setup):
    """With yaw=0, pitch=0, roll=0 the optimiser should recover ~0 rad."""
    from rm_pose_solver import solve_yaw

    plate_matrix, K, dist = plate_setup
    tvec = np.array([0.0, 0.0, 3.0])
    rvec = np.zeros(3)
    image_points = _project(plate_matrix, rvec, tvec, K, dist)

    yaw, err = solve_yaw(
        image_points, tvec, plate_matrix, K, dist,
        fixed_pitch=0.0, fixed_roll=0.0,
    )

    assert abs(yaw) < 0.05, f"Expected ~0 rad, got {yaw:.4f}"
    assert err < 1.0, f"Reprojection error too large: {err:.2f} px"


def test_solve_yaw_known_angle(plate_setup):
    """Otto yaw=20° with zero pitch/roll should be recovered within 2°."""
    from rm_pose_solver import solve_yaw

    plate_matrix, K, dist = plate_setup
    true_yaw = math.radians(20.0)

    rvec = _otto_euler_to_cv2_rvec(true_yaw, 0.0, 0.0)
    tvec = np.array([0.0, 0.0, 3.0])
    image_points = _project(plate_matrix, rvec, tvec, K, dist)

    yaw, err = solve_yaw(
        image_points, tvec, plate_matrix, K, dist,
        fixed_pitch=0.0, fixed_roll=0.0,
    )

    assert abs(yaw - true_yaw) < math.radians(2.0), (
        f"Expected ~{math.degrees(true_yaw):.1f}°, got {math.degrees(yaw):.1f}°"
    )
    assert err < 2.0


def test_solve_yaw_matches_otto_orientation_convention(plate_setup):
    """Recover yaw when images are generated using Otto std-frame Euler conventions."""
    from rm_pose_solver import solve_yaw

    plate_matrix, K, dist = plate_setup
    true_yaw = math.radians(22.0)
    fixed_pitch = math.radians(-20.0)
    fixed_roll = math.radians(3.5)

    rvec = _otto_euler_to_cv2_rvec(true_yaw, fixed_pitch, fixed_roll)
    tvec = np.array([0.15, -0.08, 2.6], dtype=np.float64)
    image_points = _project(plate_matrix, rvec, tvec, K, dist)

    yaw, err = solve_yaw(
        image_points,
        tvec,
        plate_matrix,
        K,
        dist,
        fixed_pitch=fixed_pitch,
        fixed_roll=fixed_roll,
        yaw_lo=-math.pi / 2.0,
        yaw_hi=math.pi / 2.0,
        max_iter=80,
    )

    assert abs(yaw - true_yaw) < math.radians(2.0), (
        f"Expected ~{math.degrees(true_yaw):.1f}°, got {math.degrees(yaw):.1f}°"
    )
    assert err < 2.0


def test_solve_yaw_bad_input(plate_setup):
    """Wrong shape for image_points should raise."""
    from rm_pose_solver import solve_yaw

    plate_matrix, K, dist = plate_setup
    bad = np.zeros((3, 2))  # wrong: needs (4,2)
    tvec = np.zeros(3)

    with pytest.raises(Exception):
        solve_yaw(bad, tvec, plate_matrix, K, dist, 0.0, 0.0)
