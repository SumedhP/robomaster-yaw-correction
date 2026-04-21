"""Tests for the threaded brute-force yaw solver."""

import math

import numpy as np
import pytest


@pytest.fixture
def plate_setup():
    """Small plate (115 mm x 65 mm), centered at origin, corners [BL,BR,TR,TL]."""
    w, h = 0.115 / 2, 0.065 / 2
    plate_matrix = np.array([
        [-w, -h, 0.0],
        [w, -h, 0.0],
        [w, h, 0.0],
        [-w, h, 0.0],
    ], dtype=np.float64)

    camera_matrix = np.array([
        [600.0, 0.0, 320.0],
        [0.0, 600.0, 240.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    dist_coeffs = np.zeros(5, dtype=np.float64)

    return plate_matrix, camera_matrix, dist_coeffs


def _project(plate_matrix, rvec, tvec, camera_matrix, dist_coeffs):
    import cv2

    pts, _ = cv2.projectPoints(plate_matrix, rvec, tvec, camera_matrix, dist_coeffs)
    return pts.reshape(-1, 2)


def _otto_euler_to_cv2_rvec(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Mirror project_otto orientation conversion: standard Euler -> OpenCV axis-angle."""
    import cv2

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    rotation = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)

    rvec_std, _ = cv2.Rodrigues(rotation)
    rvec_std = rvec_std.reshape(3)
    return np.array([-rvec_std[1], -rvec_std[2], rvec_std[0]], dtype=np.float64)


def test_solve_yaw_zero(plate_setup):
    """With yaw=0, pitch=0, roll=0 the solver should recover ~0 rad."""
    from rm_pose_solver import solve_yaw

    plate_matrix, camera_matrix, dist_coeffs = plate_setup
    tvec = np.array([0.0, 0.0, 3.0], dtype=np.float64)
    image_points = _project(plate_matrix, np.zeros(3), tvec, camera_matrix, dist_coeffs)

    yaw, err = solve_yaw(
        image_points,
        tvec,
        plate_matrix,
        camera_matrix,
        dist_coeffs,
        fixed_pitch=0.0,
        fixed_roll=0.0,
        yaw_lo=-math.pi / 2.0,
        yaw_hi=math.pi / 2.0,
        num_threads=4,
    )

    assert abs(yaw) < math.radians(1.1), f"Expected ~0 rad, got {yaw:.4f}"
    assert err < 1.0, f"Reprojection error too large: {err:.2f} px"


def test_solve_yaw_known_angle(plate_setup):
    """Known yaw should be recovered within one sweep bin."""
    from rm_pose_solver import solve_yaw

    plate_matrix, camera_matrix, dist_coeffs = plate_setup
    true_yaw = math.radians(20.0)

    rvec = _otto_euler_to_cv2_rvec(true_yaw, 0.0, 0.0)
    tvec = np.array([0.0, 0.0, 3.0], dtype=np.float64)
    image_points = _project(plate_matrix, rvec, tvec, camera_matrix, dist_coeffs)

    yaw, err = solve_yaw(
        image_points,
        tvec,
        plate_matrix,
        camera_matrix,
        dist_coeffs,
        fixed_pitch=0.0,
        fixed_roll=0.0,
        yaw_lo=-math.pi / 2.0,
        yaw_hi=math.pi / 2.0,
        num_threads=4,
    )

    assert abs(yaw - true_yaw) <= math.radians(1.1), (
        f"Expected ~{math.degrees(true_yaw):.1f} deg, got {math.degrees(yaw):.1f} deg"
    )
    assert err < 2.0


def test_solve_yaw_matches_otto_orientation_convention(plate_setup):
    """Recover yaw when images are generated using Otto standard-frame conventions."""
    from rm_pose_solver import solve_yaw

    plate_matrix, camera_matrix, dist_coeffs = plate_setup
    true_yaw = math.radians(22.0)
    fixed_pitch = math.radians(-20.0)
    fixed_roll = math.radians(3.5)

    rvec = _otto_euler_to_cv2_rvec(true_yaw, fixed_pitch, fixed_roll)
    tvec = np.array([0.15, -0.08, 2.6], dtype=np.float64)
    image_points = _project(plate_matrix, rvec, tvec, camera_matrix, dist_coeffs)

    yaw, err = solve_yaw(
        image_points,
        tvec,
        plate_matrix,
        camera_matrix,
        dist_coeffs,
        fixed_pitch=fixed_pitch,
        fixed_roll=fixed_roll,
        yaw_lo=-math.pi / 2.0,
        yaw_hi=math.pi / 2.0,
        num_threads=4,
    )

    assert abs(yaw - true_yaw) < math.radians(1.1), (
        f"Expected ~{math.degrees(true_yaw):.1f} deg, got {math.degrees(yaw):.1f} deg"
    )
    assert err < 2.0


def test_solve_yaw_clamps_to_search_bounds(plate_setup):
    """Out-of-range true yaw should map to the nearest search bound."""
    from rm_pose_solver import solve_yaw

    plate_matrix, camera_matrix, dist_coeffs = plate_setup
    true_yaw = math.radians(70.0)

    rvec = _otto_euler_to_cv2_rvec(true_yaw, 0.0, 0.0)
    tvec = np.array([0.0, 0.0, 3.0], dtype=np.float64)
    image_points = _project(plate_matrix, rvec, tvec, camera_matrix, dist_coeffs)

    yaw, _ = solve_yaw(
        image_points,
        tvec,
        plate_matrix,
        camera_matrix,
        dist_coeffs,
        fixed_pitch=0.0,
        fixed_roll=0.0,
        yaw_lo=-math.pi / 3.0,
        yaw_hi=math.pi / 3.0,
        num_threads=4,
    )

    assert abs(yaw - math.radians(60.0)) <= math.radians(0.3)


def test_solve_yaw_consistent_across_thread_counts(plate_setup):
    """Different thread counts should return the same best sample and error."""
    from rm_pose_solver import solve_yaw

    plate_matrix, camera_matrix, dist_coeffs = plate_setup
    true_yaw = math.radians(34.0)

    rvec = _otto_euler_to_cv2_rvec(true_yaw, 0.0, 0.0)
    tvec = np.array([0.05, -0.02, 2.8], dtype=np.float64)
    image_points = _project(plate_matrix, rvec, tvec, camera_matrix, dist_coeffs)

    common_kwargs = dict(
        fixed_pitch=0.0,
        fixed_roll=0.0,
        yaw_lo=-math.pi / 2.0,
        yaw_hi=math.pi / 2.0,
    )

    yaw_single, err_single = solve_yaw(
        image_points,
        tvec,
        plate_matrix,
        camera_matrix,
        dist_coeffs,
        num_threads=1,
        **common_kwargs,
    )
    yaw_multi, err_multi = solve_yaw(
        image_points,
        tvec,
        plate_matrix,
        camera_matrix,
        dist_coeffs,
        num_threads=4,
        **common_kwargs,
    )

    assert yaw_single == pytest.approx(yaw_multi, abs=1e-12)
    assert err_single == pytest.approx(err_multi, abs=1e-12)


def test_solve_yaw_bad_input(plate_setup):
    """Wrong shape for image_points should raise."""
    from rm_pose_solver import solve_yaw

    plate_matrix, camera_matrix, dist_coeffs = plate_setup
    bad = np.zeros((3, 2), dtype=np.float64)
    tvec = np.zeros(3, dtype=np.float64)

    with pytest.raises(Exception):
        solve_yaw(
            bad,
            tvec,
            plate_matrix,
            camera_matrix,
            dist_coeffs,
            fixed_pitch=0.0,
            fixed_roll=0.0,
            num_threads=4,
        )


def test_solve_yaw_rejects_invalid_search_args(plate_setup):
    """Bounds and thread count should be validated."""
    from rm_pose_solver import solve_yaw

    plate_matrix, camera_matrix, dist_coeffs = plate_setup
    tvec = np.array([0.0, 0.0, 3.0], dtype=np.float64)
    image_points = _project(plate_matrix, np.zeros(3), tvec, camera_matrix, dist_coeffs)

    with pytest.raises(ValueError):
        solve_yaw(
            image_points,
            tvec,
            plate_matrix,
            camera_matrix,
            dist_coeffs,
            fixed_pitch=0.0,
            fixed_roll=0.0,
            yaw_lo=1.0,
            yaw_hi=-1.0,
            num_threads=4,
        )

    with pytest.raises(ValueError):
        solve_yaw(
            image_points,
            tvec,
            plate_matrix,
            camera_matrix,
            dist_coeffs,
            fixed_pitch=0.0,
            fixed_roll=0.0,
            num_threads=0,
        )

    with pytest.raises(TypeError):
        solve_yaw(
            image_points,
            tvec,
            plate_matrix,
            camera_matrix,
            dist_coeffs,
            fixed_pitch=0.0,
            fixed_roll=0.0,
            num_threads=4,
            sweep_steps=181,
        )
