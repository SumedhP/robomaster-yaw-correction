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


def test_solve_yaw_zero(plate_setup):
    """With yaw=0, pitch=0, roll=0 the optimiser should recover ~0 rad."""
    from rm_pose_solver import solve_yaw
    import cv2

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
    """Plate rotated 20° → optimiser recovers within 2°."""
    from rm_pose_solver import solve_yaw
    import cv2

    plate_matrix, K, dist = plate_setup
    true_yaw = math.radians(20.0)

    import cv2
    R = cv2.Rodrigues(np.array([0.0, true_yaw, 0.0]))[0]
    rvec, _ = cv2.Rodrigues(R)
    tvec = np.array([0.0, 0.0, 3.0])
    image_points = _project(plate_matrix, rvec.flatten(), tvec, K, dist)

    yaw, err = solve_yaw(
        image_points, tvec, plate_matrix, K, dist,
        fixed_pitch=0.0, fixed_roll=0.0,
    )

    assert abs(yaw - true_yaw) < math.radians(2.0), (
        f"Expected ~{math.degrees(true_yaw):.1f}°, got {math.degrees(yaw):.1f}°"
    )
    assert err < 2.0


def test_refine_lights_finds_blob():
    """A bright blue blob in a dark image should be detected near the centre."""
    from rm_pose_solver import refine_lights

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Draw a small bright blue circle at (60, 40)
    import cv2
    cv2.circle(img, (60, 40), 8, (220, 20, 20), -1)  # BGR: high blue

    guess = np.array([[62.0, 42.0]], dtype=np.float64)
    refined = refine_lights(img, guess, enemy_is_blue=True, search_radius=20)

    assert len(refined) == 1
    rx, ry = refined[0]
    assert abs(rx - 60) < 5 and abs(ry - 40) < 5, (
        f"Expected ~(60,40), got ({rx:.1f},{ry:.1f})"
    )


def test_refine_lights_fallback_on_no_blob():
    """No contour found → original guess returned unchanged."""
    from rm_pose_solver import refine_lights

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    guess = np.array([[50.0, 50.0]], dtype=np.float64)
    refined = refine_lights(img, guess, enemy_is_blue=True, search_radius=20)

    assert len(refined) == 1
    assert refined[0][0] == pytest.approx(50.0)
    assert refined[0][1] == pytest.approx(50.0)


def test_solve_yaw_bad_input(plate_setup):
    """Wrong shape for image_points should raise."""
    from rm_pose_solver import solve_yaw

    plate_matrix, K, dist = plate_setup
    bad = np.zeros((3, 2))  # wrong: needs (4,2)
    tvec = np.zeros(3)

    with pytest.raises(Exception):
        solve_yaw(bad, tvec, plate_matrix, K, dist, 0.0, 0.0)
