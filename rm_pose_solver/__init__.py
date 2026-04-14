"""
rm_pose_solver
==============
Thin Python wrapper around the C++ extension.

Public API
----------
solve_yaw(image_points, position_xyz, plate_matrix, camera_matrix,
          dist_coeffs, fixed_pitch, fixed_roll, *, yaw_lo, yaw_hi, max_iter)
    -> tuple[float, float]  # (yaw_rad, reprojection_error_px)

refine_lights(bgr_image, guess_pts, enemy_is_blue, search_radius)
    -> list[tuple[float, float]]  # refined (x, y) centres
"""

from __future__ import annotations

from ._rm_pose_solver import refine_lights, solve_yaw

__all__ = ["solve_yaw", "refine_lights"]
