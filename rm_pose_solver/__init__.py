"""
rm_pose_solver
==============
Thin Python wrapper around the C++ extension.

Public API
----------
solve_yaw(image_points, position_xyz, plate_matrix, camera_matrix,
          dist_coeffs, fixed_pitch, fixed_roll, *, yaw_lo, yaw_hi, max_iter)
    -> tuple[float, float]  # (yaw_rad, reprojection_error_px)

solve_yaw_brute_force(image_points, position_xyz, plate_matrix, camera_matrix,
                      dist_coeffs, fixed_pitch, fixed_roll, *, sweep_steps)
    -> tuple[float, float]  # (yaw_rad, reprojection_error_px)
"""

from __future__ import annotations

from ._rm_pose_solver import solve_yaw  # pyright: ignore[reportMissingModuleSource]

__all__ = ["solve_yaw"]
