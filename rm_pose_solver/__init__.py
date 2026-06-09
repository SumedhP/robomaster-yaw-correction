"""
rm_pose_solver
==============
Thin Python wrapper around the C++ extension.

Public API
----------
solve_yaw(image_points, position_xyz, plate_matrix, camera_matrix,
          dist_coeffs, fixed_pitch, fixed_roll, *, yaw_lo, yaw_hi,
          num_threads)
    -> tuple[float, float]  # (yaw_rad, reprojection_error_px)
"""

from __future__ import annotations

import importlib.util
import site
import sys
from pathlib import Path
from types import ModuleType


def _load_extension_from_site_packages() -> ModuleType:
    patterns = ("_rm_pose_solver*.so", "_rm_pose_solver*.pyd")
    candidates: list[Path] = []

    for base in site.getsitepackages():
        pkg_dir = Path(base) / "rm_pose_solver"
        if not pkg_dir.is_dir():
            continue
        for pattern in patterns:
            candidates.extend(pkg_dir.glob(pattern))

    user_site = site.getusersitepackages()
    if user_site:
        user_pkg_dir = Path(user_site) / "rm_pose_solver"
        if user_pkg_dir.is_dir():
            for pattern in patterns:
                candidates.extend(user_pkg_dir.glob(pattern))

    if not candidates:
        raise ModuleNotFoundError(
            "No compiled rm_pose_solver extension found. "
            "Build/install it with: python -m pip install -e ."
        )

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    ext_path = candidates[0]

    module_name = "rm_pose_solver._rm_pose_solver"
    spec = importlib.util.spec_from_file_location(module_name, ext_path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Unable to load extension module from {ext_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


try:
    from ._rm_pose_solver import solve_yaw, get_reproj_err  # pyright: ignore[reportMissingModuleSource]
except ModuleNotFoundError as exc:
    if exc.name != "rm_pose_solver._rm_pose_solver":
        raise
    solve_yaw = _load_extension_from_site_packages().solve_yaw
    get_reproj_err = _load_extension_from_site_packages().get_reproj_err

__all__ = ["solve_yaw", "get_reproj_err"]
