# rm-pose-solver

Minimal C++ extension (pybind11 + OpenCV) for RoboMaster plate pose yaw estimation.

## What it does

### `solve_yaw`
Given pre-computed depth position, fixed pitch/roll from odometry, and observed
pixel keypoints, finds the yaw that minimises the keypoint reprojection error
via bounded Brent search (default bounds: [-π/2, π/2]).

```python
from rm_pose_solver import solve_yaw
import numpy as np

yaw_rad, reprojection_error_px = solve_yaw(
    image_points  = np.array([[x0,y0],[x1,y1],[x2,y2],[x3,y3]], dtype=np.float64),
    position_xyz  = np.array([tx, ty, tz]),          # camera-frame tvec
    plate_matrix  = np.array([...]),                  # (4,3) object points
    camera_matrix = K,                                # (3,3)
    dist_coeffs   = dist,                             # (5,) or (8,)
    fixed_pitch   = pitch_rad,
    fixed_roll    = roll_rad,
)
```

## Typing support

This package ships with:

- `rm_pose_solver/__init__.pyi`
- `rm_pose_solver/_rm_pose_solver.pyi`
- `rm_pose_solver/py.typed` (PEP 561 marker)

so type checkers can resolve signatures for `solve_yaw`.

## Build requirements

- CMake ≥ 3.18
- OpenCV ≥ 4.x (with `core`, `imgproc`, `calib3d`)
- Python ≥ 3.10

## Install (editable / development)

```bash
pip install scikit-build-core pybind11
pip install -e ".[dev]"
```

## Build a wheel

```bash
pip install build
python -m build --wheel
# → dist/rm_pose_solver-0.1.0-*.whl
```

## Run tests

```bash
pytest
```
