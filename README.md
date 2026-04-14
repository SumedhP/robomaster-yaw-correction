# rm-pose-solver

Minimal C++ extension (pybind11 + OpenCV) for RoboMaster plate pose estimation.

## What it does

### `solve_yaw`
Given pre-computed depth position, fixed pitch/roll from odometry, and observed
pixel keypoints, finds the yaw that minimises the keypoint reprojection error
via golden-section search (bounded [-π/2, π/2]).

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

### `refine_lights`
Refines light-bar centre guesses using Otsu thresholding on the blue or red
channel (mirroring the approach used by Chinese RoboMaster teams).

```python
from rm_pose_solver import refine_lights

refined = refine_lights(
    bgr_image     = frame,        # (H,W,3) uint8
    guess_pts     = np.array([[x0,y0],[x1,y1]]),
    enemy_is_blue = True,
    search_radius = 30,           # pixels
)
# refined: list of (x, y) floats
```

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
