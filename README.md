# rm-pose-solver

High-performance C++ extension (pybind11 + OpenCV) for RoboMaster plate yaw estimation from reprojection error minimization.

## Highlights

- Native C++ implementation for low-latency yaw solving.
- Two solver modes:
  - `solve_yaw`: bounded Brent search (fast local optimization).
  - `solve_yaw_brute_force`: deterministic fixed sweep in `[-60 deg, 60 deg]`.
- Typed Python package (`py.typed` + `.pyi` stubs).

## API

### solve_yaw

Bounded 1-D optimization in yaw.

```python
from rm_pose_solver import solve_yaw

yaw_rad, reprojection_error_px = solve_yaw(
    image_points=image_points,      # np.ndarray shape (4, 2), float64
    position_xyz=position_xyz,      # np.ndarray shape (3,),   float64
    plate_matrix=plate_matrix,      # np.ndarray shape (4, 3), float64
    camera_matrix=camera_matrix,    # np.ndarray shape (3, 3), float64
    dist_coeffs=dist_coeffs,        # np.ndarray shape (N,),   float64
    fixed_pitch=pitch_rad,
    fixed_roll=roll_rad,
    yaw_lo=-1.57,
    yaw_hi=1.57,
    max_iter=50,
)
```

### solve_yaw_brute_force

Brute-force yaw sweep over a fixed range of `[-60 deg, 60 deg]`. This API intentionally does not accept `yaw_lo`/`yaw_hi`.

```python
from rm_pose_solver import solve_yaw_brute_force

yaw_rad, reprojection_error_px = solve_yaw_brute_force(
    image_points=image_points,
    position_xyz=position_xyz,
    plate_matrix=plate_matrix,
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    fixed_pitch=pitch_rad,
    fixed_roll=roll_rad,
    sweep_steps=240,
)
```

## Install (development)

```bash
pip install scikit-build-core pybind11
pip install -e ".[dev]"
```

## Build wheel

```bash
pip install build
python -m build --wheel
```

## Run tests

```bash
pytest
```

## Requirements

- Python >= 3.10
- CMake >= 3.18
- OpenCV >= 4 (core + calib3d)

## Typing

The wheel includes:

- `rm_pose_solver/__init__.pyi`
- `rm_pose_solver/_rm_pose_solver.pyi`
- `rm_pose_solver/py.typed`
