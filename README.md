# rm-pose-solver

High-performance C++ extension (pybind11 + OpenCV) for RoboMaster plate yaw estimation.

The solver is intentionally simple: bounded brute-force search that returns the yaw sample with the least reprojection error.

## Highlights

- One API surface: `solve_yaw`
- Deterministic brute-force sweep over user-provided yaw bounds
- Threaded sampling (default: 4 threads)
- Typed Python package (`py.typed` + `.pyi` stubs)

## API

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
  yaw_lo=-1.57,                   # lower search bound (radians)
  yaw_hi=1.57,                    # upper search bound (radians)
  num_threads=4,                  # worker threads
)
```

`solve_yaw` samples yaw at fixed 1 degree increments starting at `yaw_lo`, always includes `yaw_hi`, computes reprojection error for each sample, and returns the best yaw and RMS reprojection error.

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

## Run benchmark

```bash
python benchmark.py
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
