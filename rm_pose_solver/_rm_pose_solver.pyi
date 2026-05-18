from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Float64Array = NDArray[np.float64]


def solve_yaw(
    image_points: Float64Array,
    position_xyz: Float64Array,
    plate_matrix: Float64Array,
    camera_matrix: Float64Array,
    plate_pitch: float,
    plate_roll: float,
    cam_pitch: float,
    cam_roll: float,
    yaw_lo: float = ...,
    yaw_hi: float = ...,
) -> tuple[float, float, float, float]: ...

def get_reproj_err(
    image_points: Float64Array,
    position_xyz: Float64Array,
    plate_matrix: Float64Array,
    camera_matrix: Float64Array,
    plate_pitch: float,
    plate_roll: float,
    cam_pitch: float,
    cam_roll: float,
    yaw_lo: float = ...,
    yaw_hi: float = ...,
) -> Float64Array: ...
