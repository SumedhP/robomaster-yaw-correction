from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Float64Array = NDArray[np.float64]


def solve_yaw(
    image_points: Float64Array,
    position_xyz: Float64Array,
    plate_matrix: Float64Array,
    camera_matrix: Float64Array,
    dist_coeffs: Float64Array,
    fixed_pitch: float,
    fixed_roll: float,
    yaw_lo: float = ...,
    yaw_hi: float = ...,
    max_iter: int = ...,
) -> tuple[float, float]: ...
