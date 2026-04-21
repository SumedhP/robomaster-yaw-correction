import numpy as np
import math
from rm_pose_solver import solve_yaw, solve_yaw_brute_force
import cv2

plate_matrix = np.array([
    [-0.0575, -0.0325, 0.0],
    [ 0.0575, -0.0325, 0.0],
    [ 0.0575,  0.0325, 0.0],
    [-0.0575,  0.0325, 0.0],
], dtype=np.float64)

K = np.array([
    [600.0,   0.0, 320.0],
    [  0.0, 600.0, 240.0],
    [  0.0,   0.0,   1.0],
], dtype=np.float64)

dist = np.zeros(5, dtype=np.float64)
tvec = np.array([0.0, 0.0, 3.0])

def _otto_euler_to_cv2_rvec(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ], dtype=np.float64)
    rvec_std, _ = cv2.Rodrigues(R)
    rvec_std = rvec_std.reshape(3)
    return np.array([-rvec_std[1], -rvec_std[2], rvec_std[0]], dtype=np.float64)

a = 0
for i in range(100):
    a += 1

import time

start_time = time.perf_counter_ns()

# Make 1 call to solve_yaw_brute_force
# _ = solve_yaw_brute_force(
#     np.random.rand(4, 2) * 640, tvec, plate_matrix, K, dist,
#     fixed_pitch=0.0, fixed_roll=0.0,
#     sweep_steps=180
# )
_ = solve_yaw(
    np.random.rand(4, 2) * 640, tvec, plate_matrix, K, dist,
    fixed_pitch=0.0, fixed_roll=0.0,
    yaw_lo=math.radians(-50), yaw_hi=math.radians(50), max_iter=100
)
end_time = time.perf_counter_ns()
elapsed_time_brute_force = end_time - start_time
print(f"Brute-force solver took {elapsed_time_brute_force / 1e6:.2f} ms")