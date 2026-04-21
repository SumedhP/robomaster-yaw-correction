import numpy as np
import math
from rm_pose_solver import solve_yaw
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

for deg in range(-50, 51, 10):
    true_yaw = math.radians(deg)
    rvec = _otto_euler_to_cv2_rvec(true_yaw, 0.0, 0.0)
    pts, _ = cv2.projectPoints(plate_matrix, rvec, tvec, K, dist)
    image_points = pts.reshape(-1, 2)

    yaw, err = solve_yaw(
        image_points, tvec, plate_matrix, K, dist,
        fixed_pitch=0.0, fixed_roll=0.0,
        yaw_lo=math.radians(-60), yaw_hi=math.radians(60),
        num_threads=4,
    )

    print(
        "True: "
        f"{deg:4d} | "
        f"Solve: {math.degrees(yaw):8.2f} deg (err={err:7.4f})"
    )
