import numpy as np
import math
import time
import statistics

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


def _build_case(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    yaw = math.radians(rng.uniform(-50.0, 50.0))
    tvec = np.array(
        [rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), rng.uniform(2.3, 3.7)],
        dtype=np.float64,
    )
    rvec = _otto_euler_to_cv2_rvec(yaw, 0.0, 0.0)
    pts, _ = cv2.projectPoints(plate_matrix, rvec, tvec, K, dist)
    return pts.reshape(-1, 2), tvec


def _simulate_background_load(seed: int) -> None:
    """Do lightweight unrelated work between solver calls to keep runs cold-ish."""
    rng = np.random.default_rng(seed)
    a = rng.random((48, 48), dtype=np.float64)
    b = rng.random((48, 48), dtype=np.float64)
    _ = a @ b
    for _ in range(1000):
        _ = math.sqrt(rng.random())


def _cold_call_latency_ms(num_threads: int, seed: int) -> float:
    _simulate_background_load(seed)
    image_points, tvec = _build_case(seed + 10_000)

    start = time.perf_counter_ns()
    solve_yaw(
        image_points,
        tvec,
        plate_matrix,
        K,
        fixed_pitch=0.0,
        fixed_roll=0.0,
        yaw_lo=math.radians(-60.0),
        yaw_hi=math.radians(60.0),
    )
    elapsed_ns = time.perf_counter_ns() - start
    return elapsed_ns / 1e6


def _characterize_threads(thread_counts: tuple[int, ...], n_trials: int) -> None:
    median_1t = None

    for threads in thread_counts:
        trials = [_cold_call_latency_ms(threads, seed=i) for i in range(n_trials)]

        avg_ms = statistics.fmean(trials)
        median_ms = statistics.median(trials)
        p90_ms = float(np.percentile(trials, 90))

        if threads == 1:
            median_1t = median_ms

        speedup = (median_1t / median_ms) if median_1t is not None else 1.0
        print(
            f"cold-start num_threads={threads}: "
            f"mean={avg_ms:.3f} ms | median={median_ms:.3f} ms | p90={p90_ms:.3f} ms | "
            f"median speedup vs 1t={speedup:.2f}x"
        )


if __name__ == "__main__":
    _characterize_threads(thread_counts=(1, 2, 4, 8), n_trials=80)