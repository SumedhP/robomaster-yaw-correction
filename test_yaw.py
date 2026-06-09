import numpy as np
import math
from rm_pose_solver import solve_yaw, get_reproj_err
import matplotlib.pyplot as plt

plate_roll = 0
plate_pitch = np.radians(15)  # in world frame

plate_positions = np.array([[0, 0.0675, -0.028],
                            [0, 0.0675, 0.028],
                            [0, -0.0675, 0.028],
                            [0, -0.0675, -0.028]])

camera_matrix = np.array([[692.49749756, 0., 481.71038818],
                          [0., 692.60083008, 271.99768066],
                          [0., 0., 1.]])

scenarios = [
    {
        "name": "Expected 0",
        "image_points": np.array([[475.125, 427.1484375],
                                  [475.5, 401.1328125],
                                  [537.375, 402.5390625],
                                  [537.75, 428.5546875]]),
        "our_position": np.array([1.42499995, -0.04998241, -0.29319313]),
        "camera_pitch": -0.004616498942486942,
        "camera_roll": 0.0,
    },
    {
        "name": "Rotated to the left",
        "image_points": np.array([[405.75, 412.734375],
                                  [410.25, 389.1796875],
                                  [442.875, 396.2109375],
                                  [438.375, 419.4140625]]),
        "our_position": np.array([1.54299998, 0.12970246, -0.2940793]),
        "camera_pitch": -0.0047123742068652065,
        "camera_roll": 1.6940847042949378e-21,
    },
    {
        "name": "Rotated to the right",
        "image_points": np.array([[430.125, 408.515625],
                                  [425.625, 387.0703125],
                                  [455.625, 381.09375],
                                  [460.5, 402.5390625]]),
        "our_position": np.array([1.67299998, 0.09472811, -0.29590839]),
        "camera_pitch": -0.004616498942486943,
        "camera_roll": 8.47041973344002e-22,
    },
    {
        "name": "0 but off to the left",
        "image_points": np.array([[87.046875, 403.9453125],
                                  [89.625, 382.8515625],
                                  [141.28125, 384.2578125],
                                  [139.3125, 405.]]),
        "our_position": np.array([1.83200002, 0.97277671, -0.32138607]),
        "camera_pitch": -0.003657746240496635,
        "camera_roll": 0.0,
    },
    {
        "name": "Flipped wrong way",
        "image_points": np.array([[444.375, 243.984375],
                                  [440.625, 227.109375],
                                  [465.375, 223.9453125],
                                  [469.5, 240.46875]]),
        "our_position": np.array([2.0999999, 0.08251555, 0.1182429]),
        "camera_pitch": 0.17639600753784182,
        "camera_roll": -8.810331494666251e-19,
    },
]

def run_scenario(scenario):
    name = scenario["name"]
    image_points = scenario["image_points"]
    our_position = scenario["our_position"]
    camera_pitch = scenario["camera_pitch"]
    camera_roll = scenario["camera_roll"]

    yaw1, res1, yaw2, res2 = solve_yaw(
        image_points, our_position, plate_positions, camera_matrix,
        plate_pitch, plate_roll, camera_pitch, camera_roll
    )
    print(f"\n{name}:")
    print(f"Yaw 1: {math.degrees(yaw1):.4f} degrees, Residual: {math.sqrt(res1 / 4.0):.6f}")
    print(f"Yaw 2: {math.degrees(yaw2):.4f} degrees, Residual: {math.sqrt(res2 / 4.0):.6f}")

    reproj_errors = get_reproj_err(
        image_points, our_position, plate_positions, camera_matrix,
        plate_pitch, plate_roll, camera_pitch, camera_roll
    )
    yaws = np.linspace(-np.pi / 2, np.pi / 2, 100)

    plt.figure(figsize=(8, 6))
    plt.plot(np.degrees(yaws), reproj_errors, label='Reprojection Error')
    plt.xlabel('Yaw (degrees)')
    plt.ylabel('Squared Reprojection Error (pixels^2)')
    plt.title(f'{name}: Squared Reprojection Error vs Yaw')
    plt.axvline(np.degrees(yaw1), color='r', linestyle='--',
                label=f'Optimal Yaw 1: {math.degrees(yaw1):.2f}°')
    plt.axvline(np.degrees(yaw2), color='g', linestyle='--',
                label=f'Optimal Yaw 2: {math.degrees(yaw2):.2f}°')
    plt.legend()
    plt.grid()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(f'{safe_name}_reprojection_error_vs_yaw.png')
    plt.close()

if __name__ == "__main__":
    for scenario in scenarios:
        run_scenario(scenario)