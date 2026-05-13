import numpy as np
import math
from rm_pose_solver import solve_yaw, get_reproj_err


image_points = np.array([[829.5, 304.1015625], # BL
                         [835.5, 277.3828125], # TL
                        [900.75, 294.78515625], # TR
                        [894.75, 321.50390625]]) # BR

our_position = np.array([1.3890000581741333, -0.7677931189537048, -0.0541527234017849])

plate_positions = np.array([[0, 0.0675, -0.028],
                            [0, 0.0675, 0.028],
                            [0, -0.0675, 0.028],
                            [0, -0.0675, -0.028]])

camera_matrix = np.array([[692.49749756, 0., 481.71038818],
                          [0., 692.60083008, 271.99768066],
                          [0., 0., 1.]])

camera_pitch = 0.22989439487457278 # Radians, positive is looking down
camera_roll = 0.0 # Radians, positive is rolling left

plate_roll = 0
plate_pitch = np.radians(15) # in world frame

plate_world_roll = plate_roll - camera_roll
plate_world_pitch = plate_pitch - camera_pitch

yaw1, res1, yaw2, res2 = solve_yaw(image_points, our_position, plate_positions, camera_matrix, plate_world_pitch, plate_world_roll)

print(f"Yaw 1: {math.degrees(yaw1)} degrees, Residual: {res1}")
print(f"Yaw 2: {math.degrees(yaw2)} degrees, Residual: {res2}")

import timeit

def benchmark():
    solve_yaw(image_points, our_position, plate_positions, camera_matrix, plate_world_pitch, plate_world_roll)

N = 1000
execution_time = timeit.timeit(benchmark, number=N)
print(f"Average execution time over {N} runs: {(execution_time / N)*1000:.6f} milliseconds")

reproj_errors = get_reproj_err(image_points, our_position, plate_positions, camera_matrix, plate_world_pitch, plate_world_roll)
yaws = np.linspace(-np.pi/2, np.pi/2, 100)

# for yaw, err in zip(yaws, reproj_errors):
#     print(f"Yaw: {math.degrees(yaw):.2f} degrees, Reprojection Error: {err:.4f} pixels")

import matplotlib.pyplot as plt
# Plot just the reprojection errors across line
plt.figure(figsize=(8, 6))
plt.plot(np.degrees(yaws), reproj_errors, label='Reprojection Error')
plt.xlabel('Yaw (degrees)')
plt.ylabel('Reprojection Error (pixels)')
plt.title('Reprojection Error vs Yaw')
plt.axvline(np.degrees(yaw1), color='r', linestyle='--', label=f'Optimal Yaw 1: {math.degrees(yaw1):.2f}°')
plt.axvline(np.degrees(yaw2), color='g', linestyle='--', label=f'Optimal Yaw 2: {math.degrees(yaw2):.2f}°')
plt.legend()
plt.grid()
plt.savefig('reprojection_error_vs_yaw.png')
