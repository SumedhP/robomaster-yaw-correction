import numpy as np
import math
from rm_pose_solver import solve_yaw


image_points = np.array([[448.5, 309.7265625], # BL
                         [445.5, 290.91796875], # TL
                        [485.25, 287.40234375], # TR
                        [488.25, 306.03515625]]) # BR

our_position = np.array([1.98, 0.04351175, -0.0743727])

# plate_points = np.array([[-0.0675, -0.028, 0.],
#                          [-0.0675, 0.028, 0.],
#                          [0.0675, 0.028, 0.],
#                          [0.0675, -0.028, 0.]])
plate_positions = np.array([[0, 0.0675, -0.028],
                            [0, 0.0675, 0.028],
                            [0, -0.0675, 0.028],
                            [0, -0.0675, -0.028]])

camera_matrix = np.array([[692.49749756, 0., 481.71038818],
                          [0., 692.60083008, 271.99768066],
                          [0., 0., 1.]])

camera_pitch = 0.13603251695632934 # Radians, positive is looking down
camera_roll = -7.003594254430487e-18 # Radians, positive is rolling left

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