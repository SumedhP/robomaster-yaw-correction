import numpy as np
import math
from rm_pose_solver import solve_yaw, get_reproj_err
import matplotlib.pyplot as plt

plate_roll = 0
plate_pitch = np.radians(15) # in world frame

plate_positions = np.array([[0, 0.0675, -0.028],
                            [0, 0.0675, 0.028],
                            [0, -0.0675, 0.028],
                            [0, -0.0675, -0.028]])

camera_matrix = np.array([[692.49749756, 0., 481.71038818],
                          [0., 692.60083008, 271.99768066],
                          [0., 0., 1.]])

def pitch_down_output():
    image_points = np.array([[417., 122.51953125], # BL
                            [421.125, 94.48242188], # TL
                            [474.75, 105.29296875], # TR
                            [471., 132.97851562]]) # BR

    our_position = np.array([1.4359999895095825, 0.07508780807256699, 0.3296569585800171])

    camera_pitch = 0.44916113734245294 # Radians, positive is looking down
    camera_roll = 0.0 # Radians, positive is rolling left

    yaw1, res1, yaw2, res2 = solve_yaw(image_points, our_position, plate_positions, camera_matrix, plate_pitch, plate_roll, camera_pitch, camera_roll)
    print(f"Pitch down scenario:")
    print(f"Yaw 1: {math.degrees(yaw1)} degrees, Residual: {math.sqrt(res1 / 4.0)}")
    print(f"Yaw 2: {math.degrees(yaw2)} degrees, Residual: {math.sqrt(res2 / 4.0)}")
    
    reproj_errors = get_reproj_err(image_points, our_position, plate_positions, camera_matrix, plate_pitch, plate_roll, camera_pitch, camera_roll)
    yaws = np.linspace(-np.pi/2, np.pi/2, 100)
    
    plt.figure(figsize=(8, 6))
    plt.plot(np.degrees(yaws), reproj_errors, label='Reprojection Error')
    plt.xlabel('Yaw (degrees)')
    plt.ylabel('Squred Reprojection Error (pixels^2)')
    plt.title('Pitch Down: Squared Reprojection Error vs Yaw')
    plt.axvline(np.degrees(yaw1), color='r', linestyle='--', label=f'Optimal Yaw 1: {math.degrees(yaw1):.2f}°')
    plt.axvline(np.degrees(yaw2), color='g', linestyle='--', label=f'Optimal Yaw 2: {math.degrees(yaw2):.2f}°')
    plt.legend()
    plt.grid()
    plt.savefig('pitch_down_reprojection_error_vs_yaw.png')

def pitch_up_output():
    image_points = np.array([[428.625, 402.890625], # BL
                             [433.5, 376.875], # TL
                            [484.5, 386.3671875], # TR
                            [480.375, 412.734375]]) # BR

    our_position = np.array([1.4479999542236328, 0.05375996232032776, -0.2550666332244873])
    camera_pitch = 0.08176711469888688 # Radians, positive is looking down
    camera_roll = 0.0 # Radians, positive is rolling left

    yaw1, res1, yaw2, res2 = solve_yaw(image_points, our_position, plate_positions, camera_matrix, plate_pitch, plate_roll, camera_pitch, camera_roll)
    print(f"\nPitch up scenario:")
    print(f"Yaw 1: {math.degrees(yaw1)} degrees, Residual: {math.sqrt(res1 / 4.0)}")
    print(f"Yaw 2: {math.degrees(yaw2)} degrees, Residual: {math.sqrt(res2 / 4.0)}")

    reproj_errors = get_reproj_err(image_points, our_position, plate_positions, camera_matrix, plate_pitch, plate_roll, camera_pitch, camera_roll)
    yaws = np.linspace(-np.pi/2, np.pi/2, 100)
    
    plt.figure(figsize=(8, 6))
    plt.plot(np.degrees(yaws), reproj_errors, label='Reprojection Error')
    plt.xlabel('Yaw (degrees)')
    plt.ylabel('Squred Reprojection Error (pixels^2)')
    plt.title('Pitch Up: Squared Reprojection Error vs Yaw')
    plt.axvline(np.degrees(yaw1), color='r', linestyle='--', label=f'Optimal Yaw 1: {math.degrees(yaw1):.2f}°')
    plt.axvline(np.degrees(yaw2), color='g', linestyle='--', label=f'Optimal Yaw 2: {math.degrees(yaw2):.2f}°')
    plt.legend()
    plt.grid()
    plt.savefig('pitch_up_reprojection_error_vs_yaw.png')

if __name__ == "__main__":
    pitch_down_output()
    pitch_up_output()