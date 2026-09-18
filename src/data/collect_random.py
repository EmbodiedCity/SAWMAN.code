import random
import airsim
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime

def quaternion2eularian_angles(q):
    (pitch, roll, yaw) = airsim.to_eularian_angles(q)
    return (roll, pitch, yaw)
import sqlite3
conn = None
cursor = None
path = 'data/random'

def set_camera_angle(client, angle):
    camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(angle * np.pi / 180, 0, 0))
    client.simSetCameraPose('0', camera_pose)

def check_obstacle(client, direction):
    set_camera_angle(client, 0)
    if direction == 'up':
        set_camera_angle(client, 90)
    elif direction == 'down':
        set_camera_angle(client, -90)
    responses = client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.DepthPerspective, True)])
    depth_image = airsim.list_to_2d_float_array(responses[0].image_data_float, responses[0].width, responses[0].height)
    center_region = depth_image[4 * depth_image.shape[0] // 9:5 * depth_image.shape[0] // 9, 4 * depth_image.shape[1] // 9:5 * depth_image.shape[1] // 9]
    return center_region.mean() < 15

def calculate_target_position(client, direction, distance=10):
    if direction == 'forward':
        (dx, dy, dz) = (10, 0, 0)
    elif direction == 'up':
        (dx, dy, dz) = (0, 0, -10)
    elif direction == 'down':
        (dx, dy, dz) = (0, 0, 10)
    pose = client.simGetVehiclePose()
    orientation = airsim.to_eularian_angles(pose.orientation)
    yaw = orientation[2]
    forward = np.array([np.cos(yaw), np.sin(yaw), 0])
    right = np.array([-np.sin(yaw), np.cos(yaw), 0])
    up = np.array([0, 0, 1])
    move_vector = dx * forward + dy * right + dz * up
    new_position = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val]) + move_vector
    return new_position

def save_images(client, folder):
    directions = ['front', 'back', 'left', 'right', 'up', 'down']
    os.makedirs(folder, exist_ok=True)
    for direction in directions:
        if direction == 'front':
            (yaw, pitch) = (0, 0)
        elif direction == 'back':
            (yaw, pitch) = (180, 0)
        elif direction == 'left':
            (yaw, pitch) = (-90, 0)
        elif direction == 'right':
            (yaw, pitch) = (90, 0)
        elif direction == 'up':
            (yaw, pitch) = (0, 90)
        elif direction == 'down':
            (yaw, pitch) = (0, -90)
        camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(pitch * np.pi / 180, 0, yaw * np.pi / 180))
        client.simSetCameraPose('0', camera_pose)
        response = client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]
        img_data = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)
        img_path = os.path.join(folder, f'{direction}.png')
        airsim.write_png(img_path, img_data)
    set_camera_angle(client, 0)

def quaternion_to_euler(q):
    """Quaternion to euler."""
    (w, x, y, z) = (q.w_val, q.x_val, q.y_val, q.z_val)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2
    else:
        pitch = np.arcsin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return (roll, pitch, yaw)

def get_current_state(client):
    state = client.simGetGroundTruthKinematics()
    pos = state.position.to_numpy_array()
    ori = quaternion2eularian_angles(state.orientation)
    return (pos, ori)

def set_vehicle_pose(client, position, orientation):
    pose = airsim.Pose(airsim.Vector3r(*position), airsim.to_quaternion(orientation[1], orientation[0], orientation[2]))
    client.simSetVehiclePose(pose, True)

def save_to_database(timestamp, position, orientation):
    cursor.execute('\n    INSERT INTO locations (timestamp, x, y, z, roll, pitch, yaw)\n    VALUES (?, ?, ?, ?, ?, ?, ?)\n    ', (timestamp, position[0], position[1], position[2], orientation[0], orientation[1], orientation[2]))
    conn.commit()

def main():
    import argparse, json
    global conn, cursor, path
    parser = argparse.ArgumentParser(description='Legacy random AirSim capture (not action-chain training data)')
    parser.add_argument('--config', default='configs/random_capture.json')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    path = cfg['output_dir']
    os.makedirs(path, exist_ok=True)
    conn = sqlite3.connect(os.path.join(path, 'locations.db'))
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS locations (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp INTEGER, x REAL, y REAL, z REAL, roll REAL, pitch REAL, yaw REAL)')
    conn.commit()
    boundary_point1 = cfg['boundary_max']
    boundary_point2 = cfg['boundary_min']
    start_points = cfg['start_points']
    start_point = random.choice(start_points)
    start_yaw = random.choice([0, np.pi / 2, np.pi, -np.pi / 2])
    client = airsim.VehicleClient()
    client.confirmConnection()
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(start_point[0], start_point[1], start_point[2]), airsim.to_quaternion(0.0, 0, start_yaw)), True)
    set_camera_angle(client, 0)
    collect_num = cfg['num_actions']
    for i in range(collect_num):
        (current_pos, _) = get_current_state(client)
        if current_pos[0] < boundary_point2[0] or current_pos[0] > boundary_point1[0] or current_pos[1] < boundary_point2[1] or (current_pos[1] > boundary_point1[1]) or (current_pos[2] < boundary_point2[2]) or (current_pos[2] > boundary_point1[2]):
            start_point = random.choice(start_points)
            start_yaw = random.choice([0, np.pi / 2, np.pi, -np.pi / 2])
            client = airsim.VehicleClient()
            client.confirmConnection()
            client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(start_point[0], start_point[1], start_point[2]), airsim.to_quaternion(0.0, 0, start_yaw)), True)
            set_camera_angle(client, 0)
            print('Outside bounds; returning to the start. Current position: ', current_pos)
        yaw_change = 2 * np.random.choice([-1, 1])
        pose = client.simGetVehiclePose()
        current_orientation = airsim.to_eularian_angles(pose.orientation)
        new_orientation = [current_orientation[1], current_orientation[0], current_orientation[2] + np.radians(yaw_change)]
        set_vehicle_pose(client, [pose.position.x_val, pose.position.y_val, pose.position.z_val], new_orientation)
        directions = ['forward', 'up', 'down']
        probabilities = [1 / 6, 2 / 6, 3 / 6]
        direction = np.random.choice(directions, p=probabilities)
        temp_directions = ['forward', 'up', 'down']
        check_num = 0
        while check_obstacle(client, direction):
            print(0)
            if direction == 'forward':
                yaw_change = 2
                pose = client.simGetVehiclePose()
                current_orientation = airsim.to_eularian_angles(pose.orientation)
                new_orientation = [current_orientation[1], current_orientation[0], current_orientation[2] + np.radians(yaw_change)]
                set_vehicle_pose(client, [pose.position.x_val, pose.position.y_val, pose.position.z_val], new_orientation)
            else:
                temp_directions.remove(direction)
                if not temp_directions:
                    break
                direction = np.random.choice(temp_directions)
            check_num += 1
            if check_num > 90:
                break
        print(i, ' / ', collect_num, ': ', direction)
        if not temp_directions or check_num > 90:
            start_point = random.choice(start_points)
            start_yaw = random.choice([0, np.pi / 2, np.pi, -np.pi / 2])
            client = airsim.VehicleClient()
            client.confirmConnection()
            client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(start_point[0], start_point[1], start_point[2]), airsim.to_quaternion(0.0, 0, start_yaw)), True)
            set_camera_angle(client, 0)
            print('Surrounded by obstacles; returning to the start')
        target_pos = calculate_target_position(client, direction)
        (current_pos, current_orientation) = get_current_state(client)
        interpolated_positions = []
        for t in range(20):
            interpolated_positions.append(current_pos + (target_pos - current_pos) * t / 19)
        for pos in interpolated_positions:
            set_vehicle_pose(client, pos, current_orientation)
            timestamp = int(time.time() * 100)
            folder = os.path.join(path, str(timestamp))
            save_images(client, folder)
            save_to_database(timestamp, pos, current_orientation)
    conn.close()
if __name__ == '__main__':
    main()
