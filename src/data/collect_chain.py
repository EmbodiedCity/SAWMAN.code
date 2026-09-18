"""AirSim six-view action-chain collection with camera and trajectory records."""
import random
import argparse
import airsim
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
import csv
from typing import Dict, List, Optional, TextIO, Tuple, Union
try:
    import cv2
except Exception:
    cv2 = None
try:
    from PIL import Image
except Exception:
    Image = None
quaternion2eularian_angles = None
path = 'data/raw/ENV0'
AIRSIM_PORT = 41451
MAX_STEPS_PER_CHAIN = 200
ACTION_DISTANCE_METERS = 10.0
ACTION_FRAMES_PER_STEP = 20
ACTION_SAMPLE_INTERVAL_METERS = ACTION_DISTANCE_METERS / ACTION_FRAMES_PER_STEP
USE_CSV_START_POINTS = True
ENV_NAME = 'ENV0'
START_POINTS_CSV_DIR = 'data/start_points'
START_POINTS_LIMIT: Optional[int] = None
USE_CSV_YAW = True
CSV_YAW_SIGN = 1.0
CSV_YAW_OFFSET_DEG = 0.0
SELECT_START_POINTS = False
START_POINT_INDEX_BASE = 0
START_POINT_INDEXES: List[int] = []
START_POINT_RANGE_START = 0
START_POINT_RANGE_COUNT: Optional[int] = None
AUTO_BOUNDARY_FROM_CSV = True
BOUNDARY_XY_MARGIN = 100.0
BOUNDARY_Z_UP_MARGIN = 20.0
BOUNDARY_Z_DOWN_MARGIN = 30.0
TARGET_RESOLUTIONS = [320]
COLLECT_ALL_RESOLUTIONS = True
SAVE_SIX_VIEW_MOSAIC = True
CAMERA_FOV_DEGREES = 90.0
VIEW_CONFIGS = [('front', 'UAV_front', 0.0, 0.0), ('back', 'UAV_back', 180.0, 0.0), ('left', 'UAV_left', -90.0, 0.0), ('right', 'UAV_right', 90.0, 0.0), ('up', 'UAV_up', 0.0, 90.0), ('down', 'UAV_down', 0.0, -90.0)]

def get_active_resolutions() -> List[int]:
    return TARGET_RESOLUTIONS if COLLECT_ALL_RESOLUTIONS else TARGET_RESOLUTIONS[:1]

def parse_resolution_list(value: str) -> List[int]:
    parts = [part.strip() for part in value.replace(';', ',').replace(' ', ',').split(',')]
    resolutions = [int(part) for part in parts if part]
    if not resolutions:
        raise argparse.ArgumentTypeError('At least one resolution is required')
    if any((size <= 0 for size in resolutions)):
        raise argparse.ArgumentTypeError('Resolution values must be positive integers')
    return resolutions

def parse_args():
    parser = argparse.ArgumentParser(description='AirSim six-view chain data collector')
    parser.add_argument('--env-name', default=ENV_NAME, help='Environment name, e.g. ENV3')
    parser.add_argument('--port', type=int, default=AIRSIM_PORT, help='AirSim RPC port')
    parser.add_argument('--out-root', default=path, help='Output root directory')
    parser.add_argument('--start-points-dir', default=START_POINTS_CSV_DIR, help='Directory containing ENV*_start_points.csv')
    parser.add_argument('--steps', type=int, default=MAX_STEPS_PER_CHAIN, help='Max successful actions per start point')
    parser.add_argument('--resolutions', type=parse_resolution_list, default=None, help='Output resolutions, e.g. 320 or 256,320')
    parser.add_argument('--range-start', type=int, default=None, help='Start point row index to begin from')
    parser.add_argument('--range-count', type=int, default=None, help='Number of start points to collect')
    parser.add_argument('--save-six-view-mosaic', dest='save_six_view_mosaic', action='store_true')
    parser.add_argument('--no-save-six-view-mosaic', dest='save_six_view_mosaic', action='store_false')
    parser.set_defaults(save_six_view_mosaic=SAVE_SIX_VIEW_MOSAIC)
    return parser.parse_args()

def get_start_points_csv_path(env_name: str) -> str:
    return os.path.join(START_POINTS_CSV_DIR, f'{env_name}_start_points.csv')

def load_start_points_from_csv(csv_path: str, limit: Optional[int]=None) -> List[Tuple[str, List[float], float]]:
    """Load start points from csv."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'Start-point CSV not found: {csv_path}')
    start_points: List[Tuple[str, List[float], float]] = []
    with open(csv_path, newline='', encoding='utf-8-sig', errors='replace') as f:
        reader = csv.DictReader(f)
        required_columns = {'x', 'y', 'z'}
        if reader.fieldnames is None or not required_columns.issubset(set(reader.fieldnames)):
            raise ValueError(f'Start-point CSV must contain columns: {sorted(required_columns)}')
        for (idx, row) in enumerate(reader):
            point_id = (row.get('id') or f'{idx:03d}').strip()
            x = float(row['x'])
            y = float(row['y'])
            z = float(row['z'])
            yaw_deg = float(row.get('yaw_deg') or 0.0)
            yaw_deg = CSV_YAW_SIGN * yaw_deg + CSV_YAW_OFFSET_DEG
            start_points.append((point_id, [x, y, z], np.radians(yaw_deg)))
            if limit is not None and len(start_points) >= limit:
                break
    if not start_points:
        raise ValueError(f'No valid coordinates in start-point CSV: {csv_path}')
    return start_points

def select_start_point_entries(start_point_entries: List[Tuple[str, List[float], float]]) -> List[Tuple[str, List[float], float]]:
    """Select start point entries."""
    if not SELECT_START_POINTS:
        return start_point_entries

    def to_zero_based(index: int) -> int:
        return int(index) - int(START_POINT_INDEX_BASE)
    if START_POINT_INDEXES:
        selected = []
        for raw_index in START_POINT_INDEXES:
            index = to_zero_based(raw_index)
            if index < 0 or index >= len(start_point_entries):
                raise IndexError(f'Start-point row index out of range: {raw_index} (base={START_POINT_INDEX_BASE}, total={len(start_point_entries)})')
            selected.append(start_point_entries[index])
        return selected
    start = to_zero_based(START_POINT_RANGE_START)
    if start < 0 or start >= len(start_point_entries):
        raise IndexError(f'Start range exceeds CSV rows: START_POINT_RANGE_START={START_POINT_RANGE_START}, base={START_POINT_INDEX_BASE}, total={len(start_point_entries)}')
    if START_POINT_RANGE_COUNT is None:
        end = len(start_point_entries)
    else:
        if START_POINT_RANGE_COUNT <= 0:
            raise ValueError('START_POINT_RANGE_COUNT must be positive or None')
        end = start + int(START_POINT_RANGE_COUNT)
    selected = start_point_entries[start:end]
    if not selected:
        raise ValueError('No start points selected; check SELECT_START_POINTS')
    return selected

def estimate_boundary_from_start_points(start_point_entries: List[Tuple[str, List[float], float]], xy_margin: float=BOUNDARY_XY_MARGIN, z_up_margin: float=BOUNDARY_Z_UP_MARGIN, z_down_margin: float=BOUNDARY_Z_DOWN_MARGIN) -> Tuple[List[float], List[float]]:
    """Estimate boundary from start points."""
    points = np.array([entry[1] for entry in start_point_entries], dtype=np.float64)
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    boundary_point1 = [float(max_xyz[0] + xy_margin), float(max_xyz[1] + xy_margin), float(max_xyz[2] + z_down_margin)]
    boundary_point2 = [float(min_xyz[0] - xy_margin), float(min_xyz[1] - xy_margin), float(min_xyz[2] - z_up_margin)]
    return (boundary_point1, boundary_point2)

def set_camera_angle(client: airsim.VehicleClient, angle: float) -> None:
    """Set camera angle."""
    camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(angle * np.pi / 180, 0, 0))
    client.simSetCameraPose('0', camera_pose)

def check_obstacle(client: airsim.VehicleClient, direction: str) -> bool:
    """Check obstacle."""
    (yaw_deg, pitch_deg) = (0.0, 0.0)
    if direction == 'back':
        yaw_deg = 180.0
    elif direction == 'left':
        yaw_deg = -90.0
    elif direction == 'right':
        yaw_deg = 90.0
    if direction == 'up':
        pitch_deg = 90.0
    elif direction == 'down':
        pitch_deg = -90.0
    camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(np.radians(pitch_deg), 0.0, np.radians(yaw_deg)))
    client.simSetCameraPose('0', camera_pose)
    responses = client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.DepthPerspective, True)])
    depth_image = airsim.list_to_2d_float_array(responses[0].image_data_float, responses[0].width, responses[0].height)
    center_region = depth_image[4 * depth_image.shape[0] // 9:5 * depth_image.shape[0] // 9, 4 * depth_image.shape[1] // 9:5 * depth_image.shape[1] // 9]
    has_obstacle = center_region.mean() < 15
    set_camera_angle(client, 0)
    return has_obstacle

def parse_rgb_response(response) -> np.ndarray:
    """Parse rgb response."""
    if response.width == 0 or response.height == 0 or len(response.image_data_uint8) == 0:
        raise RuntimeError(f'Empty RGB response: w={response.width}, h={response.height}, bytes={len(response.image_data_uint8)}')
    img_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    pixel_count = response.width * response.height
    channels = len(img_1d) // pixel_count
    if channels not in (3, 4):
        raise RuntimeError(f'Unsupported RGB channel count: {channels}')
    img = img_1d.reshape(response.height, response.width, channels)
    if channels == 4:
        img = img[:, :, :3]
    return img

def resize_rgb_image(img: np.ndarray, size: int) -> np.ndarray:
    """Resize rgb image."""
    if img.shape[0] == size and img.shape[1] == size:
        return img.copy()
    if cv2 is not None:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if Image is not None:
        resample = getattr(Image, 'Resampling', Image).LANCZOS
        return np.asarray(Image.fromarray(img).resize((size, size), resample))
    y_idx = np.linspace(0, img.shape[0] - 1, size).astype(np.int64)
    x_idx = np.linspace(0, img.shape[1] - 1, size).astype(np.int64)
    return img[y_idx][:, x_idx].copy()

def write_png_image(file_path: str, img: np.ndarray) -> None:
    """Unicode-path safe PNG writer for Windows Chinese paths."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if cv2 is not None:
        (ok, encoded) = cv2.imencode('.png', img)
        if not ok:
            raise RuntimeError(f'PNG encode failed: {file_path}')
        with open(file_path, 'wb') as f:
            f.write(encoded.tobytes())
        return
    if Image is not None:
        Image.fromarray(img).save(file_path)
        return
    raise RuntimeError('Cannot write PNG: neither cv2 nor PIL is available')

def make_six_view_mosaic(view_images: Dict[str, np.ndarray]) -> np.ndarray:
    """Make six view mosaic."""
    order = ['left', 'front', 'right', 'up', 'back', 'down']
    row1 = np.hstack([view_images[name] for name in order[:3]])
    row2 = np.hstack([view_images[name] for name in order[3:]])
    return np.vstack([row1, row2])

def compute_intrinsic_matrix(size: int, fov_degrees: float=CAMERA_FOV_DEGREES) -> np.ndarray:
    """Compute intrinsic matrix."""
    focal = size / (2.0 * np.tan(np.radians(fov_degrees) / 2.0))
    center = size / 2.0
    return np.array([[focal, 0.0, center], [0.0, focal, center], [0.0, 0.0, 1.0]], dtype=np.float64)

def save_matrix_txt(file_path: str, matrix: np.ndarray) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.savetxt(file_path, matrix, fmt='%.10f')

def get_resolution_root_name(size: int) -> str:
    return f'rgb{size}'

def get_frame_name(frame_id: int) -> str:
    return f'{frame_id:06d}.png'

def compute_normalized_intrinsic_matrix(size: int) -> np.ndarray:
    """Compute normalized intrinsic matrix."""
    k_normalized = compute_intrinsic_matrix(size)
    k_normalized[0, 0] /= size
    k_normalized[1, 1] /= size
    k_normalized[0, 2] /= size
    k_normalized[1, 2] /= size
    return k_normalized

def save_intrinsics_files(pose_dir: str, size: int) -> None:
    """Save intrinsics files."""
    os.makedirs(pose_dir, exist_ok=True)
    save_matrix_txt(os.path.join(pose_dir, 'K_pixels.txt'), compute_intrinsic_matrix(size))
    save_matrix_txt(os.path.join(pose_dir, 'K_normalized.txt'), compute_normalized_intrinsic_matrix(size))

def prepare_resolution_dirs(chain_dir: str, size: int) -> str:
    """Prepare resolution dirs."""
    resolution_root = os.path.join(chain_dir, get_resolution_root_name(size))
    for (_, folder_name, _, _) in VIEW_CONFIGS:
        os.makedirs(os.path.join(resolution_root, folder_name), exist_ok=True)
    if SAVE_SIX_VIEW_MOSAIC:
        os.makedirs(os.path.join(resolution_root, 'six_views'), exist_ok=True)
    pose_dir = os.path.join(resolution_root, 'pose')
    save_intrinsics_files(pose_dir, size)
    return resolution_root

def open_pose_record_files(chain_dir: str, resolutions: List[int]) -> Dict[int, Dict[str, TextIO]]:
    """Open pose record files."""
    handles: Dict[int, Dict[str, TextIO]] = {}
    for size in resolutions:
        resolution_root = prepare_resolution_dirs(chain_dir, size)
        pose_dir = os.path.join(resolution_root, 'pose')
        handles[size] = {'trajectory': open(os.path.join(pose_dir, 'trajectory.txt'), 'w', encoding='utf-8', newline='\n'), 'pose': open(os.path.join(pose_dir, 'camera_pose.txt'), 'w', encoding='utf-8', newline='\n'), 'c2w': open(os.path.join(pose_dir, 'camera_to_world_matrix.txt'), 'w', encoding='utf-8', newline='\n'), 'w2c': open(os.path.join(pose_dir, 'world_to_camera_extrinsics.txt'), 'w', encoding='utf-8', newline='\n')}
        handles[size]['trajectory'].write('frame_id action action_name pos_x pos_y pos_z roll pitch yaw six_views_path\n')
        handles[size]['pose'].write('frame_id view image_path x y z roll pitch yaw qw qx qy qz\n')
        handles[size]['c2w'].write('frame_id view image_path c2w_4x4_row_major\n')
        handles[size]['w2c'].write('frame_id view image_path w2c_4x4_row_major\n')
    return handles

def close_pose_record_files(pose_record_files: Dict[int, Dict[str, TextIO]]) -> None:
    for files in pose_record_files.values():
        for file_obj in files.values():
            file_obj.close()

def multiply_quaternions(q1, q2):
    """Multiply quaternions."""
    (w1, x1, y1, z1) = (q1.w_val, q1.x_val, q1.y_val, q1.z_val)
    (w2, x2, y2, z2) = (q2.w_val, q2.x_val, q2.y_val, q2.z_val)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    return airsim.Quaternionr(x / norm, y / norm, z / norm, w / norm)

def quaternion_to_rotation_matrix(q) -> np.ndarray:
    """Quaternion to rotation matrix."""
    (w, x, y, z) = (q.w_val, q.x_val, q.y_val, q.z_val)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    (w, x, y, z) = (w / norm, x / norm, y / norm, z / norm)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=np.float64)

def get_view_camera_pose(client: airsim.VehicleClient, yaw_deg: float, pitch_deg: float):
    """Get view camera pose."""
    vehicle_pose = client.simGetVehiclePose()
    local_q = airsim.to_quaternion(np.radians(pitch_deg), 0.0, np.radians(yaw_deg))
    world_q = multiply_quaternions(vehicle_pose.orientation, local_q)
    return (vehicle_pose.position, world_q)

def write_view_pose_records(pose_file: TextIO, c2w_file: TextIO, w2c_file: TextIO, frame_id: int, view_name: str, image_path: str, position, orientation) -> None:
    """Write view pose records."""
    (x, y, z) = (float(position.x_val), float(position.y_val), float(position.z_val))
    (qw, qx, qy, qz) = (float(orientation.w_val), float(orientation.x_val), float(orientation.y_val), float(orientation.z_val))
    (roll, pitch, yaw) = quaternion_to_euler(orientation)
    rotation_c2w = quaternion_to_rotation_matrix(orientation)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation_c2w
    c2w[:3, 3] = [x, y, z]
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = rotation_c2w.T
    w2c[:3, 3] = -rotation_c2w.T @ np.array([x, y, z], dtype=np.float64)
    pose_file.write(f'{frame_id} {view_name} {image_path} {x:.10f} {y:.10f} {z:.10f} {roll:.10f} {pitch:.10f} {yaw:.10f} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f}\n')
    c2w_file.write(f'{frame_id} {view_name} {image_path} ' + ' '.join((f'{v:.10f}' for v in c2w.reshape(-1))) + '\n')
    w2c_file.write(f'{frame_id} {view_name} {image_path} ' + ' '.join((f'{v:.10f}' for v in w2c.reshape(-1))) + '\n')

def save_images(client: airsim.VehicleClient, chain_dir: str, frame_id: int, resolutions: List[int], pose_record_files: Dict[int, Dict[str, TextIO]]) -> Dict[int, str]:
    """Capture front/back/left/right/up/down views and record camera poses.

    Each 320px view contributes to a 640x960 six-view training mosaic.
    """
    frame_name = get_frame_name(frame_id)
    mosaic_buffers: Dict[int, Dict[str, np.ndarray]] = {size: {} for size in resolutions} if SAVE_SIX_VIEW_MOSAIC else {}
    six_view_paths: Dict[int, str] = {size: '' for size in resolutions}
    for (view_name, folder_name, yaw_deg, pitch_deg) in VIEW_CONFIGS:
        camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(np.radians(pitch_deg), 0.0, np.radians(yaw_deg)))
        client.simSetCameraPose('0', camera_pose)
        response = client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]
        img_data = parse_rgb_response(response)
        for size in resolutions:
            resized = resize_rgb_image(img_data, size)
            resolution_root_name = get_resolution_root_name(size)
            image_rel_path = f'{folder_name}/{frame_name}'
            out_path = os.path.join(chain_dir, resolution_root_name, folder_name, frame_name)
            write_png_image(out_path, resized)
            if SAVE_SIX_VIEW_MOSAIC:
                mosaic_buffers[size][view_name] = resized
        (view_position, view_orientation) = get_view_camera_pose(client, yaw_deg, pitch_deg)
        for size in resolutions:
            image_rel_path = f'{folder_name}/{frame_name}'
            files = pose_record_files[size]
            write_view_pose_records(pose_file=files['pose'], c2w_file=files['c2w'], w2c_file=files['w2c'], frame_id=frame_id, view_name=view_name, image_path=image_rel_path, position=view_position, orientation=view_orientation)
    if SAVE_SIX_VIEW_MOSAIC:
        for size in resolutions:
            mosaic = make_six_view_mosaic(mosaic_buffers[size])
            resolution_root_name = get_resolution_root_name(size)
            six_view_path = f'six_views/{frame_name}'
            write_png_image(os.path.join(chain_dir, resolution_root_name, six_view_path), mosaic)
            six_view_paths[size] = six_view_path
    for files in pose_record_files.values():
        files['pose'].flush()
        files['c2w'].flush()
        files['w2c'].flush()
    set_camera_angle(client, 0)
    return six_view_paths

def write_trajectory_records(pose_record_files: Dict[int, Dict[str, TextIO]], frame_id: int, action: int, action_name: str, pos: np.ndarray, ori: Tuple[float, float, float], six_view_paths: Dict[int, str]) -> None:
    for (size, files) in pose_record_files.items():
        files['trajectory'].write(f'{frame_id} {action} {action_name} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f} {ori[0]:.10f} {ori[1]:.10f} {ori[2]:.10f} {six_view_paths[size]}\n')
        files['trajectory'].flush()

def quaternion_to_euler(q) -> Tuple[float, float, float]:
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
if quaternion2eularian_angles is None:

    def quaternion2eularian_angles(q):
        return quaternion_to_euler(q)

def get_current_state(client: airsim.VehicleClient) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Get current state."""
    state = client.simGetGroundTruthKinematics()
    pos = state.position.to_numpy_array()
    ori = quaternion2eularian_angles(state.orientation)
    return (pos, ori)

def set_vehicle_pose(client: airsim.VehicleClient, position: Union[np.ndarray, List[float]], orientation: Tuple[float, float, float]) -> None:
    """Set a world pose; convert internal roll/pitch/yaw to AirSim pitch/roll/yaw."""
    pose = airsim.Pose(airsim.Vector3r(*position), airsim.to_quaternion(orientation[1], orientation[0], orientation[2]))
    client.simSetVehiclePose(pose, True)

def local_delta_to_world(yaw: float, dx: float, dy: float, dz: float) -> np.ndarray:
    """Rotate a body-frame displacement into the AirSim NED world frame."""
    forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    right = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    down = np.array([0.0, 0.0, 1.0])
    return dx * forward + dy * right + dz * down

def move_relative(client: airsim.VehicleClient, dx: float, dy: float, dz: float) -> None:
    """Move relative."""
    pose = client.simGetVehiclePose()
    orientation = airsim.to_eularian_angles(pose.orientation)
    yaw = orientation[2]
    move_vector = local_delta_to_world(yaw, dx, dy, dz)
    new_position = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val]) + move_vector
    set_vehicle_pose(client, new_position, orientation)

def perform_action(client: airsim.VehicleClient, action: int) -> bool:
    """Perform action."""
    action_delta = get_action_local_delta(action)
    if action_delta is None:
        print(f'Unknown action {action}')
        return False
    move_relative(client, *action_delta)
    return True

def get_action_local_delta(action: int) -> Optional[Tuple[float, float, float]]:
    """Get action local delta."""
    action_delta_map = {6: (10.0, 0.0, 0.0), 7: (-10.0, 0.0, 0.0), 8: (0.0, -10.0, 0.0), 9: (0.0, 10.0, 0.0), 10: (0.0, 0.0, -10.0), 11: (0.0, 0.0, 10.0)}
    return action_delta_map.get(action)

def get_action_substep_delta(action: int) -> Optional[Tuple[float, float, float]]:
    action_delta = get_action_local_delta(action)
    if action_delta is None:
        return None
    return tuple((float(delta) / ACTION_FRAMES_PER_STEP for delta in action_delta))

def check_boundary(pos: np.ndarray, boundary_point1: List[float], boundary_point2: List[float]) -> bool:
    """Check boundary."""
    return pos[0] < boundary_point2[0] or pos[0] > boundary_point1[0] or pos[1] < boundary_point2[1] or (pos[1] > boundary_point1[1]) or (pos[2] < boundary_point2[2]) or (pos[2] > boundary_point1[2])

def will_exceed_boundary(client: airsim.VehicleClient, action: int, boundary_point1: List[float], boundary_point2: List[float]) -> bool:
    """Will exceed boundary."""
    (current_pos, current_ori) = get_current_state(client)
    yaw = current_ori[2]
    action_delta = get_action_local_delta(action)
    if action_delta is None:
        return True
    expected_pos = current_pos + local_delta_to_world(yaw, *action_delta)
    return check_boundary(expected_pos, boundary_point1, boundary_point2)

def check_surrounded_by_obstacles(client: airsim.VehicleClient, action_set: List[int]) -> bool:
    """Check surrounded by obstacles."""
    directions_to_check = {6: 'forward', 7: 'back', 8: 'left', 9: 'right', 10: 'up', 11: 'down'}
    checked_directions = [directions_to_check[action] for action in action_set if action in directions_to_check]
    return all((check_obstacle(client, direction) for direction in checked_directions))

def check_back_to_start(current_pos: np.ndarray, start_pos: np.ndarray, threshold: float=15.0) -> bool:
    """Check back to start."""
    distance = np.linalg.norm(current_pos - start_pos)
    return distance < threshold

def collect_chain_from_start_point(client: airsim.VehicleClient, start_point: List[float], start_yaw: float, chain_id: int, max_steps: int=200, boundary_point1: List[float]=None, boundary_point2: List[float]=None) -> None:
    """Collect chain from start point."""
    if boundary_point1 is None:
        boundary_point1 = [990.85, 374.149, 0]
    if boundary_point2 is None:
        boundary_point2 = [-574.949, -723.392, -131.831]
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(start_point[0], start_point[1], start_point[2]), airsim.to_quaternion(0.0, 0, start_yaw)), False)
    set_camera_angle(client, 0)
    time.sleep(0.1)
    (start_pos, start_ori) = get_current_state(client)
    print(f'\n=== Chain {chain_id} started ===')
    print(f'Start position: {start_pos}')
    print(f'Start orientation: {start_ori}')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    chain_dir = os.path.join(path, f'chain_{chain_id}_{timestamp}')
    os.makedirs(chain_dir, exist_ok=True)
    active_resolutions = get_active_resolutions()
    pose_record_files = open_pose_record_files(chain_dir, active_resolutions)
    csv_path = os.path.join(chain_dir, f'chain_{chain_id}.csv')
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_header = ['frame_id', 'action', 'action_name', 'pos_x', 'pos_y', 'pos_z', 'roll', 'pitch', 'yaw']
    for size in active_resolutions:
        csv_header.extend([f'rgb_root_{size}', f'six_views_{size}'])
    csv_writer.writerow(csv_header)
    start_six_view_paths = save_images(client=client, chain_dir=chain_dir, frame_id=0, resolutions=active_resolutions, pose_record_files=pose_record_files)
    print(f'Initial images saved to: {os.path.join(chain_dir, get_resolution_root_name(active_resolutions[0]))}')
    start_row = [0, -1, 'start', start_pos[0], start_pos[1], start_pos[2], start_ori[0], start_ori[1], start_ori[2]]
    for size in active_resolutions:
        start_row.extend([get_resolution_root_name(size), start_six_view_paths[size]])
    csv_writer.writerow(start_row)
    write_trajectory_records(pose_record_files=pose_record_files, frame_id=0, action=-1, action_name='start', pos=start_pos, ori=start_ori, six_view_paths=start_six_view_paths)
    action_set = [6, 7, 8, 9, 10, 11]
    action_names = {6: 'move_forth', 7: 'move_back', 8: 'move_left', 9: 'move_right', 10: 'move_up', 11: 'move_down'}
    step = 0
    frame_id = 0
    consecutive_failed_actions = 0
    action_probabilities = [1.0 / len(action_set)] * len(action_set)
    while step < max_steps:
        action = np.random.choice(action_set, p=action_probabilities)
        action_name = action_names[action]
        print(f'\nStep {step + 1}/{max_steps}: executing action {action_name} ({action})')
        if will_exceed_boundary(client, action, boundary_point1, boundary_point2):
            print(f'  Action {action_name} would exceed bounds; skipping')
            continue
        if action == 6:
            check_dir = 'forward'
        elif action == 7:
            check_dir = 'back'
        elif action == 8:
            check_dir = 'left'
        elif action == 9:
            check_dir = 'right'
        elif action == 10:
            check_dir = 'up'
        elif action == 11:
            check_dir = 'down'
        else:
            check_dir = 'forward'
        if check_obstacle(client, check_dir):
            print(f'  Action {action_name} direction blocked; skipping')
            consecutive_failed_actions += 1
            if consecutive_failed_actions >= 10:
                print(f'  Consecutive failures: {consecutive_failed_actions}; checking surrounding obstacles')
                if check_surrounded_by_obstacles(client, action_set):
                    print(f'  Obstacles in all directions; ending chain')
                    break
            continue
        consecutive_failed_actions = 0
        substep_delta = get_action_substep_delta(action)
        if substep_delta is None:
            print(f'  Unknown action {action}; skipping')
            continue
        step += 1
        action_stopped = False
        for interp_id in range(1, ACTION_FRAMES_PER_STEP + 1):
            frame_id += 1
            move_relative(client, *substep_delta)
            time.sleep(0.02)
            (current_pos, current_ori) = get_current_state(client)
            if check_boundary(current_pos, boundary_point1, boundary_point2):
                print(f'  Warning: action {action_name} interpolated frame {interp_id}/{ACTION_FRAMES_PER_STEP} exceeds bounds!')
                print(f'  Current position: [{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}]')
                print(f'  Chain terminated')
                action_stopped = True
                break
            step_six_view_paths = save_images(client=client, chain_dir=chain_dir, frame_id=frame_id, resolutions=active_resolutions, pose_record_files=pose_record_files)
            csv_row = [frame_id, action, action_name, current_pos[0], current_pos[1], current_pos[2], current_ori[0], current_ori[1], current_ori[2]]
            for size in active_resolutions:
                csv_row.extend([get_resolution_root_name(size), step_six_view_paths[size]])
            csv_writer.writerow(csv_row)
            write_trajectory_records(pose_record_files=pose_record_files, frame_id=frame_id, action=int(action), action_name=action_name, pos=current_pos, ori=current_ori, six_view_paths=step_six_view_paths)
            csv_file.flush()
            print(f'  Action {step}/{max_steps} interpolated frame {interp_id}/{ACTION_FRAMES_PER_STEP} frame={frame_id:06d} pos=[{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}]')
        if action_stopped:
            break
    csv_file.close()
    close_pose_record_files(pose_record_files)
    (final_pos, final_ori) = get_current_state(client)
    with open(csv_path, 'r', encoding='utf-8') as f:
        existing_data = f.readlines()
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        f.write(f'# Chain ID: {chain_id}\n')
        f.write(f'# Start Position: {start_pos[0]:.6f}, {start_pos[1]:.6f}, {start_pos[2]:.6f}\n')
        f.write(f'# Start Orientation: {start_ori[0]:.6f}, {start_ori[1]:.6f}, {start_ori[2]:.6f}\n')
        f.write(f'# End Position: {final_pos[0]:.6f}, {final_pos[1]:.6f}, {final_pos[2]:.6f}\n')
        f.write(f'# End Orientation: {final_ori[0]:.6f}, {final_ori[1]:.6f}, {final_ori[2]:.6f}\n')
        f.write(f'# Total Actions: {step}\n')
        f.write(f'# Total Frames: {frame_id}\n')
        f.write(f'# Action Distance Meters: {ACTION_DISTANCE_METERS:.6f}\n')
        f.write(f'# Action Frames Per Step: {ACTION_FRAMES_PER_STEP}\n')
        f.write(f'# Action Sample Interval Meters: {ACTION_SAMPLE_INTERVAL_METERS:.6f}\n')
        f.write('# Frame ID Rule: frame 0 is start; frames 1-20 are action 1; frames 21-40 are action 2; and so on.\n')
        f.write(f'# Timestamp: {timestamp}\n')
        f.write(f"# Target Resolutions: {', '.join((str(r) for r in TARGET_RESOLUTIONS))}\n")
        f.write(f"# Active Resolutions: {', '.join((str(r) for r in active_resolutions))}\n")
        f.write(f'# Collect All Resolutions: {COLLECT_ALL_RESOLUTIONS}\n')
        f.write(f'# Save Six View Mosaic: {SAVE_SIX_VIEW_MOSAIC}\n')
        f.write(f'# Camera FOV Degrees: {CAMERA_FOV_DEGREES:.6f}\n')
        f.write('#\n')
        f.writelines(existing_data)
    print(f'\n=== Chain {chain_id} completed ===')
    print(f'Total steps: {step}')
    print(f'Total frames: {frame_id}')
    print(f'Frames sampled per action: {ACTION_FRAMES_PER_STEP}, sample spacing: {ACTION_SAMPLE_INTERVAL_METERS:.2f}m')
    print(f'Start position: {start_pos}')
    print(f'Final position: {final_pos}')
    print(f'Data saved to: {chain_dir}')
    print(f'CSV file: {csv_path}')

def main() -> None:
    """Main."""
    global ENV_NAME, START_POINTS_CSV_DIR, SELECT_START_POINTS
    global START_POINT_RANGE_START, START_POINT_RANGE_COUNT
    global SAVE_SIX_VIEW_MOSAIC, TARGET_RESOLUTIONS, COLLECT_ALL_RESOLUTIONS, path
    args = parse_args()
    ENV_NAME = args.env_name
    START_POINTS_CSV_DIR = args.start_points_dir
    path = args.out_root
    SAVE_SIX_VIEW_MOSAIC = args.save_six_view_mosaic
    if args.resolutions is not None:
        TARGET_RESOLUTIONS = args.resolutions
        COLLECT_ALL_RESOLUTIONS = True
    if args.range_start is not None or args.range_count is not None:
        SELECT_START_POINTS = True
        START_POINT_INDEXES.clear()
        START_POINT_RANGE_START = 0 if args.range_start is None else args.range_start
        START_POINT_RANGE_COUNT = args.range_count
    os.makedirs(path, exist_ok=True)
    boundary_point1 = [100, 100, 0]
    boundary_point2 = [-100, -100, -50]
    manual_start_points = [[0, 0, -10]]
    if USE_CSV_START_POINTS:
        start_points_csv = get_start_points_csv_path(ENV_NAME)
        start_point_entries = load_start_points_from_csv(start_points_csv, limit=START_POINTS_LIMIT)
    else:
        start_points_csv = ''
        start_point_entries = [(f'{idx:03d}', point, 0.0) for (idx, point) in enumerate(manual_start_points)]
    total_start_points = len(start_point_entries)
    start_point_entries = select_start_point_entries(start_point_entries)
    if AUTO_BOUNDARY_FROM_CSV and USE_CSV_START_POINTS:
        (boundary_point1, boundary_point2) = estimate_boundary_from_start_points(start_point_entries)
    max_steps_per_chain = args.steps
    print(f'\nStarting chain collection')
    if USE_CSV_START_POINTS:
        print(f'Environment: {ENV_NAME}')
        print(f'Start-point CSV: {start_points_csv}')
    print(f'Start-point selection: {SELECT_START_POINTS} ({len(start_point_entries)}/{total_start_points})')
    if SELECT_START_POINTS:
        print(f'Selected start-point IDs: {[entry[0] for entry in start_point_entries]}')
    print(f'Visiting all {len(start_point_entries)} start points')
    print(f'Maximum per chain: {max_steps_per_chain} actions')
    print(f'Per action: {ACTION_DISTANCE_METERS:.1f}m, sampled every {ACTION_SAMPLE_INTERVAL_METERS:.2f}m; total {ACTION_FRAMES_PER_STEP} frames/action')
    print(f"Capture resolutions: {', '.join((str(r) for r in get_active_resolutions()))}")
    print(f'Save six-view mosaic: {SAVE_SIX_VIEW_MOSAIC}')
    print(f'Automatic bounds: {AUTO_BOUNDARY_FROM_CSV and USE_CSV_START_POINTS}')
    print(f'Bounds: X[{boundary_point2[0]:.1f}, {boundary_point1[0]:.1f}], Y[{boundary_point2[1]:.1f}, {boundary_point1[1]:.1f}], Z[{boundary_point2[2]:.1f}, {boundary_point1[2]:.1f}]')
    print(f'Actions: move_forth(6), move_back(7), move_left(8), move_right(9), move_up(10), move_down(11)')
    client = airsim.VehicleClient(port=args.port)
    client.confirmConnection()
    print('Connected to AirSim')
    chain_id_offset = START_POINT_RANGE_START - START_POINT_INDEX_BASE if SELECT_START_POINTS and (not START_POINT_INDEXES) else 0
    for (chain_id, (start_id, start_point, csv_yaw)) in enumerate(start_point_entries, start=chain_id_offset):
        if USE_CSV_START_POINTS and USE_CSV_YAW:
            start_yaw = csv_yaw
        else:
            start_yaw = random.choice([0, np.pi / 2, np.pi, -np.pi / 2])
        local_chain_no = chain_id - chain_id_offset + 1
        print(f"\n{'=' * 60}")
        print(f'Collecting chain {local_chain_no}/{len(start_point_entries)} chains')
        print(f'chain_id: {chain_id}')
        print(f'Start-point ID: {start_id}')
        print(f'Start point: {start_point}')
        print(f'Start orientation: {np.degrees(start_yaw):.1f} degrees')
        print(f"{'=' * 60}")
        try:
            collect_chain_from_start_point(client=client, start_point=start_point, start_yaw=start_yaw, chain_id=chain_id, max_steps=max_steps_per_chain, boundary_point1=boundary_point1, boundary_point2=boundary_point2)
        except Exception as e:
            print(f'\n!!! Chain {chain_id} failed during collection: {e}')
            print('Continuing with the next chain...')
            continue
        time.sleep(0.5)
    print(f"\n{'=' * 60}")
    print(f'All chains completed!')
    print(f'Output directory: {path}')
    print(f"{'=' * 60}")
if __name__ == '__main__':
    main()
