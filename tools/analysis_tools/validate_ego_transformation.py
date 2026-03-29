#!/usr/bin/env python3
"""Validation script for ego-motion transformation.

This script validates that the ego-motion transformation is working correctly
by comparing raw labels with transformed labels.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def get_ego_matrix(ego_traj_path: str) -> np.ndarray:
    """Load ego-motion matrix from trajectory file."""
    with open(ego_traj_path, 'r') as f:
        lines = f.readlines()
    return np.array([float(i) for i in lines[0].split(",")]).reshape(4, 4)


def rotmat_to_euler(rot_mat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to Euler angles."""
    sy = np.sqrt(rot_mat[0, 0] * rot_mat[0, 0] + rot_mat[1, 0] * rot_mat[1, 0])
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
        pitch = np.arctan2(-rot_mat[2, 0], sy)
        yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
    else:
        roll = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
        pitch = np.arctan2(-rot_mat[2, 0], sy)
        yaw = 0
        
    return np.array([roll, pitch, yaw])


def transform_boxes_world_to_lidar(gt_boxes: np.ndarray, ego_motion: np.ndarray) -> np.ndarray:
    """Transform bounding boxes from world to LiDAR coordinates."""
    ego_yaw = rotmat_to_euler(ego_motion[:3, :3])[2]
    gt_boxes[:, 6] -= ego_yaw
    
    centers_world = gt_boxes[:, :3]
    centers_world_homog = np.concatenate(
        [centers_world, np.ones((centers_world.shape[0], 1))], 
        axis=1
    )
    centers_lidar_homog = np.matmul(
        np.linalg.inv(ego_motion), 
        centers_world_homog.T
    ).T
    gt_boxes[:, :3] = centers_lidar_homog[:, :3]
    
    return gt_boxes


def parse_sit_label_line(line: str) -> dict:
    """Parse a single SiT label line."""
    parts = line.strip().split()
    if len(parts) != 9:
        raise ValueError(f"Expected 9 fields, got {len(parts)}: {line}")

    class_name, instance_token, height, width, length, x, y, z, yaw = parts

    return {
        'class_name': class_name,
        'instance_token': instance_token,
        'dimensions': [float(height), float(width), float(length)],  # h, w, l
        'location': [float(x), float(y), float(z)],  # x, y, z
        'rotation_y': float(yaw)
    }


def validate_transformation(dataset_root: str, scene: str, frame: str):
    """Validate ego-motion transformation for a specific frame.
    
    Args:
        dataset_root (str): Root directory of raw SiT dataset.
        scene (str): Scene name (e.g., 'Cafe_street_1-002').
        frame (str): Frame ID (e.g., '0').
    """
    dataset_path = Path(dataset_root)
    scene_path = dataset_path / scene
    
    # Paths
    label_file = scene_path / 'label_3d' / f'{frame}.txt'
    ego_file = scene_path / 'ego_trajectory' / f'{frame}.txt'
    
    print("=" * 80)
    print("EGO-MOTION TRANSFORMATION VALIDATION")
    print("=" * 80)
    print(f"Dataset root: {dataset_root}")
    print(f"Scene:        {scene}")
    print(f"Frame:        {frame}")
    print()
    
    # Check files exist
    if not label_file.exists():
        print(f"❌ ERROR: Label file not found: {label_file}")
        return False
    
    if not ego_file.exists():
        print(f"❌ ERROR: Ego trajectory file not found: {ego_file}")
        return False
    
    print(f"✅ Label file found: {label_file}")
    print(f"✅ Ego trajectory file found: {ego_file}")
    print()
    
    # Load and parse labels
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        print("⚠️  WARNING: No labels in file")
        return True
    
    # Parse first label
    first_line = lines[0].strip()
    print("Raw label line:")
    print(f"  {first_line}")
    print()
    
    try:
        label_data = parse_sit_label_line(first_line)
    except Exception as e:
        print(f"❌ ERROR: Failed to parse label: {e}")
        return False
    
    # Extract world coordinates
    x_world, y_world, z_world = label_data['location']
    h, w, l = label_data['dimensions']
    yaw_world = label_data['rotation_y']
    
    print("Parsed label (WORLD frame):")
    print(f"  Class:       {label_data['class_name']}")
    print(f"  Position:    x={x_world:.3f}, y={y_world:.3f}, z={z_world:.3f}")
    print(f"  Dimensions:  h={h:.3f}, w={w:.3f}, l={l:.3f}")
    print(f"  Yaw:         {yaw_world:.3f} rad ({np.degrees(yaw_world):.1f}°)")
    print()
    
    # Load ego-motion matrix
    try:
        ego_motion = get_ego_matrix(str(ego_file))
    except Exception as e:
        print(f"❌ ERROR: Failed to load ego-motion matrix: {e}")
        return False
    
    print("Ego-motion matrix:")
    print(ego_motion)
    print()
    
    # Extract ego translation and rotation
    ego_translation = ego_motion[:3, 3]
    ego_rotation = ego_motion[:3, :3]
    ego_yaw = rotmat_to_euler(ego_rotation)[2]
    
    print("Ego pose (WORLD frame):")
    print(f"  Position:    x={ego_translation[0]:.3f}, y={ego_translation[1]:.3f}, z={ego_translation[2]:.3f}")
    print(f"  Yaw:         {ego_yaw:.3f} rad ({np.degrees(ego_yaw):.1f}°)")
    print()
    
    # Manual transformation for verification
    print("Manual transformation verification:")
    print("-" * 40)
    
    # Position transformation
    pos_world = np.array([x_world, y_world, z_world, 1.0])
    pos_lidar_homog = np.linalg.inv(ego_motion) @ pos_world
    pos_lidar_manual = pos_lidar_homog[:3]
    
    # Orientation transformation
    yaw_lidar_manual = yaw_world - ego_yaw
    
    print(f"Position (LIDAR frame):  x={pos_lidar_manual[0]:.3f}, y={pos_lidar_manual[1]:.3f}, z={pos_lidar_manual[2]:.3f}")
    print(f"Yaw (LIDAR frame):       {yaw_lidar_manual:.3f} rad ({np.degrees(yaw_lidar_manual):.1f}°)")
    print()
    
    # Apply transformation using our function
    print("Transformation using our function:")
    print("-" * 40)
    
    gt_boxes = np.array([[x_world, y_world, z_world, l, w, h, yaw_world]], dtype=np.float32)
    try:
        gt_boxes_transformed = transform_boxes_world_to_lidar(gt_boxes, ego_motion)
    except Exception as e:
        print(f"❌ ERROR: Transformation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    x_lidar, y_lidar, z_lidar, l_out, w_out, h_out, yaw_lidar = gt_boxes_transformed[0]
    
    print(f"Position (LIDAR frame):  x={x_lidar:.3f}, y={y_lidar:.3f}, z={z_lidar:.3f}")
    print(f"Dimensions:              l={l_out:.3f}, w={w_out:.3f}, h={h_out:.3f}")
    print(f"Yaw (LIDAR frame):       {yaw_lidar:.3f} rad ({np.degrees(yaw_lidar):.1f}°)")
    print()
    
    # Compare manual vs function
    print("Comparison:")
    print("-" * 40)
    pos_diff = np.abs(pos_lidar_manual - np.array([x_lidar, y_lidar, z_lidar]))
    yaw_diff = np.abs(yaw_lidar_manual - yaw_lidar)
    
    print(f"Position difference:     dx={pos_diff[0]:.6f}, dy={pos_diff[1]:.6f}, dz={pos_diff[2]:.6f}")
    print(f"Position max error:      {pos_diff.max():.6f} m")
    print(f"Yaw difference:          {yaw_diff:.6f} rad ({np.degrees(yaw_diff):.3f}°)")
    print()
    
    # Validation
    tolerance_pos = 0.001  # 1mm tolerance
    tolerance_yaw = 0.01   # ~0.5 degree tolerance
    
    if pos_diff.max() < tolerance_pos and yaw_diff < tolerance_yaw:
        print("=" * 80)
        print("✅ VALIDATION PASSED: Transformation is correct!")
        print("=" * 80)
        print()
        print("Summary:")
        print(f"  - Position error:  {pos_diff.max():.6f} m (< {tolerance_pos} m) ✅")
        print(f"  - Yaw error:       {np.degrees(yaw_diff):.3f}° (< {np.degrees(tolerance_yaw):.1f}°) ✅")
        print()
        print("The ego-motion transformation is working correctly.")
        print("You can now regenerate the dataset with confidence.")
        return True
    else:
        print("=" * 80)
        print("❌ VALIDATION FAILED: Transformation has errors!")
        print("=" * 80)
        print()
        print("Summary:")
        if pos_diff.max() >= tolerance_pos:
            print(f"  - Position error:  {pos_diff.max():.6f} m (>= {tolerance_pos} m) ❌")
        if yaw_diff >= tolerance_yaw:
            print(f"  - Yaw error:       {np.degrees(yaw_diff):.3f}° (>= {np.degrees(tolerance_yaw):.1f}°) ❌")
        print()
        print("Please check the transformation code.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Validate ego-motion transformation')
    parser.add_argument('--dataset-root', required=True,
                       help='Root directory of raw SiT dataset')
    parser.add_argument('--scene', default='Cafe_street_1-002',
                       help='Scene name (default: Cafe_street_1-002)')
    parser.add_argument('--frame', default='0',
                       help='Frame ID (default: 0)')
    
    args = parser.parse_args()
    
    success = validate_transformation(args.dataset_root, args.scene, args.frame)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

