#!/usr/bin/env python3
"""
Simple test script to convert a few SiT samples for testing PointPillars pipeline
"""

import os
import numpy as np
import open3d as o3d
from pathlib import Path

def load_pcd_file(pcd_path):
    """Load PCD file and return points as numpy array."""
    try:
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        points = np.asarray(pcd.points)

        # Handle intensity - check if colors contain intensity data
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            intensity = (colors * 255).astype(np.uint8)
            intensity = intensity.mean(axis=1).astype(np.float32)
        else:
            intensity = np.zeros(len(points), dtype=np.float32)

        # Combine x, y, z, intensity
        points_with_intensity = np.column_stack([points, intensity])
        return points_with_intensity.astype(np.float32)
    except Exception as e:
        print(f"Error loading {pcd_path}: {e}")
        return np.empty((0, 4), dtype=np.float32)

def convert_pcd_to_bin(pcd_path, bin_path):
    """Convert PCD file to KITTI .bin format."""
    points = load_pcd_file(pcd_path)
    if len(points) == 0:
        return False

    # Ensure we have 4 dimensions (x, y, z, intensity)
    if points.shape[1] < 4:
        padding = np.zeros((points.shape[0], 4 - points.shape[1]), dtype=np.float32)
        points = np.column_stack([points, padding])

    # Save as binary float32
    points.astype(np.float32).tofile(bin_path)
    return True

def main():
    # Sample data paths
    sit_root = Path("/sit-sample")
    output_root = Path("data/sit_test")

    # Create output directories
    (output_root / "training" / "velodyne").mkdir(parents=True, exist_ok=True)
    (output_root / "training" / "label_2").mkdir(parents=True, exist_ok=True)
    (output_root / "training" / "calib").mkdir(parents=True, exist_ok=True)
    (output_root / "training" / "image_2").mkdir(parents=True, exist_ok=True)

    # Process just a few samples from Cafe_street_1-002
    sequence_dir = sit_root / "Cafe_street_1-002"
    velo_dir = sequence_dir / "velo" / "concat" / "data"
    label_dir = sequence_dir / "label_3d"
    calib_dir = sequence_dir / "calib"

    # Convert first 5 frames
    sample_frames = ["0", "1", "2", "3", "4"]

    for frame_idx in sample_frames:
        print(f"Converting frame {frame_idx}...")

        # Convert PCD to .bin
        pcd_path = velo_dir / f"{frame_idx}.pcd"
        bin_path = output_root / "training" / "velodyne" / f"{frame_idx}.bin"

        if pcd_path.exists():
            if convert_pcd_to_bin(pcd_path, bin_path):
                print(f"  ✓ Converted {pcd_path} -> {bin_path}")
            else:
                print(f"  ✗ Failed to convert {pcd_path}")
        else:
            print(f"  ✗ PCD file not found: {pcd_path}")

        # Copy label file (simplified - just create empty files for now)
        label_path = output_root / "training" / "label_2" / f"{frame_idx}.txt"
        with open(label_path, 'w') as f:
            # Create empty label file or add some dummy data
            f.write("")  # Empty for now

        # Create dummy calibration file
        calib_path = output_root / "training" / "calib" / f"{frame_idx}.txt"
        calib_content = """P0: 1 0 0 0 0 1 0 0 0 0 1 0
P1: 1 0 0 0 0 1 0 0 0 0 1 0
P2: 1 0 0 0 0 1 0 0 0 0 1 0
P3: 1 0 0 0 0 1 0 0 0 0 1 0
R0_rect: 1 0 0 0 1 0 0 0 1
Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0
Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0
"""
        with open(calib_path, 'w') as f:
            f.write(calib_content)

        # Create dummy image file
        image_path = output_root / "training" / "image_2" / f"{frame_idx}.png"
        Path(image_path).touch()

    print("Sample conversion complete!")

    # Create minimal info files
    import mmengine

    # Create basic info structure
    metainfo = {
        'categories': {'Pedestrian': 0, 'Car': 1},
        'dataset': 'sit',
        'info_version': '1.1'
    }

    # Create simple data list
    data_list = []
    for i, frame_idx in enumerate(sample_frames):
        info = {
            'sample_idx': int(frame_idx),
            'lidar_points': {
                'num_pts_feats': 4,
                'lidar_path': f'training/velodyne/{frame_idx}.bin'
            },
            'image': {
                'image_idx': int(frame_idx),
                'image_shape': [1024, 1024],
                'image_path': f'training/image_2/{frame_idx}.png'
            },
            'calib': {
                'R0_rect': np.eye(4, dtype=np.float32),
                'Tr_velo_to_cam': np.eye(4, dtype=np.float32),
                'P2': np.eye(3, 4, dtype=np.float32),
                'Tr_imu_to_velo': np.eye(4, dtype=np.float32)
            }
        }
        data_list.append(info)

    # Save info files
    train_info = {
        'metainfo': metainfo,
        'data_list': data_list
    }

    mmengine.dump(train_info, output_root / 'sit_infos_train.pkl')
    mmengine.dump(train_info, output_root / 'sit_infos_val.pkl')  # Use same for val

    print(f"Created info files with {len(data_list)} samples")

if __name__ == "__main__":
    main()
