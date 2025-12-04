# Copyright (c) OpenMMLab. All rights reserved.
"""SiT Dataset Converter.

Convert SiT dataset format to MMDetection3D compatible format.
"""

import argparse
import os
import os.path as osp
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mmengine
import numpy as np
from mmdet3d.structures import points_cam2img
from mmdet3d.structures.ops import box_np_ops

# Try to import open3d for PCD loading, fallback to manual parsing
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    try:
        import lzf
        HAS_LZF = True
    except ImportError:
        HAS_LZF = False


def load_pcd_file(pcd_path: str) -> np.ndarray:
    """Load PCD file and return points as numpy array.

    Args:
        pcd_path (str): Path to PCD file.

    Returns:
        np.ndarray: Point cloud data [N, 4] (x, y, z, intensity).
    """
    if HAS_OPEN3D:
        # Use open3d for PCD loading
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)

        # Handle intensity - check if colors contain intensity data
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            # Assume colors are normalized intensity values
            intensity = (colors * 255).astype(np.uint8)
            intensity = intensity.mean(axis=1).astype(np.float32)  # Average RGB to intensity
        else:
            intensity = np.zeros(len(points), dtype=np.float32)

        # Combine x, y, z, intensity
        points_with_intensity = np.column_stack([points, intensity])
        return points_with_intensity.astype(np.float32)
    else:
        # Manual PCD parsing (limited support)
        print(f"Warning: open3d not available, manual PCD parsing not implemented. "
              f"Skipping {pcd_path}")
        return np.empty((0, 4), dtype=np.float32)


def convert_pcd_to_bin(pcd_path: str, bin_path: str) -> bool:
    """Convert PCD file to KITTI .bin format.

    Args:
        pcd_path (str): Path to input PCD file.
        bin_path (str): Path to output .bin file.

    Returns:
        bool: True if conversion successful.
    """
    try:
        points = load_pcd_file(pcd_path)
        if len(points) == 0:
            print(f"Warning: No points loaded from {pcd_path}")
            return False

        # Ensure we have 4 dimensions (x, y, z, intensity)
        if points.shape[1] < 4:
            # Pad with zeros for intensity if missing
            padding = np.zeros((points.shape[0], 4 - points.shape[1]), dtype=np.float32)
            points = np.column_stack([points, padding])

        # Save as binary float32
        points.astype(np.float32).tofile(bin_path)
        return True

    except Exception as e:
        print(f"Error converting {pcd_path} to {bin_path}: {e}")
        return False


def parse_sit_label_line(line: str) -> Dict:
    """Parse a single SiT label line.

    Args:
        line (str): Label line in format: class_name instance_token height width length x y z yaw

    Returns:
        dict: Parsed label information.
    """
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


def convert_label_3d_to_kitti(sit_label_path: str, kitti_label_path: str,
                             calib_data: Optional[Dict] = None) -> bool:
    """Convert SiT 3D labels to KITTI format.

    Args:
        sit_label_path (str): Path to SiT label file.
        kitti_label_path (str): Path to output KITTI label file.
        calib_data (dict, optional): Calibration data for coordinate transformations.

    Returns:
        bool: True if conversion successful.
    """
    try:
        with open(sit_label_path, 'r') as f:
            lines = f.readlines()

        kitti_labels = []

        for line in lines:
            if line.strip():
                label_data = parse_sit_label_line(line)

                # Map SiT classes to KITTI classes
                class_mapping = {
                    'Pedestrian': 'Pedestrian',
                    'Pedestrain_sitting': 'Pedestrian',  # Map to Pedestrian
                    'Car': 'Car'
                }

                kitti_class = class_mapping.get(label_data['class_name'])
                if kitti_class is None:
                    print(f"Warning: Unknown class {label_data['class_name']}, skipping")
                    continue

                # KITTI format: type truncated occluded alpha bbox_2d[4] dims[3] loc[3] rot_y score
                # For now, set defaults for fields we can't compute without calibration
                truncated = 0.0  # Not truncated
                occluded = 0     # Fully visible
                alpha = 0.0      # Observation angle (would need camera calibration)
                bbox_2d = [0, 0, 0, 0]  # 2D bbox (would need projection)
                score = 1.0      # Ground truth

                # Dimensions: KITTI uses h, w, l format
                h, w, l = label_data['dimensions']

                # Location: x, y, z in camera coordinates (would need transformation)
                x, y, z = label_data['location']

                # Rotation
                rot_y = label_data['rotation_y']

                # Create KITTI label line
                kitti_line = f"{kitti_class} {truncated} {occluded} {alpha} " \
                           f"{bbox_2d[0]} {bbox_2d[1]} {bbox_2d[2]} {bbox_2d[3]} " \
                           f"{h} {w} {l} {x} {y} {z} {rot_y} {score}\n"

                kitti_labels.append(kitti_line)

        # Write KITTI labels
        with open(kitti_label_path, 'w') as f:
            f.writelines(kitti_labels)

        return True

    except Exception as e:
        print(f"Error converting {sit_label_path} to {kitti_label_path}: {e}")
        return False


def parse_sit_calibration(calib_path: str) -> Dict:
    """Parse SiT calibration file.

    Args:
        calib_path (str): Path to SiT calibration file.

    Returns:
        dict: Parsed calibration data.
    """
    calib_data = {}

    with open(calib_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            key, values = line.split(':', 1)
            values = [float(x) for x in values.strip().split()]

            if key.startswith('P') and key.endswith('_intrinsic'):
                # 3x3 intrinsic matrix
                calib_data[key] = np.array(values).reshape(3, 3)
            elif key.startswith('P') and key.endswith('_extrinsic'):
                # 3x4 extrinsic matrix
                calib_data[key] = np.array(values).reshape(3, 4)
            elif key == 'R0_rect':
                # 3x3 rectification matrix
                calib_data[key] = np.array(values).reshape(3, 3)
            elif key.startswith('P') and key.endswith('_distortion'):
                # 5 distortion coefficients
                calib_data[key] = np.array(values)

    return calib_data


def convert_calib_to_kitti(sit_calib_path: str, kitti_calib_path: str) -> bool:
    """Convert SiT calibration to KITTI format.

    Args:
        sit_calib_path (str): Path to SiT calibration file.
        kitti_calib_path (str): Path to output KITTI calibration file.

    Returns:
        bool: True if conversion successful.
    """
    try:
        calib_data = parse_sit_calibration(sit_calib_path)

        # Use P2 as the main camera (KITTI convention)
        if 'P2_intrinsic' in calib_data and 'P2_extrinsic' in calib_data:
            # Create 3x4 projection matrix: P = K * [R|t]
            K = calib_data['P2_intrinsic']
            RT = calib_data['P2_extrinsic']

            # Extend RT to 4x4 if needed
            if RT.shape[1] == 4:
                RT_4x4 = np.eye(4)
                RT_4x4[:3, :] = RT
            else:
                RT_4x4 = RT

            P2 = K @ RT_4x4[:3, :]  # 3x4 projection matrix

            # R0_rect (extend to 3x3 if needed)
            R0_rect = calib_data.get('R0_rect', np.eye(3))

            # Tr_velo_to_cam: Use P2_extrinsic as approximation
            Tr_velo_to_cam = RT_4x4

            # Tr_imu_to_velo: Identity matrix (not available in SiT)
            Tr_imu_to_velo = np.eye(4)

            # Create KITTI calibration content
            calib_lines = []
            calib_lines.append(f"P0: {' '.join([f'{x:.6f}' for x in np.eye(3, 4).flatten()])}")
            calib_lines.append(f"P1: {' '.join([f'{x:.6f}' for x in np.eye(3, 4).flatten()])}")
            calib_lines.append(f"P2: {' '.join([f'{x:.6f}' for x in P2.flatten()])}")
            calib_lines.append(f"P3: {' '.join([f'{x:.6f}' for x in np.eye(3, 4).flatten()])}")
            calib_lines.append(f"R0_rect: {' '.join([f'{x:.6f}' for x in R0_rect.flatten()])}")
            calib_lines.append(f"Tr_velo_to_cam: {' '.join([f'{x:.6f}' for x in Tr_velo_to_cam.flatten()])}")
            calib_lines.append(f"Tr_imu_to_velo: {' '.join([f'{x:.6f}' for x in Tr_imu_to_velo.flatten()])}")

            # Write calibration file
            with open(kitti_calib_path, 'w') as f:
                f.write('\n'.join(calib_lines) + '\n')

            return True
        else:
            print(f"Warning: Missing P2 calibration data in {sit_calib_path}")
            return False

    except Exception as e:
        print(f"Error converting {sit_calib_path} to {kitti_calib_path}: {e}")
        return False


def create_imagesets(output_root: str, sequences: List[str], split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
    """Create ImageSets split files.

    Args:
        output_root (str): Root directory for output.
        sequences (List[str]): List of sequence names.
        split_ratio (tuple): Train/val/test split ratios.
    """
    # Collect all frame indices once from the normalized KITTI-style layout:
    #   output_root/training/velodyne/*.bin
    #
    # NOTE:
    #   Earlier versions incorrectly iterated over `sequences` while always
    #   reading from the same `training/velodyne` directory and prefixing each
    #   frame index with the sequence name. This resulted in the same frames
    #   being duplicated for every sequence (e.g. `seqA_000001`, `seqB_000001`)
    #   even though they all pointed to the same underlying `.bin` files.
    #   Since the conversion step already normalizes all sequences into a
    #   single `training/velodyne` folder, we should only scan that folder
    #   once and use the raw frame indices.
    all_frames: List[str] = []
    seq_dir = osp.join(output_root, 'training', 'velodyne')
    if osp.exists(seq_dir):
        bin_files = [f for f in os.listdir(seq_dir) if f.endswith('.bin')]
        # Keep the filename stem as-is (e.g. '000123') instead of casting to int
        # so that any zero-padding is preserved in the ImageSets files.
        frame_ids = sorted([osp.splitext(f)[0] for f in bin_files])
        all_frames.extend(frame_ids)

    # Split frames
    n_frames = len(all_frames)
    n_train = int(n_frames * split_ratio[0])
    n_val = int(n_frames * split_ratio[1])

    train_frames = all_frames[:n_train]
    val_frames = all_frames[n_train:n_train + n_val]
    test_frames = all_frames[n_train + n_val:]

    # Create ImageSets directory
    imagesets_dir = osp.join(output_root, 'ImageSets')
    os.makedirs(imagesets_dir, exist_ok=True)

    # Write split files
    def write_split_file(filename, frames):
        with open(osp.join(imagesets_dir, filename), 'w') as f:
            for frame in frames:
                f.write(f"{frame}\n")

    write_split_file('train.txt', train_frames)
    write_split_file('val.txt', val_frames)
    write_split_file('test.txt', test_frames)


def get_sit_image_info(data_path: str,
                       training: bool = True,
                       label_info: bool = True,
                       velodyne: bool = True,
                       calib: bool = False,
                       image_ids: List[str] = None,
                       relative_path: bool = True,
                       with_imageshape: bool = True):
    """Get SiT image info similar to KITTI format.

    Args:
        data_path (str): Path to the data directory.
        training (bool): Whether it's training data.
        label_info (bool): Whether to include label info.
        velodyne (bool): Whether to include velodyne info.
        calib (bool): Whether to include calibration info.
        image_ids (List[int]): List of image ids.
        relative_path (bool): Whether to use relative paths.
        with_imageshape (bool): Whether to include image shape.

    Returns:
        List[dict]: List of info dictionaries.
    """
    root_path = Path(data_path)

    if image_ids is None:
        # Get all available frame indices from the normalized KITTI-style layout:
        #   data_path/training/velodyne/*.bin (train)
        #   data_path/testing/velodyne/*.bin  (test, if present)
        if training:
            velodyne_dir = root_path / 'training' / 'velodyne'
        else:
            velodyne_dir = root_path / 'testing' / 'velodyne'

        if velodyne_dir.exists():
            # Preserve the original zero-padded string frame IDs (e.g. "000001")
            # instead of converting them to integers. Converting to int would
            # drop leading zeros and later generate wrong file paths such as
            # "training/velodyne/1.bin" instead of "training/velodyne/000001.bin".
            bin_files = [f for f in os.listdir(velodyne_dir) if f.endswith('.bin')]
            image_ids = sorted([osp.splitext(f)[0] for f in bin_files])
        else:
            image_ids = []

    def map_func(frame_id):
        info = {}

        # Normalize frame id to string and integer forms. The string keeps the
        # exact zero-padding used on disk (e.g. "000001"), while the integer is
        # convenient for code that expects a numeric index in metadata.
        frame_id_str = str(frame_id)
        try:
            frame_id_int = int(frame_id_str)
        except ValueError:
            # Fallback: if the frame id cannot be parsed as int, just keep 0.
            # This should not happen for normal SiT/KITTI-style IDs.
            frame_id_int = 0

        # Point cloud info
        if velodyne:
            pc_info = {'num_pts_feats': 4}
            if training:
                pc_info['lidar_path'] = f'training/velodyne/{frame_id_str}.bin'
            else:
                pc_info['lidar_path'] = f'testing/velodyne/{frame_id_str}.bin'
            info['lidar_points'] = pc_info

        # Image info (placeholder)
        if with_imageshape:
            image_info = {
                # Keep a numeric index for compatibility with downstream tools
                # while preserving zero-padded strings in file paths.
                'image_idx': frame_id_int,
                'image_shape': np.array([1024, 1024], dtype=np.int32)  # Placeholder shape
            }
            if training:
                image_info['image_path'] = f'training/image_2/{frame_id_str}.png'
            else:
                image_info['image_path'] = f'testing/image_2/{frame_id_str}.png'
            info['image'] = image_info

        # Calibration info (placeholder)
        if calib:
            # Use identity matrices as placeholders
            calib_info = {
                'R0_rect': np.eye(4, dtype=np.float32),
                'Tr_velo_to_cam': np.eye(4, dtype=np.float32),
                'P2': np.eye(3, 4, dtype=np.float32),
                'Tr_imu_to_velo': np.eye(4, dtype=np.float32)
            }
            info['calib'] = calib_info

        # Load annotations if available. Always initialize `instances` so that
        # downstream Det3DDataset/SiTDataset logic can safely access it even
        # when a frame has no labels.
        info['instances'] = []
        if label_info and training:
            label_path = root_path / 'training' / 'label_2' / f'{frame_id_str}.txt'
            if label_path.exists():
                annotations = get_kitti_style_annotations(str(label_path))
                if annotations and len(annotations['name']) > 0:
                    instances = convert_annos_to_instances(annotations)
                    info['instances'] = instances

        return info

    return [map_func(idx) for idx in image_ids]


def get_kitti_style_annotations(label_path: str) -> dict:
    """Parse KITTI-style label file and return annotations dict.

    Args:
        label_path (str): Path to label file.

    Returns:
        dict: Annotations in KITTI format.
    """
    annotations = {
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
        'score': [],
        'num_points_in_gt': []  # Will be calculated later
    }

    if not osp.exists(label_path):
        return annotations

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.strip():
            parts = line.strip().split()
            if len(parts) >= 15:  # KITTI format has 15 fields minimum
                annotations['name'].append(parts[0])
                annotations['truncated'].append(float(parts[1]))
                annotations['occluded'].append(int(parts[2]))
                annotations['alpha'].append(float(parts[3]))
                annotations['bbox'].append([float(x) for x in parts[4:8]])
                annotations['dimensions'].append([float(x) for x in parts[8:11]])  # h, w, l
                annotations['location'].append([float(x) for x in parts[11:14]])
                annotations['rotation_y'].append(float(parts[14]))
                annotations['score'].append(1.0)  # Ground truth score
                annotations['num_points_in_gt'].append(0)  # Placeholder, will be calculated

    # Convert to numpy arrays
    for key in annotations:
        if key in ['name']:
            annotations[key] = np.array(annotations[key], dtype='<U10')
        elif key in ['bbox', 'dimensions', 'location']:
            annotations[key] = np.array(annotations[key], dtype=np.float32)
        elif key in ['truncated', 'alpha', 'rotation_y', 'score']:
            annotations[key] = np.array(annotations[key], dtype=np.float32)
        elif key in ['occluded', 'num_points_in_gt']:
            annotations[key] = np.array(annotations[key], dtype=np.int32)

    return annotations


def convert_annos_to_instances(annos: dict) -> list:
    """Convert KITTI-style annotations to instances format.

    Args:
        annos (dict): Annotations in KITTI format.

    Returns:
        list: List of instance dictionaries.
    """
    instances = []

    if len(annos['name']) == 0:
        return instances

    # Class mapping
    class_mapping = {'Pedestrian': 0, 'Car': 1}

    for i in range(len(annos['name'])):
        instance = {
            'bbox': annos['bbox'][i].tolist(),
            'bbox_label': class_mapping.get(annos['name'][i], -1),
            'bbox_3d': [
                annos['location'][i][0],  # x
                annos['location'][i][1],  # y
                annos['location'][i][2],  # z
                annos['dimensions'][i][1],  # w
                annos['dimensions'][i][0],  # h
                annos['dimensions'][i][2],  # l
                annos['rotation_y'][i]     # yaw
            ],
            'bbox_3d_isvalid': True,
            'bbox_label_3d': class_mapping.get(annos['name'][i], -1),
            'depth': 0.0,  # Placeholder depth
            'center_2d': [0.0, 0.0],  # Placeholder center 2D
            'attr_label': -1,  # No attribute
            'num_lidar_pts': int(annos['num_points_in_gt'][i]),
            'num_radar_pts': 0,  # No radar points
            'difficulty': 0,  # Placeholder
            'unaligned_bbox_3d': None  # Keep as None, might be handled specially
        }
        instances.append(instance)

    return instances


def create_sit_infos(data_path: str,
                     save_path: str = None,
                     pkl_prefix: str = 'sit',
                     relative_path: bool = True,
                     split_ratio: Tuple[float, float, float] = (0.7, 0.15,
                                                                0.15)):
    """Create info file of SiT dataset.

    Args:
        data_path (str): Path to the data directory.
        save_path (str, optional): Path to save the info file.
        pkl_prefix (str): Prefix of the info file.
        relative_path (bool): Whether to use relative paths.
        split_ratio (tuple): Train/val/test split ratios. Only the
            train/val portions are used here, but the semantics should
            match :func:`create_imagesets` so that the frame indices in
            ``train.txt`` / ``val.txt`` align with the samples in
            ``*_infos_train.pkl`` / ``*_infos_val.pkl``.
    """
    if save_path is None:
        save_path = data_path

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # SiT dataset metainfo
    metainfo = {
        'categories': {'Pedestrian': 0, 'Car': 1},
        'dataset': 'sit',
        'info_version': '1.1'
    }

    # Create full info list
    print('Creating SiT training info...')
    full_infos = get_sit_image_info(
        data_path,
        training=True,
        label_info=True,
        velodyne=True,
        calib=True,
        relative_path=relative_path)

    # Split into train / val BEFORE saving so that train/val splits match the
    # ImageSets split produced by :func:`create_imagesets`, which uses the
    # same ``split_ratio`` convention.
    n_samples = len(full_infos)
    n_train = int(n_samples * split_ratio[0])
    n_val = int(n_samples * split_ratio[1])

    sit_infos_train = full_infos[:n_train]
    sit_infos_val = full_infos[n_train:n_train + n_val]

    # Calculate num_points_in_gt per instance on the TRAIN split only
    _calculate_num_points_in_gt(data_path, sit_infos_train, relative_path)

    # Save train infos
    train_data_info = {
        'metainfo': metainfo,
        'data_list': sit_infos_train
    }
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'SiT info train file is saved to {filename}')
    mmengine.dump(train_data_info, filename)

    # Save val infos
    val_data_info = {
        'metainfo': metainfo,
        'data_list': sit_infos_val
    }
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'SiT info val file is saved to {filename}')
    mmengine.dump(val_data_info, filename)

    # Create test info (placeholder – test split may not exist)
    sit_infos_test = get_sit_image_info(
        data_path,
        training=False,
        label_info=False,
        velodyne=True,
        calib=True,
        relative_path=relative_path)

    test_data_info = {
        'metainfo': metainfo,
        'data_list': sit_infos_test
    }

    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'SiT info test file is saved to {filename}')
    mmengine.dump(test_data_info, filename)


def _calculate_num_points_in_gt(data_path: str,
                                infos: List[dict],
                                relative_path: bool = True):
    """Calculate number of LiDAR points inside each GT box.

    For the new-style info format used by Det3DDataset, we:
      - read the point cloud from info['lidar_points']['lidar_path']
      - read 3D boxes from info['instances'][*]['bbox_3d']
      - write counts into instances[*]['num_lidar_pts']

    Args:
        data_path (str): Dataset root path (contains training/velodyne).
        infos (List[dict]): List of per-frame info dicts.
        relative_path (bool): Whether lidar_path is relative to data_path.
    """
    from mmdet3d.structures.ops import box_np_ops

    root_path = Path(data_path)

    for info in mmengine.track_iter_progress(infos):
        if 'instances' not in info or len(info['instances']) == 0:
            continue

        # Load point cloud
        pc_path = info['lidar_points']['lidar_path']
        if relative_path:
            pc_path = root_path / pc_path

        if not pc_path.exists():
            # Skip if point cloud is missing
            continue

        points = np.fromfile(str(pc_path), dtype=np.float32).reshape(-1, 4)
        points_xyz = points[:, :3]

        # Collect boxes in [x, y, z, w, h, l, yaw] format from instances.
        # Some instances may not have a valid 7-D bbox_3d; we must keep track
        # of which instances contribute to `boxes` so that we can align the
        # point counts correctly.
        boxes = []
        valid_insts = []
        for inst in info['instances']:
            bbox_3d = np.asarray(inst['bbox_3d'], dtype=np.float32)
            if bbox_3d.shape[0] != 7:
                continue
            boxes.append(bbox_3d)
            valid_insts.append(inst)

        if not boxes:
            continue

        boxes = np.stack(boxes, axis=0).astype(np.float32)

        # Count points per box using standard utility
        point_indices = box_np_ops.points_in_rbbox(points_xyz, boxes)
        counts = point_indices.sum(axis=0).astype(np.int32)

        # Write back into only the instances that had valid 7-D boxes.
        for inst, num in zip(valid_insts, counts):
            inst['num_lidar_pts'] = int(num)


def create_sit_database(data_path: str,
                        save_path: str = None,
                        pkl_prefix: str = 'sit',
                        relative_path: bool = True):
    """Create ground truth database for SiT dataset.

    This is a thin wrapper around the generic `create_groundtruth_database`
    helper so that `sit_converter.py --create-db` produces the same layout
    as `tools/create_data.py sit ...`.

    Args:
        data_path (str): Dataset root path (e.g. data/sit).
        save_path (str, optional): Where to save database files (defaults to data_path).
        pkl_prefix (str): Prefix of the info/db files (default: 'sit').
        relative_path (bool): Whether to use relative paths in dbinfos.
    """
    from tools.dataset_converters.create_gt_database import \
        create_groundtruth_database

    if save_path is None:
        save_path = data_path

    save_path = Path(save_path)
    info_path = save_path / f'{pkl_prefix}_infos_train.pkl'
    if not info_path.exists():
        print(f'Info file {info_path} not found. Please create info files first.')
        return

    db_info_save_path = save_path / f'{pkl_prefix}_dbinfos_train.pkl'
    database_save_path = save_path / f'{pkl_prefix}_gt_database'
    create_groundtruth_database(
        'SiTDataset',
        str(data_path),
        pkl_prefix,
        info_path=None,
        used_classes=['Pedestrian', 'Car'],
        database_save_path=str(database_save_path),
        db_info_save_path=str(db_info_save_path),
        relative_path=relative_path)


def convert_sequence(sit_root: str, output_root: str, sequence: str) -> bool:
    """Convert a single SiT sequence to KITTI format.

    Args:
        sit_root (str): Root directory of SiT dataset.
        output_root (str): Root directory for converted output.
        sequence (str): Sequence name to convert.

    Returns:
        bool: True if conversion successful.
    """
    sit_seq_dir = osp.join(sit_root, sequence)
    output_training_dir = osp.join(output_root, 'training')

    # Create output directories
    os.makedirs(osp.join(output_training_dir, 'velodyne'), exist_ok=True)
    os.makedirs(osp.join(output_training_dir, 'label_2'), exist_ok=True)
    os.makedirs(osp.join(output_training_dir, 'calib'), exist_ok=True)
    os.makedirs(osp.join(output_training_dir, 'image_2'), exist_ok=True)

    print(f"Converting sequence: {sequence}")

    # Get all frame indices from PCD files
    velo_dir = osp.join(sit_seq_dir, 'velo', 'concat', 'data')
    if not osp.exists(velo_dir):
        print(f"Warning: Velocity directory not found: {velo_dir}")
        return False

    pcd_files = sorted([f for f in os.listdir(velo_dir) if f.endswith('.pcd')])
    if not pcd_files:
        print(f"Warning: No PCD files found in {velo_dir}")
        return False

    success_count = 0
    total_count = len(pcd_files)

    for pcd_file in pcd_files:
        frame_idx = pcd_file.split('.')[0]

        # Convert PCD to .bin
        pcd_path = osp.join(velo_dir, pcd_file)
        bin_path = osp.join(output_training_dir, 'velodyne', f'{frame_idx}.bin')

        if convert_pcd_to_bin(pcd_path, bin_path):
            print(f"  Converted PCD: {pcd_file} -> {frame_idx}.bin")
        else:
            print(f"  Failed to convert PCD: {pcd_file}")
            continue

        # Convert labels
        label_path = osp.join(sit_seq_dir, 'label_3d', f'{frame_idx}.txt')
        kitti_label_path = osp.join(output_training_dir, 'label_2', f'{frame_idx}.txt')

        if osp.exists(label_path):
            if convert_label_3d_to_kitti(label_path, kitti_label_path):
                print(f"  Converted labels: {frame_idx}.txt")
            else:
                print(f"  Failed to convert labels: {frame_idx}.txt")
        else:
            # Create empty label file
            with open(kitti_label_path, 'w') as f:
                pass
            print(f"  No labels found for frame {frame_idx}, created empty file")

        # Convert calibration
        calib_path = osp.join(sit_seq_dir, 'calib', f'{frame_idx}.txt')
        kitti_calib_path = osp.join(output_training_dir, 'calib', f'{frame_idx}.txt')

        if osp.exists(calib_path):
            if convert_calib_to_kitti(calib_path, kitti_calib_path):
                print(f"  Converted calibration: {frame_idx}.txt")
            else:
                print(f"  Failed to convert calibration: {frame_idx}.txt")
        else:
            print(f"  No calibration found for frame {frame_idx}")

        # Create placeholder image file (SiT may not have images)
        image_path = osp.join(output_training_dir, 'image_2', f'{frame_idx}.png')
        # For now, just touch the file (would need actual image conversion)
        Path(image_path).touch()

        success_count += 1

    print(f"Converted {success_count}/{total_count} frames for sequence {sequence}")
    return success_count > 0


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(description='Convert SiT dataset to MMDetection3D format')
    parser.add_argument('--sit-root', required=True, help='Root directory of SiT dataset')
    parser.add_argument('--output-root', required=True, help='Output directory for converted data')
    parser.add_argument('--sequence', action='append', help='Specific sequence to convert')
    parser.add_argument('--convert-all', action='store_true', help='Convert all sequences')
    parser.add_argument('--split-ratio', nargs=3, type=float, default=[0.7, 0.15, 0.15],
                       help='Train/val/test split ratios')
    parser.add_argument('--create-info', action='store_true',
                       help='Create info files after conversion')
    parser.add_argument('--create-db', action='store_true',
                       help='Create database files after conversion')

    args = parser.parse_args()

    # Validate inputs
    if not osp.exists(args.sit_root):
        print(f"Error: SiT root directory does not exist: {args.sit_root}")
        return

    # Get sequences to convert
    if args.convert_all:
        sequences = []
        for item in os.listdir(args.sit_root):
            seq_dir = osp.join(args.sit_root, item)
            if osp.isdir(seq_dir) and osp.exists(osp.join(seq_dir, 'velo', 'concat', 'data')):
                sequences.append(item)
    elif args.sequence:
        sequences = args.sequence
    else:
        print("Error: Must specify --sequence or --convert-all")
        return

    print(f"Found sequences: {sequences}")

    # Convert sequences
    converted_sequences = []
    for seq in sequences:
        if convert_sequence(args.sit_root, args.output_root, seq):
            converted_sequences.append(seq)
        else:
            print(f"Failed to convert sequence: {seq}")

    # Create ImageSets
    if converted_sequences:
        create_imagesets(args.output_root, converted_sequences, tuple(args.split_ratio))
        print(f"Created ImageSets with split ratio {args.split_ratio}")

        # Create info files
        if args.create_info:
            print("Creating info files...")
            create_sit_infos(
                args.output_root,
                pkl_prefix='sit',
                split_ratio=tuple(args.split_ratio))

        # Create database files
        if args.create_db:
            print("Creating database files...")
            create_sit_database(args.output_root, pkl_prefix='sit')

    print(f"Conversion completed. Converted {len(converted_sequences)}/{len(sequences)} sequences.")


if __name__ == '__main__':
    main()
