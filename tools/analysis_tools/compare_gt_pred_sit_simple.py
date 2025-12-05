#!/usr/bin/env python3
"""Simple text-based comparison of GT and predictions for SIT dataset.

This script prints a side-by-side comparison without requiring visualization.

Usage:
    python tools/analysis_tools/compare_gt_pred_sit_simple.py \
        --sample-idx 1859 \
        --gt-file data/sit/sit_infos_val.pkl \
        --pred-dir data/sit/kitti_val_pred/pred_instances_3d
"""

import argparse
import os.path as osp

import numpy as np
from mmengine import load
from mmengine.logging import print_log


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple text comparison of GT and predictions')
    parser.add_argument(
        '--sample-idx',
        type=int,
        required=True,
        help='Sample index (e.g., 1859)')
    parser.add_argument(
        '--gt-file',
        type=str,
        default='data/sit/sit_infos_val.pkl',
        help='Path to ground truth info file')
    parser.add_argument(
        '--pred-dir',
        type=str,
        default='data/sit/kitti_val_pred/pred_instances_3d',
        help='Directory containing prediction text files')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.1,
        help='Score threshold for filtering predictions')
    return parser.parse_args()


def load_gt_from_info(gt_file, sample_idx):
    """Load GT annotations from info file."""
    data_infos = load(gt_file)
    
    if isinstance(data_infos, dict):
        data_list = data_infos.get('data_list', [])
    else:
        data_list = data_infos
    
    # Try multiple ways to match the sample
    for info in data_list:
        # Try sample_idx directly
        idx = info.get('sample_idx')
        if idx == sample_idx:
            return info.get('instances', [])
        
        # Try image_idx
        idx = info.get('image', {}).get('image_idx')
        if idx == sample_idx:
            return info.get('instances', [])
        
        # Try lidar_idx (might be string like "0001859")
        lidar_idx = info.get('point_cloud', {}).get('lidar_idx') or \
                    info.get('lidar_points', {}).get('lidar_idx')
        if lidar_idx is not None:
            # Try as int
            try:
                if int(lidar_idx) == sample_idx:
                    return info.get('instances', [])
            except (ValueError, TypeError):
                pass
            # Try as string match
            if str(lidar_idx) == str(sample_idx) or \
               str(lidar_idx) == f'{sample_idx:06d}':
                return info.get('instances', [])
        
        # Try matching by position in list (if sample_idx matches list index)
        # This is a fallback - check if the index in the list matches
        try:
            list_idx = data_list.index(info)
            if list_idx == sample_idx:
                return info.get('instances', [])
        except (ValueError, AttributeError):
            pass
    
    # If not found, try direct index access (sample_idx might be list position)
    try:
        if 0 <= sample_idx < len(data_list):
            info = data_list[sample_idx]
            return info.get('instances', [])
    except (IndexError, TypeError):
        pass
    
    return []


def load_pred_from_txt(pred_file, score_thr):
    """Load predictions from KITTI format text file."""
    if not osp.exists(pred_file):
        return []
    
    predictions = []
    class_name_to_id = {'Pedestrian': 0, 'Car': 1}
    
    with open(pred_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 15:
                continue
            
            class_name = parts[0]
            if class_name not in class_name_to_id:
                continue
            
            score = float(parts[15]) if len(parts) > 15 else 1.0
            if score < score_thr:
                continue
            
            x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
            h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
            yaw = float(parts[14])
            
            predictions.append({
                'class': class_name,
                'class_id': class_name_to_id[class_name],
                'location': [x, y, z],
                'dimensions': [h, w, l],
                'yaw': yaw,
                'score': score
            })
    
    return predictions


def format_box_info(box_data, label, is_gt=True):
    """Format box information for display."""
    if is_gt:
        if isinstance(box_data, dict):
            bbox_3d = box_data.get('bbox_3d', [])
            if len(bbox_3d) >= 7:
                x, y, z = bbox_3d[0], bbox_3d[1], bbox_3d[2]
                w, l, h = bbox_3d[3], bbox_3d[4], bbox_3d[5]
                yaw = bbox_3d[6]
                class_name = ['Pedestrian', 'Car'][label]
                return f"{class_name:12s} | ({x:7.2f}, {y:7.2f}, {z:6.2f}) | [{h:.2f}, {w:.2f}, {l:.2f}] | yaw={yaw:.3f}"
        return "Invalid GT box"
    else:
        class_name = box_data['class']
        x, y, z = box_data['location']
        h, w, l = box_data['dimensions']
        yaw = box_data['yaw']
        score = box_data['score']
        return f"{class_name:12s} | ({x:7.2f}, {y:7.2f}, {z:6.2f}) | [{h:.2f}, {w:.2f}, {l:.2f}] | yaw={yaw:.3f} | score={score:.3f}"


def main():
    args = parse_args()
    
    # Load GT with debug info
    data_infos = load(args.gt_file)
    if isinstance(data_infos, dict):
        data_list = data_infos.get('data_list', [])
    else:
        data_list = data_infos
    
    print_log(f'Total samples in GT file: {len(data_list)}')
    
    # Try to find the sample and show what we're looking for
    found_sample = None
    for i, info in enumerate(data_list):
        sample_idx_val = info.get('sample_idx')
        image_idx_val = info.get('image', {}).get('image_idx')
        lidar_idx_val = info.get('point_cloud', {}).get('lidar_idx') or \
                       info.get('lidar_points', {}).get('lidar_idx')
        
        if sample_idx_val == args.sample_idx or \
           image_idx_val == args.sample_idx or \
           (lidar_idx_val is not None and (
               str(lidar_idx_val) == str(args.sample_idx) or
               str(lidar_idx_val) == f'{args.sample_idx:06d}' or
               (isinstance(lidar_idx_val, (int, str)) and int(lidar_idx_val) == args.sample_idx)
           )) or \
           i == args.sample_idx:
            found_sample = info
            print_log(f'Found sample at list index {i}')
            print_log(f'  sample_idx: {sample_idx_val}')
            print_log(f'  image_idx: {image_idx_val}')
            print_log(f'  lidar_idx: {lidar_idx_val}')
            break
    
    if found_sample is None:
        print_log(f'Warning: Sample {args.sample_idx} not found in GT file')
        print_log('Showing first few samples for reference:')
        for i in range(min(3, len(data_list))):
            info = data_list[i]
            print_log(f'  Sample {i}: sample_idx={info.get("sample_idx")}, '
                     f'image_idx={info.get("image", {}).get("image_idx")}, '
                     f'lidar_idx={info.get("point_cloud", {}).get("lidar_idx")}')
        gt_instances = []
    else:
        # Try multiple possible locations for GT annotations
        gt_instances = found_sample.get('instances', [])
        
        # Also check 'annos' field (KITTI-style format)
        if not gt_instances and 'annos' in found_sample:
            annos = found_sample['annos']
            if isinstance(annos, dict) and 'name' in annos:
                num_annos = len(annos['name']) if hasattr(annos['name'], '__len__') else 0
                if num_annos > 0:
                    print_log(f'Found {num_annos} annotations in "annos" field (KITTI format)')
                    # Convert annos to instances format for display
                    gt_instances = []
                    for i in range(num_annos):
                        instance = {
                            'bbox_3d': [
                                float(annos['location'][i][0]),
                                float(annos['location'][i][1]),
                                float(annos['location'][i][2]),
                                float(annos['dimensions'][i][2]),  # l
                                float(annos['dimensions'][i][1]),  # w
                                float(annos['dimensions'][i][0]),  # h
                                float(annos['rotation_y'][i])
                            ],
                            'bbox_label_3d': 0 if annos['name'][i] == 'Pedestrian' else 1
                        }
                        gt_instances.append(instance)
        
        print_log(f'Found {len(gt_instances)} GT instances')
        
        # Debug: show what fields are available
        if len(gt_instances) == 0:
            keys_str = ', '.join(list(found_sample.keys()))
            print_log(f'Available keys in sample: {keys_str}')
            if 'instances' in found_sample:
                instances_len = len(found_sample["instances"]) if found_sample["instances"] else 0
                print_log(f'  instances type: {type(found_sample["instances"])}, length: {instances_len}')
            if 'annos' in found_sample:
                annos = found_sample['annos']
                print_log(f'  annos type: {type(annos)}')
                if isinstance(annos, dict):
                    annos_keys_str = ', '.join(list(annos.keys()))
                    print_log(f'  annos keys: {annos_keys_str}')
                    if 'name' in annos:
                        name_len = len(annos["name"]) if hasattr(annos["name"], "__len__") else "N/A"
                        print_log(f'  annos["name"] length: {name_len}')
    
    # Load predictions
    pred_file = osp.join(args.pred_dir, f'{args.sample_idx:06d}.txt')
    predictions = load_pred_from_txt(pred_file, args.score_thr)
    
    # Print comparison
    print_log('\n' + '='*100)
    print_log(f'Comparison for Sample {args.sample_idx:06d}')
    print_log('='*100)
    
    print_log('\nGROUND TRUTH:')
    print_log('-' * 100)
    print_log(f"{'Class':12s} | {'Location (x, y, z)':25s} | {'Dimensions [h, w, l]':20s} | {'Yaw':10s}")
    print_log('-' * 100)
    
    if gt_instances:
        for i, inst in enumerate(gt_instances):
            label = inst.get('bbox_label', inst.get('bbox_label_3d', 0))
            print_log(f"GT {i+1:2d}: {format_box_info(inst, label, is_gt=True)}")
    else:
        print_log('  No ground truth annotations')
    
    print_log('\nPREDICTIONS (score >= {:.2f}):'.format(args.score_thr))
    print_log('-' * 100)
    print_log(f"{'Class':12s} | {'Location (x, y, z)':25s} | {'Dimensions [h, w, l]':20s} | {'Yaw':10s} | {'Score':8s}")
    print_log('-' * 100)
    
    if predictions:
        for i, pred in enumerate(predictions):
            print_log(f"Pred {i+1:2d}: {format_box_info(pred, pred['class_id'], is_gt=False)}")
    else:
        print_log('  No predictions')
    
    print_log('\nSUMMARY:')
    print_log(f'  GT boxes: {len(gt_instances)}')
    print_log(f'  Pred boxes: {len(predictions)}')
    
    if len(gt_instances) == 0 and len(predictions) > 0:
        print_log('\nNOTE: This sample has no GT annotations but has predictions.')
        print_log('      This could mean:')
        print_log('      1. The validation set may not have GT for all samples')
        print_log('      2. This sample genuinely has no annotations')
        print_log('      3. Try a sample from the training set (sit_infos_train.pkl)')
        print_log('         which should have GT annotations')
    
    print_log('='*100 + '\n')


if __name__ == '__main__':
    main()

