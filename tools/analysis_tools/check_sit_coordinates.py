#!/usr/bin/env python
"""Diagnostic script to check SiT dataset coordinate system and box distributions.

This script:
1. Loads SiT annotation file
2. Extracts all bounding box centers and dimensions
3. Visualizes spatial distribution
4. Prints statistics to identify coordinate system issues
"""

import argparse
import sys
from pathlib import Path

import mmengine
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze SiT dataset coordinate system')
    parser.add_argument(
        '--ann-file',
        type=str,
        default='data/sit/sit_infos_val.pkl',
        help='Path to annotation file')
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of samples to analyze')
    parser.add_argument(
        '--output',
        type=str,
        default='sit_box_distribution.png',
        help='Output visualization path')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load annotations
    print(f"Loading annotations from: {args.ann_file}")
    if not Path(args.ann_file).exists():
        print(f"ERROR: File not found: {args.ann_file}")
        sys.exit(1)
    
    data = mmengine.load(args.ann_file)
    
    if 'data_list' not in data:
        print("ERROR: 'data_list' not found in annotation file")
        sys.exit(1)
    
    # Extract all box centers and dimensions
    all_centers = []
    all_dimensions = []
    all_rotations = []
    all_labels = []
    
    num_samples = min(args.num_samples, len(data['data_list']))
    print(f"Analyzing {num_samples} samples...")
    
    for i, sample in enumerate(data['data_list'][:num_samples]):
        if 'instances' in sample and len(sample['instances']) > 0:
            for inst in sample['instances']:
                if 'bbox_3d' in inst:
                    bbox_3d = inst['bbox_3d']
                    # bbox_3d format: [x, y, z, dim1, dim2, dim3, rot_y]
                    center = bbox_3d[:3]
                    dimensions = bbox_3d[3:6]
                    rotation = bbox_3d[6] if len(bbox_3d) > 6 else 0.0
                    label = inst.get('bbox_label', -1)
                    
                    all_centers.append(center)
                    all_dimensions.append(dimensions)
                    all_rotations.append(rotation)
                    all_labels.append(label)
    
    if len(all_centers) == 0:
        print("ERROR: No bounding boxes found in dataset!")
        sys.exit(1)
    
    centers = np.array(all_centers)
    dimensions = np.array(all_dimensions)
    rotations = np.array(all_rotations)
    labels = np.array(all_labels)
    
    # Print statistics
    print("\n" + "="*60)
    print("=== Ground Truth Box Statistics ===")
    print("="*60)
    print(f"Total boxes analyzed: {len(centers)}")
    print(f"\nBox Center Ranges:")
    print(f"  X: [{centers[:, 0].min():.2f}, {centers[:, 0].max():.2f}] meters")
    print(f"  Y: [{centers[:, 1].min():.2f}, {centers[:, 1].max():.2f}] meters")
    print(f"  Z: [{centers[:, 2].min():.2f}, {centers[:, 2].max():.2f}] meters")
    
    print(f"\nBox Center Mean:")
    print(f"  X: {centers[:, 0].mean():.2f} ± {centers[:, 0].std():.2f}")
    print(f"  Y: {centers[:, 1].mean():.2f} ± {centers[:, 1].std():.2f}")
    print(f"  Z: {centers[:, 2].mean():.2f} ± {centers[:, 2].std():.2f}")
    
    print(f"\nBox Dimensions Ranges:")
    print(f"  Dim1: [{dimensions[:, 0].min():.2f}, {dimensions[:, 0].max():.2f}]")
    print(f"  Dim2: [{dimensions[:, 1].min():.2f}, {dimensions[:, 1].max():.2f}]")
    print(f"  Dim3: [{dimensions[:, 2].min():.2f}, {dimensions[:, 2].max():.2f}]")
    
    print(f"\nBox Dimensions Mean:")
    print(f"  Dim1: {dimensions[:, 0].mean():.2f} ± {dimensions[:, 0].std():.2f}")
    print(f"  Dim2: {dimensions[:, 1].mean():.2f} ± {dimensions[:, 1].std():.2f}")
    print(f"  Dim3: {dimensions[:, 2].mean():.2f} ± {dimensions[:, 2].std():.2f}")
    
    print(f"\nRotation Range:")
    print(f"  Yaw: [{rotations.min():.2f}, {rotations.max():.2f}] radians")
    print(f"  Yaw: [{np.rad2deg(rotations.min()):.1f}°, {np.rad2deg(rotations.max()):.1f}°]")
    
    # Analyze dimensions to identify bug
    print("\n" + "="*60)
    print("=== DIMENSION ORDER ANALYSIS ===")
    print("="*60)
    print("Checking if dimensions follow expected patterns...")
    print("\nExpected for LiDAR boxes (l, w, h):")
    print("  Pedestrian: l≈0.8, w≈0.6, h≈1.73")
    print("  Car: l≈1.76, w≈0.6, h≈1.73")
    print("\nIf stored as (h, w, l), would see:")
    print("  Pedestrian: dim1≈1.73, dim2≈0.6, dim3≈0.8")
    print("  Car: dim1≈1.73, dim2≈0.6, dim3≈1.76")
    
    # Check if dim1 is consistently ~1.73 (suggests h, w, l order)
    dim1_near_pedestrian_height = np.abs(dimensions[:, 0] - 1.73) < 0.2
    dim3_near_pedestrian_length = np.abs(dimensions[:, 2] - 0.8) < 0.2
    
    print(f"\nActual dimension statistics:")
    print(f"  Boxes with Dim1≈1.73: {dim1_near_pedestrian_height.sum()} ({dim1_near_pedestrian_height.mean()*100:.1f}%)")
    print(f"  Boxes with Dim3≈0.8: {dim3_near_pedestrian_length.sum()} ({dim3_near_pedestrian_length.mean()*100:.1f}%)")
    
    if dim1_near_pedestrian_height.mean() > 0.5:
        print("\n⚠️  WARNING: Dimension order likely (h, w, l) - BUG DETECTED!")
        print("    Should be (l, w, h) for LiDARInstance3DBoxes")
        print("    Fix required in converter or metric!")
    else:
        print("\n✓  Dimension order appears correct (l, w, h)")
    
    # Class-wise statistics
    if 'metainfo' in data and 'categories' in data['metainfo']:
        categories = data['metainfo']['categories']
        label_names = {v: k for k, v in categories.items()}
        
        print("\n" + "="*60)
        print("=== Per-Class Statistics ===")
        print("="*60)
        for label_id in np.unique(labels):
            if label_id >= 0:
                mask = labels == label_id
                class_name = label_names.get(label_id, f"Class_{label_id}")
                print(f"\n{class_name} (label={label_id}):")
                print(f"  Count: {mask.sum()}")
                print(f"  Dimensions: [{dimensions[mask, 0].mean():.2f}, "
                      f"{dimensions[mask, 1].mean():.2f}, "
                      f"{dimensions[mask, 2].mean():.2f}]")
    
    # Visualization
    print("\n" + "="*60)
    print(f"Generating visualization: {args.output}")
    print("="*60)
    
    fig = plt.figure(figsize=(18, 5))
    
    # XY plane (bird's eye view)
    ax1 = fig.add_subplot(131)
    scatter = ax1.scatter(centers[:, 0], centers[:, 1], c=labels, s=2, alpha=0.6, cmap='tab10')
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title('Bird\'s Eye View (XY plane)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.axhline(0, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
    ax1.axvline(0, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
    
    # XZ plane (side view)
    ax2 = fig.add_subplot(132)
    ax2.scatter(centers[:, 0], centers[:, 2], c=labels, s=2, alpha=0.6, cmap='tab10')
    ax2.set_xlabel('X (m)', fontsize=12)
    ax2.set_ylabel('Z (m)', fontsize=12)
    ax2.set_title('Side View (XZ plane)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
    ax2.axvline(0, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
    
    # YZ plane (front view)
    ax3 = fig.add_subplot(133)
    ax3.scatter(centers[:, 1], centers[:, 2], c=labels, s=2, alpha=0.6, cmap='tab10')
    ax3.set_xlabel('Y (m)', fontsize=12)
    ax3.set_ylabel('Z (m)', fontsize=12)
    ax3.set_title('Front View (YZ plane)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
    ax3.axvline(0, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"✓  Saved to {args.output}")
    
    # Also create dimension distribution plot
    dim_output = args.output.replace('.png', '_dimensions.png')
    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(dimensions[:, 0], bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Dimension 1 (m)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Dimension 1 Distribution', fontsize=14, fontweight='bold')
    axes[0].axvline(1.73, color='red', linestyle='--', label='Expected h=1.73')
    axes[0].axvline(0.8, color='blue', linestyle='--', label='Expected l=0.8')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(dimensions[:, 1], bins=50, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Dimension 2 (m)', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Dimension 2 Distribution', fontsize=14, fontweight='bold')
    axes[1].axvline(0.6, color='red', linestyle='--', label='Expected w=0.6')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].hist(dimensions[:, 2], bins=50, alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Dimension 3 (m)', fontsize=12)
    axes[2].set_ylabel('Count', fontsize=12)
    axes[2].set_title('Dimension 3 Distribution', fontsize=14, fontweight='bold')
    axes[2].axvline(1.73, color='red', linestyle='--', label='Expected h=1.73')
    axes[2].axvline(0.8, color='blue', linestyle='--', label='Expected l=0.8')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(dim_output, dpi=150, bbox_inches='tight')
    print(f"✓  Dimension distribution saved to {dim_output}")
    
    print("\n" + "="*60)
    print("=== DIAGNOSIS COMPLETE ===")
    print("="*60)
    print("\nNext steps:")
    print("1. Check the visualizations for spatial outliers")
    print("2. If dimension order is wrong (h,w,l instead of l,w,h):")
    print("   → Fix in tools/dataset_converters/sit_converter.py:113")
    print("   → Regenerate dataset")
    print("3. If coordinates seem offset or rotated:")
    print("   → Check original SiT dataset format")
    print("   → Verify coordinate transformation in converter")


if __name__ == '__main__':
    main()

