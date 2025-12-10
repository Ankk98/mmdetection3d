#!/usr/bin/env python
"""Quick diagnostic to check dimension order bug.

This script quickly checks if the dimension order bug exists in the dataset.
Run this first before the full coordinate check.
"""

import sys
from pathlib import Path
import mmengine


def main():
    print("="*70)
    print("QUICK DIMENSION ORDER CHECK")
    print("="*70)
    
    # Check validation annotations
    ann_file = 'data/sit/sit_infos_val.pkl'
    print(f"\nLoading: {ann_file}")
    
    if not Path(ann_file).exists():
        print(f"ERROR: File not found: {ann_file}")
        print("Have you generated the SiT dataset?")
        return
    
    data = mmengine.load(ann_file)
    
    # Find first sample with instances
    sample_with_instances = None
    for sample in data['data_list'][:10]:
        if 'instances' in sample and len(sample['instances']) > 0:
            sample_with_instances = sample
            break
    
    if sample_with_instances is None:
        print("ERROR: No samples with instances found!")
        return
    
    # Check first instance
    first_instance = sample_with_instances['instances'][0]
    bbox_3d = first_instance['bbox_3d']
    label = first_instance.get('bbox_label', -1)
    
    # Get class name
    if 'metainfo' in data and 'categories' in data['metainfo']:
        categories = data['metainfo']['categories']
        label_names = {v: k for k, v in categories.items()}
        class_name = label_names.get(label, f"Unknown_{label}")
    else:
        class_name = f"Label_{label}"
    
    print(f"\nSample data:")
    print(f"  Class: {class_name}")
    print(f"  BBox 3D: {bbox_3d}")
    print(f"  Format: [x, y, z, dim1, dim2, dim3, rotation_y]")
    
    center = bbox_3d[:3]
    dimensions = bbox_3d[3:6]
    rotation = bbox_3d[6] if len(bbox_3d) > 6 else 0.0
    
    print(f"\nParsed:")
    print(f"  Center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
    print(f"  Dimensions: [{dimensions[0]:.2f}, {dimensions[1]:.2f}, {dimensions[2]:.2f}]")
    print(f"  Rotation: {rotation:.3f} rad ({rotation*57.3:.1f}°)")
    
    print("\n" + "="*70)
    print("DIMENSION ORDER ANALYSIS")
    print("="*70)
    
    print("\nExpected dimensions for common classes:")
    print("  Pedestrian (l, w, h): [0.80, 0.60, 1.73]")
    print("  Car (l, w, h):        [1.76, 0.60, 1.73]")
    
    print("\nIf stored incorrectly as (h, w, l):")
    print("  Pedestrian would be:  [1.73, 0.60, 0.80]")
    print("  Car would be:         [1.73, 0.60, 1.76]")
    
    print(f"\nActual dimensions: [{dimensions[0]:.2f}, {dimensions[1]:.2f}, {dimensions[2]:.2f}]")
    
    # Heuristic check
    is_pedestrian_like = (0.6 < dimensions[0] < 1.0 and 0.4 < dimensions[1] < 0.8 and 1.5 < dimensions[2] < 2.0)
    is_car_like = (1.5 < dimensions[0] < 2.5 and 0.4 < dimensions[1] < 0.8 and 1.5 < dimensions[2] < 2.0)
    
    # Check for (h, w, l) pattern
    is_height_first = (1.5 < dimensions[0] < 2.0 and 0.4 < dimensions[1] < 0.8 and 0.6 < dimensions[2] < 2.0)
    
    print("\n" + "="*70)
    if is_height_first:
        print("🔴 BUG DETECTED: Dimensions appear to be in (h, w, l) order!")
        print("="*70)
        print("\nEvidence:")
        print(f"  - dim1 ≈ {dimensions[0]:.2f} matches typical height (~1.73)")
        print(f"  - dim2 ≈ {dimensions[1]:.2f} matches typical width (~0.60)")
        print(f"  - dim3 ≈ {dimensions[2]:.2f} matches typical length (0.8-1.76)")
        
        print("\n⚠️  CRITICAL: LiDARInstance3DBoxes expects (l, w, h) order!")
        print("\nFix required in:")
        print("  Option A: tools/dataset_converters/sit_converter.py line 113")
        print("            Change: 'dimensions': [float(length), float(width), float(height)]")
        print("  Option B: mmdet3d/evaluation/metrics/sit_metric.py")
        print("            Add dimension reordering before creating boxes")
        
        print("\nAfter fix:")
        print("  1. Regenerate dataset: python tools/dataset_converters/sit_convert_all_scenes.py ...")
        print("  2. Verify fix: python tools/analysis_tools/quick_dimension_check.py")
        print("  3. Test model: python tools/test.py ... (should see IoU > 0)")
        
    elif is_pedestrian_like or is_car_like:
        print("✅ Dimensions appear CORRECT: (l, w, h) order detected")
        print("="*70)
        print("\nDimension order is not the issue.")
        print("Check other potential problems:")
        print("  1. Run full coordinate check: python tools/analysis_tools/check_sit_coordinates.py")
        print("  2. Verify anchor configuration in config file")
        print("  3. Check if coordinate transformation is correct")
    else:
        print("⚠️  UNCERTAIN: Cannot determine dimension order from this sample")
        print("="*70)
        print(f"\nDimensions [{dimensions[0]:.2f}, {dimensions[1]:.2f}, {dimensions[2]:.2f}]")
        print("don't match typical Pedestrian or Car sizes.")
        print("\nCheck original SiT dataset:")
        print("  1. cd ../sit-converter/dataset/")
        print("  2. cat scene_*/labels_3d.txt | head -5")
        print("  3. Verify dimension format in original data")
    
    print("\n" + "="*70)
    
    # Additional check: look at multiple samples
    print("\nChecking first 10 samples for pattern...")
    print("="*70)
    
    dim1_values = []
    dim3_values = []
    
    for i, sample in enumerate(data['data_list'][:20]):
        if 'instances' in sample and len(sample['instances']) > 0:
            for inst in sample['instances']:
                dims = inst['bbox_3d'][3:6]
                dim1_values.append(dims[0])
                dim3_values.append(dims[2])
    
    if len(dim1_values) > 5:
        import numpy as np
        dim1_arr = np.array(dim1_values)
        dim3_arr = np.array(dim3_values)
        
        # Check if dim1 is consistently near 1.73 (height)
        dim1_near_height = np.abs(dim1_arr - 1.73) < 0.3
        dim3_near_length = np.logical_or(
            np.abs(dim3_arr - 0.8) < 0.2,  # Pedestrian length
            np.abs(dim3_arr - 1.76) < 0.3   # Car length
        )
        
        print(f"\nStatistics from {len(dim1_values)} boxes:")
        print(f"  Dim1 range: [{dim1_arr.min():.2f}, {dim1_arr.max():.2f}]")
        print(f"  Dim1 mean: {dim1_arr.mean():.2f} ± {dim1_arr.std():.2f}")
        print(f"  Dim3 range: [{dim3_arr.min():.2f}, {dim3_arr.max():.2f}]")
        print(f"  Dim3 mean: {dim3_arr.mean():.2f} ± {dim3_arr.std():.2f}")
        
        print(f"\nPattern analysis:")
        print(f"  Boxes with dim1≈1.73 (height): {dim1_near_height.sum()} ({dim1_near_height.mean()*100:.0f}%)")
        print(f"  Boxes with dim3≈0.8 or 1.76 (length): {dim3_near_length.sum()} ({dim3_near_length.mean()*100:.0f}%)")
        
        if dim1_near_height.mean() > 0.5:
            print("\n🔴 CONFIRMED: Pattern matches (h, w, l) order across multiple samples!")
        elif dim3_near_length.mean() > 0.5:
            print("\n✅ CONFIRMED: Pattern matches (l, w, h) order across multiple samples!")
        else:
            print("\n⚠️  Pattern unclear - manual verification needed")


if __name__ == '__main__':
    main()

