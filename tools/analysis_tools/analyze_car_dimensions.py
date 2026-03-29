#!/usr/bin/env python3
"""
Analyze Car dimensions in SiT dataset to determine optimal anchor sizes.

This script analyzes the actual Car dimensions from the database to help
configure anchor sizes in PointPillars config.

Usage:
    python tools/analysis_tools/analyze_car_dimensions.py
"""

import pickle
import numpy as np
from pathlib import Path
import sys

def analyze_dimensions(data_root='data/sit'):
    """Analyze Car and Pedestrian dimensions from database."""
    
    db_path = Path(data_root) / 'sit_dbinfos_train.pkl'
    
    if not db_path.exists():
        print(f"❌ Error: Database file not found: {db_path}")
        print(f"   Please ensure dataset is properly converted.")
        sys.exit(1)
    
    print(f"📂 Loading database from: {db_path}")
    with open(db_path, 'rb') as f:
        db_infos = pickle.load(f)
    
    print(f"✅ Loaded database\n")
    
    for class_name in ['Pedestrian', 'Car']:
        if class_name not in db_infos:
            print(f"⚠️  Class '{class_name}' not found in database")
            continue
        
        boxes = db_infos[class_name]
        print(f"\n{'='*60}")
        print(f"  {class_name.upper()} ANALYSIS")
        print(f"{'='*60}")
        print(f"Total samples: {len(boxes)}")
        
        # Extract dimensions (l, w, h from box3d_lidar)
        # box3d_lidar format: [x, y, z, l, w, h, yaw]
        dimensions = np.array([box['box3d_lidar'][3:6] for box in boxes])
        
        # Extract individual dimensions
        lengths = dimensions[:, 0]  # l (x-axis in LiDAR)
        widths = dimensions[:, 1]   # w (y-axis in LiDAR)
        heights = dimensions[:, 2]  # h (z-axis in LiDAR)
        
        print(f"\n📊 Dimension Statistics (in meters):")
        print(f"\n{'Dimension':<15} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print(f"{'-'*65}")
        
        for name, values in [('Length (l)', lengths), ('Width (w)', widths), ('Height (h)', heights)]:
            print(f"{name:<15} "
                  f"{np.mean(values):<10.3f} "
                  f"{np.median(values):<10.3f} "
                  f"{np.std(values):<10.3f} "
                  f"{np.min(values):<10.3f} "
                  f"{np.max(values):<10.3f}")
        
        # Calculate percentiles for anchor design
        print(f"\n📈 Percentiles (for anchor design):")
        print(f"\n{'Dimension':<15} {'25%':<10} {'50%':<10} {'75%':<10} {'90%':<10}")
        print(f"{'-'*55}")
        
        for name, values in [('Length (l)', lengths), ('Width (w)', widths), ('Height (h)', heights)]:
            p25, p50, p75, p90 = np.percentile(values, [25, 50, 75, 90])
            print(f"{name:<15} {p25:<10.3f} {p50:<10.3f} {p75:<10.3f} {p90:<10.3f}")
        
        # Recommended anchor size (using median or mean)
        mean_l, mean_w, mean_h = np.mean(dimensions, axis=0)
        median_l, median_w, median_h = np.median(dimensions, axis=0)
        
        print(f"\n🎯 RECOMMENDED ANCHOR SIZES:")
        print(f"   Using MEAN:   [{mean_l:.2f}, {mean_w:.2f}, {mean_h:.2f}]")
        print(f"   Using MEDIAN: [{median_l:.2f}, {median_w:.2f}, {median_h:.2f}]")
        
        # Show current config (from typical PointPillars setup)
        if class_name == 'Car':
            print(f"\n⚠️  CURRENT CONFIG (LIKELY WRONG):")
            print(f"   sizes = [[1.76, 0.6, 1.73]]  ← 0.6m long car?!")
            print(f"\n✅ SUGGESTED FIX:")
            print(f"   sizes = [[{median_w:.2f}, {median_l:.2f}, {median_h:.2f}]]")
            print(f"   Note: PointPillars uses (width, length, height) order")
        
        elif class_name == 'Pedestrian':
            print(f"\n📝 Current config (probably OK):")
            print(f"   sizes = [[0.8, 0.6, 1.73]]")
            if abs(median_w - 0.8) > 0.2 or abs(median_l - 0.6) > 0.2:
                print(f"\n⚠️  Consider updating to:")
                print(f"   sizes = [[{median_w:.2f}, {median_l:.2f}, {median_h:.2f}]]")
    
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"\nTo update your config:")
    print(f"1. Edit: configs/pointpillars/pointpillars_sit_3class.py")
    print(f"2. Find the 'sizes' parameter in anchor_generator (around line 149)")
    print(f"3. Update with the recommended values above")
    print(f"4. Restart training\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyze Car dimensions in SiT dataset')
    parser.add_argument('--data-root', default='data/sit', help='Path to SiT dataset')
    args = parser.parse_args()
    
    analyze_dimensions(args.data_root)
