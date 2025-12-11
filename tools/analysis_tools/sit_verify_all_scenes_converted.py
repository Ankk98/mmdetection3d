#!/usr/bin/env python3
"""Verify that data from all scene type subfolders was converted.

This script checks:
1. How many sequences exist in raw data (by scene type)
2. How many frames were converted
3. Whether all scene types are represented

Usage:
    python tools/analysis_tools/sit_verify_all_scenes_converted.py \
        --raw-root /path/to/raw/data \
        --converted-root /path/to/converted/data
"""

import argparse
import os
from pathlib import Path
from collections import defaultdict

def count_raw_sequences(raw_root):
    """Count sequences in raw data by scene type."""
    scene_counts = defaultdict(list)
    
    raw_path = Path(raw_root)
    if not raw_path.exists():
        print(f"Error: Raw data root does not exist: {raw_root}")
        return {}
    
    # Expected scene types
    scene_types = [
        'Cafe_street', 'Cafeteria', 'Corridor', 'Courtyard', 
        'Crossroad', 'Hallway', 'Lobby', 'Outdoor_Alley', 
        'Subway_Entrance', 'Three_way_Intersection'
    ]
    
    for scene_type in scene_types:
        scene_dir = raw_path / scene_type
        if not scene_dir.exists():
            continue
        
        # Find all sequences (subdirectories with velo/concat/data)
        sequences = []
        for item in scene_dir.iterdir():
            if item.is_dir():
                velo_dir = item / 'velo' / 'concat' / 'data'
                if velo_dir.exists():
                    pcd_files = list(velo_dir.glob('*.pcd'))
                    if pcd_files:
                        sequences.append({
                            'name': item.name,
                            'frame_count': len(pcd_files)
                        })
        
        if sequences:
            scene_counts[scene_type] = sequences
    
    return scene_counts

def count_converted_frames(converted_root):
    """Count converted frames."""
    converted_path = Path(converted_root)
    velodyne_dir = converted_path / 'training' / 'velodyne'
    
    if not velodyne_dir.exists():
        return 0
    
    bin_files = list(velodyne_dir.glob('*.bin'))
    return len(bin_files)

def estimate_expected_frames(scene_counts):
    """Estimate total frames we'd expect from all sequences."""
    total = 0
    for scene_type, sequences in scene_counts.items():
        for seq in sequences:
            total += seq['frame_count']
    return total

def main():
    parser = argparse.ArgumentParser(
        description='Verify that data from all scene type subfolders was converted',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths (raw: /data/sit/raw, converted: data/sit)
  python tools/analysis_tools/sit_verify_all_scenes_converted.py

  # Specify custom paths
  python tools/analysis_tools/sit_verify_all_scenes_converted.py \\
      --raw-root /run/media/user/drive/datasets/sit/raw \\
      --converted-root /workspace/mmdetection3d/data/sit

  # Use paths relative to current directory
  python tools/analysis_tools/sit_verify_all_scenes_converted.py \\
      --raw-root ../raw_data/sit \\
      --converted-root ./data/sit
        """
    )
    parser.add_argument(
        '--raw-root',
        type=str,
        default='/data/sit/raw',
        help='Root directory containing raw SiT scene type subfolders '
             '(default: /data/sit/raw)'
    )
    parser.add_argument(
        '--converted-root',
        type=str,
        default='data/sit',
        help='Root directory containing converted SiT data '
             '(default: data/sit)'
    )
    
    args = parser.parse_args()
    raw_root = args.raw_root
    converted_root = args.converted_root
    
    print("=" * 80)
    print("Verifying SiT Dataset Conversion Coverage")
    print("=" * 80)
    print()
    print(f"Raw data root:      {raw_root}")
    print(f"Converted data root: {converted_root}")
    print()
    
    # Count raw sequences
    print("1. Analyzing raw data structure...")
    scene_counts = count_raw_sequences(raw_root)
    
    if not scene_counts:
        print("   ERROR: No scene types found in raw data!")
        return
    
    print(f"   Found {len(scene_counts)} scene types with sequences:")
    print()
    
    total_sequences = 0
    total_expected_frames = 0
    
    for scene_type, sequences in sorted(scene_counts.items()):
        seq_count = len(sequences)
        frame_count = sum(s['frame_count'] for s in sequences)
        total_sequences += seq_count
        total_expected_frames += frame_count
        
        print(f"   {scene_type:25s}: {seq_count:3d} sequences, {frame_count:5d} frames")
        # Show first few sequence names
        if seq_count <= 5:
            seq_names = [s['name'] for s in sequences]
            print(f"     Sequences: {', '.join(seq_names)}")
        else:
            seq_names = [s['name'] for s in sequences[:3]]
            print(f"     Sequences: {', '.join(seq_names)} ... ({seq_count - 3} more)")
    
    print()
    print(f"   Total: {total_sequences} sequences, {total_expected_frames} expected frames")
    print()
    
    # Count converted frames
    print("2. Checking converted data...")
    converted_frames = count_converted_frames(converted_root)
    
    if converted_frames == 0:
        print("   ERROR: No converted frames found!")
        return
    
    print(f"   Converted frames: {converted_frames}")
    print()
    
    # Compare
    print("3. Comparison:")
    print(f"   Expected frames (from raw data): {total_expected_frames}")
    print(f"   Converted frames:                 {converted_frames}")
    
    if converted_frames == total_expected_frames:
        print("   ✅ PERFECT MATCH! All frames were converted.")
    elif converted_frames > total_expected_frames:
        print(f"   ⚠️  WARNING: More converted frames ({converted_frames}) than expected ({total_expected_frames})")
        print("      This might indicate duplicate conversions or frame ID conflicts.")
    else:
        missing = total_expected_frames - converted_frames
        percentage = (converted_frames / total_expected_frames) * 100
        print(f"   ⚠️  WARNING: Missing {missing} frames ({100 - percentage:.1f}%)")
        print(f"      Only {percentage:.1f}% of expected frames were converted.")
    
    print()
    print("4. Scene type coverage:")
    
    # Check if all scene types are represented
    # Since frame IDs are merged, we can't directly trace which scene they came from
    # But we can verify by checking if conversion happened for each scene type
    print("   (Note: Frame IDs are merged during conversion, so we can't")
    print("    directly trace which scene each frame came from.)")
    print()
    
    # Check if ImageSets exist (created during conversion)
    imagesets_dir = Path(converted_root) / 'ImageSets'
    if imagesets_dir.exists():
        train_file = imagesets_dir / 'train.txt'
        if train_file.exists():
            with open(train_file) as f:
                train_frames = len([l.strip() for l in f if l.strip()])
            print(f"   Training frames (from ImageSets): {train_frames}")
        
        val_file = imagesets_dir / 'val.txt'
        if val_file.exists():
            with open(val_file) as f:
                val_frames = len([l.strip() for l in f if l.strip()])
            print(f"   Validation frames (from ImageSets): {val_frames}")
    
    print()
    print("=" * 80)
    print("Recommendation:")
    print("=" * 80)
    
    if converted_frames == total_expected_frames:
        print("✅ Your dataset appears to include data from all scene types!")
        print("   All expected frames were successfully converted.")
    elif converted_frames >= total_expected_frames * 0.95:
        print("✅ Your dataset likely includes data from all scene types.")
        print("   Minor differences might be due to empty/invalid frames being skipped.")
    else:
        print("⚠️  Your dataset might be missing data from some scene types.")
        print("   Consider re-running the conversion for missing scenes.")
        print()
        print("   To check which scenes were converted, look for conversion logs")
        print("   that show 'Converting sequence: <sequence_name>' messages.")
    
    print()

if __name__ == '__main__':
    main()

