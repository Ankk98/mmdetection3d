#!/usr/bin/env python3
"""Convert all SiT scene types while maintaining a global frame counter.

This script ensures that sequences from different scene types don't overwrite
each other by maintaining a single global frame counter across all conversions.
"""

import argparse
import os
import os.path as osp
import subprocess
import sys
from pathlib import Path


def get_sequence_count(sit_root, scene_type):
    """Count total frames in all sequences of a scene type."""
    scene_dir = Path(sit_root) / scene_type
    if not scene_dir.exists():
        return 0
    
    total_frames = 0
    for item in scene_dir.iterdir():
        if item.is_dir():
            velo_dir = item / 'velo' / 'concat' / 'data'
            if velo_dir.exists():
                pcd_files = list(velo_dir.glob('*.pcd'))
                total_frames += len(pcd_files)
    
    return total_frames


def count_converted_frames(output_root):
    """Count currently converted frames in output directory."""
    velodyne_dir = Path(output_root) / 'training' / 'velodyne'
    if not velodyne_dir.exists():
        return 0
    return len(list(velodyne_dir.glob('*.bin')))


def main():
    parser = argparse.ArgumentParser(
        description='Convert all SiT scene types with global frame counter')
    parser.add_argument('--sit-root', required=True,
                       help='Root directory containing scene type folders')
    parser.add_argument('--output-root', required=True,
                       help='Output directory for converted data')
    parser.add_argument('--split-ratio', nargs=3, type=float, default=[0.7, 0.15, 0.15],
                       help='Train/val/test split ratios')
    parser.add_argument('--create-info', action='store_true',
                       help='Create info files after conversion')
    parser.add_argument('--create-db', action='store_true',
                       help='Create database files after conversion')
    parser.add_argument('--converter-script', 
                       default='tools/dataset_converters/sit_converter.py',
                       help='Path to sit_converter.py script')
    
    args = parser.parse_args()
    
    # Expected scene types
    scene_types = [
        'Cafe_street', 'Cafeteria', 'Corridor', 'Courtyard',
        'Crossroad', 'Hallway', 'Lobby', 'Outdoor_Alley',
        'Subway_Entrance', 'Three_way_Intersection'
    ]
    
    # Find existing scene types
    available_scenes = []
    for scene_type in scene_types:
        scene_dir = Path(args.sit_root) / scene_type
        if scene_dir.exists():
            available_scenes.append(scene_type)
    
    if not available_scenes:
        print(f"Error: No scene types found in {args.sit_root}")
        return 1
    
    print(f"Found {len(available_scenes)} scene types to convert:")
    for scene in available_scenes:
        frame_count = get_sequence_count(args.sit_root, scene)
        print(f"  - {scene}: ~{frame_count} frames")
    print()
    
    # Convert each scene type, maintaining global frame counter
    # Start from current number of converted frames (in case we're resuming)
    global_frame_counter = count_converted_frames(args.output_root)
    all_converted_sequences = []
    
    print(f"Starting conversion from frame index: {global_frame_counter}")
    if global_frame_counter > 0:
        print("(Resuming from existing converted frames)")
    print()
    
    for scene_type in available_scenes:
        scene_path = osp.join(args.sit_root, scene_type)
        
        # Check how many frames exist before conversion
        frames_before = count_converted_frames(args.output_root)
        
        print("=" * 80)
        print(f"Converting scene type: {scene_type}")
        print(f"Starting at frame index: {global_frame_counter}")
        print("=" * 80)
        
        # Build command (don't create info/db during individual conversions)
        cmd = [
            sys.executable,
            args.converter_script,
            '--sit-root', scene_path,
            '--output-root', args.output_root,
            '--convert-all',
            '--start-frame-idx', str(global_frame_counter),
            '--split-ratio'] + [str(x) for x in args.split_ratio]
        
        # Note: We skip --create-info and --create-db here, will do at end
        
        # Run converter
        result = subprocess.run(cmd, cwd=osp.dirname(osp.dirname(osp.dirname(__file__))))
        
        if result.returncode != 0:
            print(f"Warning: Conversion failed for {scene_type}")
            continue
        
        # Count frames after conversion to get actual number converted
        frames_after = count_converted_frames(args.output_root)
        frames_converted = frames_after - frames_before
        
        global_frame_counter = frames_after
        
        print(f"Completed {scene_type}. Converted {frames_converted} frames.")
        print(f"Total frames so far: {global_frame_counter}")
        print()
    
    print("=" * 80)
    print("All scene types converted!")
    print(f"Total frames converted: {global_frame_counter}")
    print("=" * 80)
    
    # Final step: Recreate ImageSets and info files with all data
    print("\nRecreating ImageSets and info files with all converted data...")
    
    final_cmd = [
        sys.executable,
        args.converter_script,
        '--sit-root', args.sit_root,  # Point to a dummy path (we won't convert again)
        '--output-root', args.output_root,
        '--start-frame-idx', '0'  # Not used, but required
    ]
    
    # Just recreate ImageSets - we need to do this manually since all frames are already converted
    # The create_imagesets function will scan the output directory
    from tools.dataset_converters.sit_converter import create_imagesets, create_sit_infos, create_sit_database
    
    # Get all sequences (for ImageSets creation)
    all_sequences = []
    for scene_type in available_scenes:
        scene_dir = Path(args.sit_root) / scene_type
        for item in scene_dir.iterdir():
            if item.is_dir():
                velo_dir = item / 'velo' / 'concat' / 'data'
                if velo_dir.exists() and list(velo_dir.glob('*.pcd')):
                    all_sequences.append(item.name)
    
    if all_sequences:
        create_imagesets(args.output_root, all_sequences, tuple(args.split_ratio))
        print(f"Created ImageSets with {len(all_sequences)} sequences")
        
        if args.create_info:
            create_sit_infos(
                args.output_root,
                pkl_prefix='sit',
                split_ratio=tuple(args.split_ratio))
            print("Created info files")
        
        if args.create_db:
            create_sit_database(args.output_root, pkl_prefix='sit')
            print("Created database files")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

