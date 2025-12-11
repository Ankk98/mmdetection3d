#!/usr/bin/env python3
"""
Render 3D point cloud of SiT dataset ground truth cuboids via WebRTC.

This script loads a SiT dataset sample, extracts ground truth bounding boxes,
and visualizes them with the point cloud using Open3D WebRTC server for browser viewing.

Usage:
    python tools/analysis_tools/render_sit_gt.py \
        --data-root data/sit \
        --infos data/sit/sit_infos_val.pkl \
        --idx 0

    # With Xvfb for headless environments:
    xvfb-run -a python tools/analysis_tools/render_sit_gt.py \
        --data-root data/sit \
        --infos data/sit/sit_infos_val.pkl \
        --idx 0
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

# Set environment variables for headless rendering BEFORE importing Open3D
# These help Open3D work in Docker/headless environments
# Note: Some of these need to be set before Open3D imports, but GUI initialization
# happens later, so setting them here should still help
if 'DISPLAY' not in os.environ:
    # Force software rendering for headless environments
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
    os.environ['GALLIUM_DRIVER'] = 'llvmpipe'
    # Disable X11 authorization to avoid "Authorization required" errors
    os.environ['XAUTHORITY'] = '/dev/null'
    # Use offscreen EGL platform
    os.environ['EGL_PLATFORM'] = 'surfaceless'

import mmengine
import numpy as np
import open3d as o3d


def load_points(bin_path: Path) -> np.ndarray:
    """Load point cloud from binary file.
    
    Args:
        bin_path: Path to .bin file
        
    Returns:
        Point cloud array [N, 3] (x, y, z)
    """
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]


def boxes_to_o3d(boxes: np.ndarray, color: List[float], label_names: Optional[List[str]] = None):
    """Convert boxes [x,y,z,l,w,h,yaw] to Open3D OBB geometries.
    
    Args:
        boxes: Array of shape [N, 7] with [x, y, z, l, w, h, yaw]
        color: RGB color [r, g, b] in range [0, 1]
        label_names: Optional list of label names for each box
        
    Returns:
        List of Open3D OrientedBoundingBox geometries
    """
    geometries = []
    skipped = 0
    for i, b in enumerate(boxes):
        x, y, z, l, w, h, yaw = b.tolist()
        
        # Validate and fix dimensions (must be positive)
        l, w, h = abs(l), abs(w), abs(h)
        
        # Skip boxes with zero or very small dimensions
        if l < 0.01 or w < 0.01 or h < 0.01:
            skipped += 1
            continue
        
        # Normalize yaw to [-pi, pi] range to avoid rotation issues
        yaw = ((yaw + np.pi) % (2 * np.pi)) - np.pi
        
        try:
            R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])
            extent = [l, w, h]
            obb = o3d.geometry.OrientedBoundingBox(center=[x, y, z], R=R, extent=extent)
            obb.color = color
            geometries.append(obb)
        except Exception as e:
            # Skip boxes that fail to create (invalid geometry)
            skipped += 1
            continue
    
    if skipped > 0:
        print(f"  ⚠ Skipped {skipped} invalid boxes (zero/negative dimensions or invalid geometry)")
    
    return geometries


def main():
    parser = argparse.ArgumentParser(
        description='Render SiT dataset ground truth cuboids with point cloud')
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/sit',
        help='SiT dataset root directory')
    parser.add_argument(
        '--infos',
        type=str,
        default='data/sit/sit_infos_val.pkl',
        help='Path to info pkl file (e.g., sit_infos_val.pkl)')
    parser.add_argument(
        '--idx',
        type=int,
        default=0,
        help='Sample index in the info file')
    parser.add_argument(
        '--port',
        type=int,
        default=8888,
        help='Port for WebRTC server (default: 8888, use Docker -p flag to map ports)')
    args = parser.parse_args()
    
    # Set Open3D WebRTC environment variables before enabling WebRTC
    # Bind to 0.0.0.0 to accept connections from outside the container
    os.environ['WEBRTC_IP'] = '0.0.0.0'
    os.environ['WEBRTC_PORT'] = str(args.port)

    # Load info file
    print(f"Loading info file: {args.infos}")
    if not Path(args.infos).exists():
        raise FileNotFoundError(f'Info file not found: {args.infos}')
    
    info = mmengine.load(args.infos)
    if 'data_list' not in info:
        raise ValueError("Info file must contain 'data_list' key")
    
    data_list = info['data_list']
    if args.idx >= len(data_list):
        raise IndexError(f'Index {args.idx} out of range. Dataset has {len(data_list)} samples.')
    
    sample = data_list[args.idx]
    frame_id = sample.get('sample_idx', args.idx)
    
    print(f"Loading sample {args.idx} (frame_id: {frame_id})")

    # Load point cloud
    bin_path = Path(args.data_root) / 'training' / 'velodyne' / f'{frame_id:06d}.bin'
    if not bin_path.exists():
        bin_path = Path(args.data_root) / 'training' / 'velodyne' / f'{frame_id}.bin'
    if not bin_path.exists():
        raise FileNotFoundError(f'Point cloud not found for frame {frame_id}: {bin_path}')
    
    print(f"Loading point cloud: {bin_path}")
    pts = load_points(bin_path)
    print(f"Loaded {len(pts)} points")

    # Extract GT boxes
    gt_boxes = []
    gt_labels = []
    for inst in sample.get('instances', []):
        if inst.get('bbox_label', -1) >= 0:
            gt_boxes.append(inst['bbox_3d'])
            gt_labels.append(inst.get('bbox_label', -1))
    
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32) if gt_boxes else np.zeros((0, 7), np.float32)
    print(f"Found {len(gt_boxes)} ground truth boxes")
    
    if len(gt_boxes) > 0:
        print(f"Box format: [x, y, z, l, w, h, yaw]")
        print(f"Example box: {gt_boxes[0]}")

    # Build geometries
    geoms = []
    
    # Point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray points
    geoms.append(pcd)
    
    # GT boxes (green)
    if len(gt_boxes) > 0:
        gt_geoms = boxes_to_o3d(gt_boxes, color=[0, 1, 0])  # Green
        geoms.extend(gt_geoms)
        print(f"Added {len(gt_geoms)} GT boxes (green)")

    # WebRTC server mode (always enabled)
    print(f"\n{'='*60}")
    print("Starting WebRTC server for browser viewing...")
    print(f"{'='*60}")
    
    # Check if we're in a headless environment
    is_headless = 'DISPLAY' not in os.environ
    if is_headless:
        print("⚠ Running in headless mode (no DISPLAY)")
        print("  Using software rendering (LIBGL_ALWAYS_SOFTWARE=1)")
        print("\n  NOTE: If you encounter a segmentation fault, use Xvfb:")
        print("    xvfb-run -a python ...")
        print("  Or ensure Docker has GPU access: --device=/dev/dri --group-add video")
    
    # Check if Xvfb is available (better option for headless)
    xvfb_available = os.system('which Xvfb > /dev/null 2>&1') == 0
    xvfb_run_available = os.system('which xvfb-run > /dev/null 2>&1') == 0
    if is_headless and (not xvfb_available or not xvfb_run_available):
        print("\n  ⚠ WARNING: Xvfb not found. The script may segfault!")
        print("  To install Xvfb, run:")
        print("     apt-get update && apt-get install -y xvfb")
        print("  Then run the script with:")
        print("     xvfb-run -a python tools/analysis_tools/render_sit_gt.py [args]")
        print("\n  Continuing anyway (may segfault)...")
    
    try:
        # Enable WebRTC
        o3d.visualization.webrtc_server.enable_webrtc()
        
        # Initialize GUI application (required before creating O3DVisualizer)
        # WARNING: In headless environments without Xvfb, app.initialize() may segfault
        # due to EGL initialization failures. This happens in native code and cannot
        # be caught with Python try-except. Use Xvfb or --save as alternatives.
        app = o3d.visualization.gui.Application.instance
        
        print("Initializing Open3D GUI application...")
        print("  (This may segfault in headless environments - use Xvfb to prevent this)")
        
        # Note: app.initialize() may segfault in headless environments if EGL setup fails
        # This is a known issue with Open3D in Docker without proper GPU/display access
        # The segfault happens in native code and cannot be caught with Python exceptions
        app.initialize()
        print("✓ GUI application initialized successfully")
        
        # Create O3DVisualizer (required for WebRTC)
        vis = o3d.visualization.O3DVisualizer(
            f"SiT GT - Sample {args.idx} (Frame {frame_id})", 1920, 1080)
        
        # Add geometries
        for g in geoms:
            vis.add_geometry(str(id(g)), g)
        
        vis.show_axes = True
        
        # Note: O3DVisualizer doesn't support get_render_option()
        # Render options can be adjusted in the browser UI
        
        # Add visualizer as window to application
        app.add_window(vis)
        
        # Note: Open3D WebRTC server is bound to 0.0.0.0 to accept external connections
        # Use Docker port mapping (-p 8888:8888) to expose it
        print(f"\n✓ WebRTC server started!")
        print(f"  Server running on 0.0.0.0:{args.port} (inside container)")
        print(f"  From Docker host: http://localhost:{args.port}")
        print(f"  From remote: http://<server_ip>:{args.port}")
        print(f"  (Map with: docker run -p {args.port}:{args.port} ...)")
        print(f"\n  Press Ctrl+C to stop the server\n")
        
        try:
            app.run()
        except KeyboardInterrupt:
            print("\nShutting down WebRTC server...")
            return
        except Exception as e:
            print(f"\n❌ Error running WebRTC server: {e}")
            raise
            
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as e:
        print(f"\n❌ Failed to start WebRTC server: {e}")
        print("\nThis is often due to missing display/EGL setup in Docker.")
        print("\nTroubleshooting:")
        print("  1. Use Xvfb for headless rendering (recommended):")
        print("     apt-get install xvfb")
        print(f"     xvfb-run -a python {sys.argv[0]} [other args]")
        print("  2. Ensure Docker has proper GPU/display access:")
        print("     docker run --device=/dev/dri --group-add video ...")
        print("  3. Check that Mesa EGL libraries are installed in Docker")
        print(f"\nError details: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
