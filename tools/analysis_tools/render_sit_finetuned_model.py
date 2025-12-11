#!/usr/bin/env python3
"""
Use a SiT fine-tuned 3D object detection model to predict cuboids and render via WebRTC.

This script loads a SiT fine-tuned model, runs inference on SiT point clouds,
and visualizes the predictions with the point cloud using Open3D WebRTC server.

Usage:
    python tools/analysis_tools/render_sit_finetuned_model.py \
        --data-root data/sit \
        --infos data/sit/sit_infos_val.pkl \
        --config configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_sit-3d-2class.py \
        --checkpoint work_dirs/sit_finetuned/latest.pth \
        --idx 0 \
        --show-gt

    # With Xvfb for headless environments:
    xvfb-run -a python tools/analysis_tools/render_sit_finetuned_model.py \
        --data-root data/sit \
        --infos data/sit/sit_infos_val.pkl \
        --config configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_sit-3d-2class.py \
        --checkpoint work_dirs/sit_finetuned/latest.pth \
        --idx 0
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

# Set environment variables for headless rendering BEFORE importing Open3D
# These help Open3D work in Docker/headless environments
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
import torch
from mmengine.config import Config

# Torch 2.6 defaults torch.load(weights_only=True). Force weights_only=False so
# older checkpoints load without UnpicklingError.
_orig_torch_load = torch.load


def _torch_load_weights_only_false(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_weights_only_false


def resolve_checkpoint_path(path: str) -> str:
    """Resolve MMDet-style last_checkpoint files to actual .pth path."""
    p = Path(path)
    if p.is_file() and p.suffix not in {'.pth', '.ckpt'}:
        try:
            content = p.read_text().strip().splitlines()
            if content:
                resolved = Path(content[0]).expanduser()
                # If absolute path provided inside file, use it. Otherwise resolve relative to file.
                if not resolved.is_absolute():
                    resolved = (p.parent / resolved).resolve()
                return str(resolved)
        except Exception:
            pass
    return path

from mmdet3d.apis import inference_detector, init_model
from mmdet3d.structures import LiDARInstance3DBoxes


def load_points(bin_path: Path) -> np.ndarray:
    """Load point cloud from binary file.
    
    Args:
        bin_path: Path to .bin file
        
    Returns:
        Point cloud array [N, 3] (x, y, z)
    """
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]


def boxes_to_o3d(boxes: np.ndarray, color: List[float]):
    """Convert boxes [x,y,z,l,w,h,yaw] to Open3D OBB geometries.
    
    Args:
        boxes: Array of shape [N, 7] with [x, y, z, l, w, h, yaw]
        color: RGB color [r, g, b] in range [0, 1]
        
    Returns:
        List of Open3D OrientedBoundingBox geometries
    """
    geometries = []
    skipped = 0
    for b in boxes:
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


def extract_pred_boxes(result, score_thr: float = 0.3):
    """Extract predicted boxes from model result.
    
    Args:
        result: Model inference result (Det3DDataSample)
        score_thr: Score threshold for filtering predictions
        
    Returns:
        Array of boxes [N, 7] and array of scores [N]
    """
    if result is None:
        return np.zeros((0, 7), dtype=np.float32), np.array([])
    
    # Get predictions
    pred_instances = result.pred_instances_3d
    
    if pred_instances is None or len(pred_instances) == 0:
        return np.zeros((0, 7), dtype=np.float32), np.array([])
    
    # Get boxes and scores
    if hasattr(pred_instances, 'bboxes_3d'):
        bboxes_3d = pred_instances.bboxes_3d
        if hasattr(bboxes_3d, 'tensor'):
            boxes = bboxes_3d.tensor.detach().cpu().numpy()
        else:
            boxes = np.asarray(bboxes_3d, dtype=np.float32)
    else:
        boxes = np.zeros((0, 7), dtype=np.float32)
    
    if hasattr(pred_instances, 'scores_3d'):
        scores = pred_instances.scores_3d.detach().cpu().numpy()
    else:
        scores = np.ones(len(boxes))
    
    # Filter by score threshold
    if len(boxes) > 0:
        mask = scores >= score_thr
        boxes = boxes[mask]
        scores = scores[mask]
    
    return boxes, scores


def main():
    parser = argparse.ArgumentParser(
        description='Use SiT fine-tuned model to predict and render cuboids')
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/sit',
        help='SiT dataset root directory')
    parser.add_argument(
        '--infos',
        type=str,
        default='data/sit/sit_infos_val.pkl',
        help='Path to info pkl file')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to model config file (SiT fine-tuned)')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint file')
    parser.add_argument(
        '--idx',
        type=int,
        default=0,
        help='Sample index in the info file')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device for inference (cuda:0, cpu, etc.)')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='Score threshold for predictions')
    parser.add_argument(
        '--show-gt',
        action='store_true',
        help='Also show ground truth boxes (green)')
    parser.add_argument(
        '--port',
        type=int,
        default=8888,
        help='Port for WebRTC server (default: 8888, use Docker -p flag to map ports)')
    args = parser.parse_args()
    
    # Resolve last_checkpoint pointer files to actual .pth path
    args.checkpoint = resolve_checkpoint_path(args.checkpoint)

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

    # Initialize model
    print(f"Loading model from config: {args.config}")
    print(f"Loading checkpoint: {args.checkpoint}")
    model = init_model(args.config, args.checkpoint, device=args.device)
    print("Model loaded successfully!")

    # Run inference
    print("Running inference...")
    result, data = inference_detector(model, str(bin_path))
    print("Inference completed!")

    # Extract predictions
    pred_boxes, pred_scores = extract_pred_boxes(result, score_thr=args.score_thr)
    print(f"Found {len(pred_boxes)} predictions (score >= {args.score_thr})")
    if len(pred_boxes) > 0:
        print(f"Score range: [{pred_scores.min():.3f}, {pred_scores.max():.3f}]")
        print(f"Example box: {pred_boxes[0]}")

    # Build geometries
    geoms = []
    
    # Point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray points
    geoms.append(pcd)
    
    # Predictions (blue/cyan for fine-tuned model)
    if len(pred_boxes) > 0:
        pred_geoms = boxes_to_o3d(pred_boxes, color=[0, 0.8, 1])  # Cyan/blue
        geoms.extend(pred_geoms)
        print(f"Added {len(pred_geoms)} prediction boxes (cyan)")
    
    # Ground truth (green) - optional
    if args.show_gt:
        gt_boxes = []
        for inst in sample.get('instances', []):
            if inst.get('bbox_label', -1) >= 0:
                gt_boxes.append(inst['bbox_3d'])
        gt_boxes = np.asarray(gt_boxes, dtype=np.float32) if gt_boxes else np.zeros((0, 7), np.float32)
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
    if is_headless and not xvfb_available:
        print("\n  💡 Tip: Install Xvfb for better headless support:")
        print("     apt-get update && apt-get install -y xvfb")
        print("     Then run: xvfb-run -a python ...")
    
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
            f"SiT Fine-tuned Model Predictions - Sample {args.idx} (Frame {frame_id})", 1920, 1080)
        
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
