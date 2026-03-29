#!/usr/bin/env python3
"""
Visualize a SiT sample with GT (green) and Pred (red) cuboids in a WebRTC
Open3D viewer. Open the URL from a browser (e.g., Quest 3) to inspect in 3D.

Usage:
    python tools/analysis_tools/vis_sit_scene_web.py \
        --data-root data/sit \
        --infos data/sit/sit_infos_val.pkl \
        --pred work_dirs/predictions/val_results/pred_instances_3d.pkl \
        --idx 0

Notes:
- This is a 3D orbit/pan/zoom view; not head-tracked VR.
- Ensure Open3D is installed: `pip install open3d`
"""

import argparse
from pathlib import Path
from typing import List

import mmengine
import numpy as np
import open3d as o3d
import torch


def load_points(bin_path: Path) -> np.ndarray:
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]


def boxes_to_o3d(boxes: np.ndarray, color: List[float]):
    """Convert boxes [x,y,z,l,w,h,yaw] to Open3D OBB geometries."""
    geometries = []
    for b in boxes:
        x, y, z, l, w, h, yaw = b.tolist()
        R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])
        extent = [l, w, h]
        obb = o3d.geometry.OrientedBoundingBox(center=[x, y, z], R=R, extent=extent)
        obb.color = color
        geometries.append(obb)
    return geometries


def main():
    parser = argparse.ArgumentParser(
        description='WebRTC visualization of SiT GT vs Pred 3D boxes')
    parser.add_argument('--data-root', default='data/sit', help='SiT root')
    parser.add_argument('--infos', default='data/sit/sit_infos_val.pkl', help='info pkl')
    parser.add_argument(
        '--pred', default='work_dirs/predictions/val_results/pred_instances_3d.pkl',
        help='prediction pkl')
    parser.add_argument('--idx', type=int, default=0, help='sample index in infos')
    args = parser.parse_args()

    info = mmengine.load(args.infos)
    data_list = info['data_list']
    sample = data_list[args.idx]
    frame_id = sample['sample_idx']

    bin_path = Path(args.data_root) / 'training' / 'velodyne' / f'{frame_id:06d}.bin'
    if not bin_path.exists():
        bin_path = Path(args.data_root) / 'training' / 'velodyne' / f'{frame_id}.bin'
    if not bin_path.exists():
        raise FileNotFoundError(f'Point cloud not found for {frame_id}: {bin_path}')

    pts = load_points(bin_path)

    # GT boxes
    gt_boxes = []
    for inst in sample.get('instances', []):
        if inst.get('bbox_label', -1) >= 0:
            gt_boxes.append(inst['bbox_3d'])
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32) if gt_boxes else np.zeros((0, 7), np.float32)

    # Predictions
    pred_all = mmengine.load(args.pred)
    pred = pred_all[args.idx]
    b3d = pred['bboxes_3d']
    if hasattr(b3d, 'tensor'):
        dt_boxes = b3d.tensor.detach().cpu().numpy()
    else:
        dt_boxes = np.asarray(b3d, dtype=np.float32)

    # Build geometries
    geoms = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    geoms.append(pcd)
    geoms += boxes_to_o3d(gt_boxes, color=[0, 1, 0])   # GT green
    geoms += boxes_to_o3d(dt_boxes, color=[1, 0, 0])   # Pred red

    # Launch WebRTC viewer
    o3d.visualization.webrtc_server.enable_webrtc()
    vis = o3d.visualization.O3DVisualizer("SiT GT vs Pred", 1024, 768)
    for g in geoms:
        vis.add_geometry(str(id(g)), g)
    vis.show_axes = True
    o3d.visualization.webrtc_server.show(vis)
    print("Open in browser: http://localhost:8888 (or host_ip:8888 from Quest)")
    o3d.visualization.webrtc_server.run()


if __name__ == '__main__':
    main()
