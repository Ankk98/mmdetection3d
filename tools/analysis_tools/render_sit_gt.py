#!/usr/bin/env python3
"""
Render 3D point cloud of SiT dataset ground truth cuboids with fastplotlib (WebGPU/Vulkan).

This script loads a SiT dataset sample, extracts ground truth bounding boxes,
and visualizes them with a fastplotlib 3D view (interactive orbit/pan/zoom).

Usage:
    python tools/analysis_tools/render_sit_gt.py \
        --data-root data/sit \
        --infos data/sit/sit_infos_val.pkl \
        --idx 0

Notebook:
    from tools.analysis_tools.render_sit_gt import render_sample
    render_sample(...)
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import mmengine
import numpy as np
from fastplotlib import Figure


def load_points(bin_path: Path) -> np.ndarray:
    """Load point cloud from binary file."""
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]


def _yaw_to_rot(yaw: np.ndarray) -> np.ndarray:
    """Yaw (N,) -> rotation matrices (N, 3, 3) about z-axis."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack(
        [
            np.stack([c, -s, np.zeros_like(c)], axis=-1),
            np.stack([s, c, np.zeros_like(c)], axis=-1),
            np.stack([np.zeros_like(c), np.zeros_like(c), np.ones_like(c)], axis=-1),
        ],
        axis=-2,
    )


_EDGE_IDX = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ],
    dtype=np.int64,
)


def boxes_to_lines(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes [x,y,z,l,w,h,yaw] to line segments for fastplotlib."""
    if boxes.size == 0:
        return np.empty((0, 2, 3), dtype=np.float32)

    centers = boxes[:, 0:3]
    dims = np.abs(boxes[:, 3:6])
    yaw = ((boxes[:, 6] + np.pi) % (2 * np.pi)) - np.pi

    base = np.array(
        [
            [1, 1, 1],
            [1, -1, 1],
            [-1, -1, 1],
            [-1, 1, 1],
            [1, 1, -1],
            [1, -1, -1],
            [-1, -1, -1],
            [-1, 1, -1],
        ],
        dtype=np.float32,
    )

    half_dims = dims / 2.0
    corners = base[None, :, :] * half_dims[:, None, :]
    R = _yaw_to_rot(yaw)
    rotated = corners @ np.transpose(R, (0, 2, 1))
    translated = rotated + centers[:, None, :]

    lines = translated[:, _EDGE_IDX, :]
    return lines.reshape(-1, 2, 3).astype(np.float32)


def _compute_scene_range(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        return np.array([-10, -10, -10], dtype=np.float32), np.array([10, 10, 10], dtype=np.float32)
    return points.min(axis=0), points.max(axis=0)


def render_with_fastplotlib(
    pts: np.ndarray,
    boxes_by_label: Dict[str, Tuple[np.ndarray, Tuple[float, float, float, float]]],
    title: str,
    block: bool = True,
) -> None:
    """Render point cloud + multiple box sets using fastplotlib."""
    fig = Figure()
    ax = fig[0, 0]

    ax.add_scatter3d(
        positions=pts.astype(np.float32),
        colors=(0.6, 0.6, 0.6, 0.8),
        sizes=1.0,
    )

    for label, (boxes, color) in boxes_by_label.items():
        lines = boxes_to_lines(boxes)
        if lines.size == 0:
            continue
        ax.add_lines(data=lines, colors=color, thickness=2.0, name=label)

    pmin, pmax = _compute_scene_range(pts)
    ax.camera.set_range(pmin, pmax)
    ax.title = title

    fig.show()
    if block:
        fig.app.run()


def render_sample(data_root: str, infos: str, idx: int, block: bool = False) -> None:
    """Helper for notebook/interactive use."""
    info = mmengine.load(infos)
    data_list = info["data_list"]
    sample = data_list[idx]
    frame_id = sample.get("sample_idx", idx)

    bin_path = Path(data_root) / "training" / "velodyne" / f"{frame_id:06d}.bin"
    if not bin_path.exists():
        bin_path = Path(data_root) / "training" / "velodyne" / f"{frame_id}.bin"

    pts = load_points(bin_path)

    gt_boxes = []
    for inst in sample.get("instances", []):
        if inst.get("bbox_label", -1) >= 0:
            gt_boxes.append(inst["bbox_3d"])
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32) if gt_boxes else np.zeros((0, 7), np.float32)

    render_with_fastplotlib(
        pts,
        {"gt": (gt_boxes, (0.0, 1.0, 0.0, 1.0))},
        title=f"SiT GT - Sample {idx} (Frame {frame_id})",
        block=block,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Render SiT dataset ground truth cuboids with fastplotlib"
    )
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
        '--no-block',
        action='store_true',
        help='Show window/widget without blocking (useful in notebooks)')
    args = parser.parse_args()

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

    bin_path = Path(args.data_root) / 'training' / 'velodyne' / f'{frame_id:06d}.bin'
    if not bin_path.exists():
        bin_path = Path(args.data_root) / 'training' / 'velodyne' / f'{frame_id}.bin'
    if not bin_path.exists():
        raise FileNotFoundError(f'Point cloud not found for frame {frame_id}: {bin_path}')

    pts = load_points(bin_path)

    gt_boxes = []
    for inst in sample.get('instances', []):
        if inst.get('bbox_label', -1) >= 0:
            gt_boxes.append(inst['bbox_3d'])
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32) if gt_boxes else np.zeros((0, 7), np.float32)

    render_with_fastplotlib(
        pts,
        {"gt": (gt_boxes, (0.0, 1.0, 0.0, 1.0))},
        title=f"SiT GT - Sample {args.idx} (Frame {frame_id})",
        block=not args.no_block,
    )


if __name__ == '__main__':
    main()
