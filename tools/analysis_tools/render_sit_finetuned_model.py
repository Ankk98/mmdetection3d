#!/usr/bin/env python3
"""
Use a SiT fine-tuned 3D object detection model to predict cuboids and render with Three.js.

This script loads a SiT fine-tuned model, runs inference on SiT point clouds,
and visualizes the predictions with the point cloud using Three.js in browser.

Usage:
    python tools/analysis_tools/render_sit_finetuned_model.py \
        --data-root data/sit \
        --infos data/sit/sit_infos_val.pkl \
        --config configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_sit-3d-2class.py \
        --checkpoint work_dirs/sit_finetuned/latest.pth \
        --idx 0 \
        --show-gt
"""

import argparse
import json
import os
import webbrowser
from pathlib import Path
from typing import Dict, Tuple, Optional

import mmengine
import numpy as np
import torch
from mmdet3d.apis import inference_detector, init_model

# Import shared visualization helpers from render_sit_gt
import sys
sys.path.append(str(Path(__file__).parent))
from render_sit_gt import render_with_threejs, load_points

# Torch 2.6 defaults torch.load(weights_only=True). Force weights_only=False so
# older checkpoints load without UnpicklingError.
_orig_torch_load = torch.load


def _torch_load_weights_only_false(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_weights_only_false


def extract_pred_boxes(result, score_thr: float = 0.3):
    """Extract predicted boxes from model result."""
    if result is None:
        return np.zeros((0, 7), dtype=np.float32), np.array([])
    
    pred_instances = result.pred_instances_3d
    
    if pred_instances is None or len(pred_instances) == 0:
        return np.zeros((0, 7), dtype=np.float32), np.array([])
    
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
    
    if len(boxes) > 0:
        mask = scores >= score_thr
        boxes = boxes[mask]
        scores = scores[mask]
    
    return boxes, scores


def resolve_checkpoint_path(path: str) -> str:
    """Resolve MMDet-style last_checkpoint files to actual .pth path."""
    p = Path(path)
    if p.is_file() and p.suffix not in {'.pth', '.ckpt'}:
        try:
            content = p.read_text().strip().splitlines()
            if content:
                resolved = Path(content[0]).expanduser()
                if not resolved.is_absolute():
                    resolved = (p.parent / resolved).resolve()
                return str(resolved)
        except Exception:
            pass
    return path


def main():
    parser = argparse.ArgumentParser(
        description='Use SiT fine-tuned model to predict and render cuboids with Three.js')
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
        '--no-block',
        action='store_true',
        help='Show window/widget without blocking (useful in notebooks)')
    args = parser.parse_args()
    
    args.checkpoint = resolve_checkpoint_path(args.checkpoint)

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

    print(f"Loading model from config: {args.config}")
    print(f"Loading checkpoint: {args.checkpoint}")
    model = init_model(args.config, args.checkpoint, device=args.device)

    print("Running inference...")
    result, data = inference_detector(model, str(bin_path))
    print("Inference completed!")

    pred_boxes, pred_scores = extract_pred_boxes(result, score_thr=args.score_thr)
    print(f"Found {len(pred_boxes)} predictions (score >= {args.score_thr})")
    if len(pred_boxes) > 0:
        print(f"Score range: [{pred_scores.min():.3f}, {pred_scores.max():.3f}]")
        print(f"Example box: {pred_boxes[0]}")

    # Build scene data for Three.js
    boxes_param = {}
    
    # 1. Predictions (Cyan)
    if len(pred_boxes) > 0:
        boxes_param['Prediction (Finetuned)'] = (pred_boxes, (0.0, 0.8, 1.0, 1.0))

    # 2. Ground Truth (Green)
    if args.show_gt:
        gt_boxes = []
        for inst in sample.get('instances', []):
            if inst.get('bbox_label', -1) >= 0:
                gt_boxes.append(inst['bbox_3d'])
        
        if gt_boxes:
            gt_boxes_np = np.asarray(gt_boxes, dtype=np.float32)
            boxes_param['Ground Truth'] = (gt_boxes_np, (0.0, 1.0, 0.0, 1.0))

    render_with_threejs(
        pts,
        boxes_param,
        title=f"SiT Finetuned Prediction - Sample {args.idx} (Frame {frame_id})",
        block=not args.no_block,
        output_dir=None
    )


if __name__ == '__main__':
    main()
