#!/usr/bin/env python
"""Minimal ROCm inference sanity check for MMDetection3D.

This script loads a MMDetection3D model on device ``cuda:0`` (HIP-backed
under ROCm), runs a single forward pass on a user-provided input, and
prints a short summary of the detection results.

Example (inside ROCm container):

    python tools/test_rocm_inference.py \\
        --config configs/second/hv_second_secfpn_4x8_80e_kitti-3d-3class.py \\
        --checkpoint /path/to/your_checkpoint.pth \\
        --pcd /path/to/example_pointcloud.bin
"""
import argparse
from pathlib import Path
from typing import Any, Dict

import torch

from mmdet3d.apis import inference_detector, init_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='ROCm inference smoke test for MMDetection3D',
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to MMDetection3D config file.',
    )
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='Path to model checkpoint.',
    )
    parser.add_argument(
        '--pcd',
        required=True,
        help='Path to input point cloud (e.g. KITTI .bin file).',
    )
    parser.add_argument(
        '--device',
        default='cuda:0',
        help="Device to use (default: 'cuda:0'). Under ROCm this is HIP-backed.",
    )
    return parser.parse_args()


def summarize_result(result: Any) -> Dict[str, int]:
    """Derive a simple summary from a Det3DDataSample-like result."""
    summary: Dict[str, int] = {}

    if hasattr(result, 'pred_instances_3d'):
        instances = result.pred_instances_3d
        if hasattr(instances, 'bboxes_3d') and hasattr(instances, 'labels_3d'):
            labels = instances.labels_3d.cpu().numpy()
            for label in labels:
                key = str(int(label))
                summary[key] = summary.get(key, 0) + 1
    return summary


def main() -> int:
    args = parse_args()

    cfg_path = Path(args.config)
    ckpt_path = Path(args.checkpoint)
    pcd_path = Path(args.pcd)

    if not cfg_path.is_file():
        raise FileNotFoundError(f'Config not found: {cfg_path}')
    if not ckpt_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    if not pcd_path.is_file():
        raise FileNotFoundError(f'Point cloud not found: {pcd_path}')

    print('=== ROCm MMDetection3D inference test ===')
    print(f'Config:     {cfg_path}')
    print(f'Checkpoint: {ckpt_path}')
    print(f'Input PCD:  {pcd_path}')
    print(f'Using device: {args.device}')

    print('\n[1/3] Initializing model...')
    model = init_model(
        config=str(cfg_path),
        checkpoint=str(ckpt_path),
        device=args.device,
    )
    print('Model initialized.')

    # Sanity check on device
    print('\n[2/3] Checking device & CUDA/HIP availability...')
    print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}')

    print('\n[3/3] Running single inference...')
    result, _ = inference_detector(model, str(pcd_path))
    summary = summarize_result(result)

    print('\n=== Inference summary ===')
    if not summary:
        print('No 3D detections found or result format not recognized.')
    else:
        print('Detected boxes per label index:')
        for label, count in sorted(summary.items(), key=lambda kv: int(kv[0])):
            print(f'  class {label}: {count} boxes')

    print('\nROCm MMDetection3D inference smoke test completed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


