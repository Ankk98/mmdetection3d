#!/usr/bin/env python3
"""Validate SiT GT annotations without visualization.

This script checks the converted SiT info pkl for:
  - class counts and sample coverage per class
  - duplicate sample_idx
  - NaN/Inf values in boxes
  - box dimension validity (>0)
  - box centers within the specified point cloud range
  - yaw range sanity

It does NOT render any visuals; it prints a concise report to stdout and exits
with non-zero status if critical issues are found.

Example:
    python tools/analysis_tools/validate_sit_gt.py \\
        --info data/sit/sit_infos_val.pkl \\
        --pcd-limit-range -50 -50 -5 50 50 3
"""

import argparse
import math
import sys
from collections import Counter, defaultdict

import mmengine
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate SiT GT annotations (no visualization)')
    parser.add_argument(
        '--info',
        required=True,
        help='Path to sit_infos_<split>.pkl (train/val/test)')
    parser.add_argument(
        '--pcd-limit-range',
        type=float,
        nargs=6,
        default=[-50, -50, -5, 50, 50, 3],
        help='Point cloud range: xmin ymin zmin xmax ymax zmax')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Optional: limit number of samples to speed up checks')
    return parser.parse_args()


def within_range(centers, limit):
    lim = np.array(limit, dtype=np.float32)
    return ((centers > lim[:3]) & (centers < lim[3:])).all(axis=1)


def main():
    args = parse_args()
    info = mmengine.load(args.info)

    data_list = info['data_list']
    categories = info.get('metainfo', {}).get('categories', {})
    label2name = {v: k for k, v in categories.items()} if categories else None

    total_samples = len(data_list)
    sample_limit = args.max_samples or total_samples

    class_counts = Counter()
    samples_with_class = Counter()
    dup_check = set()
    duplicate_sample_idx = []
    out_of_range_boxes = 0
    nan_boxes = 0
    invalid_dim = 0
    yaw_outside_pi = 0

    for i, sample in enumerate(data_list[:sample_limit]):
        sample_idx = sample.get('sample_idx', i)
        if sample_idx in dup_check:
            duplicate_sample_idx.append(sample_idx)
        else:
            dup_check.add(sample_idx)

        instances = sample.get('instances', [])
        names = []
        centers = []
        dims = []
        yaws = []

        for inst in instances:
            # name
            label = inst.get('bbox_label', inst.get('bbox_label_3d', -1))
            if label2name is not None and label in label2name:
                names.append(label2name[label])
            else:
                # fallback to stored name if present
                names.append(inst.get('name', 'unknown'))

            # box data
            bbox_3d = np.array(inst.get('bbox_3d', []), dtype=np.float32)
            if bbox_3d.shape[0] == 7:
                centers.append(bbox_3d[:3])
                dims.append(bbox_3d[3:6])
                yaws.append(bbox_3d[6])

        if names:
            per_sample_classes = set(names)
            for cls in per_sample_classes:
                samples_with_class[cls] += 1
            class_counts.update(names)

        if centers:
            centers_np = np.stack(centers, axis=0)
            dims_np = np.stack(dims, axis=0)
            yaws_np = np.array(yaws)

            # Range check
            mask = within_range(centers_np, args.pcd_limit_range)
            out_of_range_boxes += (~mask).sum()

            # NaN/Inf check
            if not np.isfinite(centers_np).all() or not np.isfinite(dims_np).all(
            ) or not np.isfinite(yaws_np).all():
                nan_boxes += 1

            # Dimension validity (>0)
            invalid_dim += (dims_np <= 0).any(axis=1).sum()

            # Yaw sanity (-pi, pi)
            yaw_outside_pi += (np.abs(yaws_np) > math.pi).sum()

    print('=== SiT GT Validation Report (no visualization) ===')
    print(f'Info file         : {args.info}')
    print(f'Samples checked   : {sample_limit} / {total_samples}')
    print(f'Point cloud range : {args.pcd_limit_range}')
    print('')
    print('Class counts:')
    for cls, cnt in class_counts.most_common():
        print(f'  {cls:12s}: {cnt:8d}')
    print('')
    print('Sample coverage (% of samples containing the class):')
    for cls, cnt in samples_with_class.most_common():
        pct = 100.0 * cnt / sample_limit
        print(f'  {cls:12s}: {pct:6.2f}% ({cnt}/{sample_limit})')

    print('')
    print('Integrity checks:')
    print(f'  Duplicate sample_idx : {len(duplicate_sample_idx)}')
    print(f'  Boxes out of range   : {out_of_range_boxes}')
    print(f'  NaN/Inf boxes        : {nan_boxes}')
    print(f'  Invalid dimensions   : {invalid_dim}')
    print(f'  Yaw outside [-pi,pi] : {yaw_outside_pi}')

    issues = duplicate_sample_idx or out_of_range_boxes > 0 or nan_boxes > 0 or invalid_dim > 0
    if issues:
        print('\\nResult: ❌ Issues detected (see counts above)')
        sys.exit(1)
    else:
        print('\\nResult: ✅ No critical issues detected')
        sys.exit(0)


if __name__ == '__main__':
    main()
