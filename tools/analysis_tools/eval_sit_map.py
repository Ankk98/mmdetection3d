import argparse
import glob
import os
from typing import Dict, List, Sequence

import numpy as np
from mmengine.fileio import load

from mmdet3d.evaluation.functional.kitti_utils import kitti_eval
from mmdet3d.evaluation.metrics.kitti_metric import KittiMetric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Python-only mAP evaluator for SiT on KITTI-style dumps.\n\n'
            'This script compares:\n'
            '  1) Ground-truth annotations from a sit_infos_*.pkl file, and\n'
            '  2) Detection results dumped in KITTI text format\n'
            '     (e.g. data/sit/kitti_val_pred).\n\n'
            'It uses the internal KITTI evaluation utilities from '
            'MMDetection3D, but restricts evaluation to 2D bbox metrics '
            'to avoid the native C++ KITTI binary and 3D/BEV kernels that '
            'can segfault on SiT\'s LiDAR-only camera placeholders.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--ann-file',
        type=str,
        required=True,
        help='Path to SiT info pkl file, e.g. data/sit/sit_infos_val.pkl.')
    parser.add_argument(
        '--pred-dir',
        type=str,
        required=True,
        help=('Directory containing KITTI-format prediction .txt files, '
              'e.g. data/sit/kitti_val_pred or a subdirectory such as '
              'data/sit/kitti_val_pred/pred_instances_3d.'))
    parser.add_argument(
        '--classes',
        type=str,
        default='Pedestrian,Car',
        help=('Comma-separated class names to evaluate. These should match '
              'the class names used during conversion (case-sensitive).'))
    parser.add_argument(
        '--metric-types',
        type=str,
        default='bbox',
        help=('Comma-separated metric types to evaluate. For SiT it is '
              'strongly recommended to keep this as "bbox" only to avoid '
              '3D/BEV kernels that may crash.'))
    return parser.parse_args()


def load_gt_annos(ann_file: str) -> List[Dict]:
    """Load GT annotations from a sit_infos_*.pkl file and convert to KITTI
    annotation dicts.

    Supports two formats:

    1) New-style dict with keys:
       - 'metainfo'
       - 'data_list'
       This is the format produced by the current SiT converter /
       create_data.py integration. In this case we delegate conversion to
       :meth:`KittiMetric.convert_annos_to_kitti_annos`.

    2) Legacy KITTI-style list[dict] where each element already contains an
       'annos' field (and often 'image' / 'point_cloud'/ 'lidar_points').
       In this case we skip KittiMetric and read 'annos' directly.
    """
    pkl_infos = load(ann_file)

    # Case 1: new-style dict with metainfo + data_list (preferred path for SiT)
    if isinstance(pkl_infos, dict):
        # Minimal KittiMetric instance, only used for its conversion helper.
        # - metric='bbox' so we only care about 2D KITTI-style evaluation.
        metric_helper = KittiMetric(
            ann_file=ann_file,
            metric='bbox',
            format_only=False,
            backend_args=None)

        data_infos = metric_helper.convert_annos_to_kitti_annos(pkl_infos)
        data_list: Sequence[Dict] = data_infos['data_list']

        # Each element in data_list now has a 'kitti_annos' field.
        gt_annos = [info['kitti_annos'] for info in data_list]
        sample_ids = [str(info['sample_idx']) for info in data_list]
        return gt_annos, sample_ids

    # Case 2: legacy list-style KITTI infos (e.g. older create_data scripts)
    if isinstance(pkl_infos, list):
        if len(pkl_infos) == 0:
            return [], []

        # Expect classic KITTI keys; we read the pre-computed 'annos' field.
        if 'annos' not in pkl_infos[0]:
            raise TypeError(
                'Expected legacy KITTI infos list with an "annos" field in '
                'each element, but got keys: '
                f'{list(pkl_infos[0].keys())}')

        gt_annos: List[Dict] = [info['annos'] for info in pkl_infos]

        sample_ids: List[str] = []
        for idx, info in enumerate(pkl_infos):
            sid = None
            # Try a few common fields in decreasing order of preference.
            if 'image' in info and 'image_idx' in info['image']:
                sid = str(info['image']['image_idx'])
            elif 'point_cloud' in info and 'lidar_idx' in info['point_cloud']:
                sid = str(info['point_cloud']['lidar_idx'])
            elif 'lidar_points' in info and 'lidar_path' in info['lidar_points']:
                # Derive from filename if it looks like KITTI (000123.bin)
                stem = os.path.splitext(
                    os.path.basename(info['lidar_points']['lidar_path']))[0]
                sid = stem
            elif 'image_idx' in info:
                sid = str(info['image_idx'])
            else:
                # Fallback: use list index as sample id
                sid = str(idx)
            sample_ids.append(sid)

        return gt_annos, sample_ids

    raise TypeError(
        f'Unsupported info format loaded from {ann_file!r}: '
        f'expected dict or list, got {type(pkl_infos)}')


def parse_kitti_txt_file(path: str) -> Dict[str, np.ndarray]:
    """Parse a single KITTI-format detection txt file into a dict matching
    the structure expected by the evaluator.
    """
    names = []
    truncated = []
    occluded = []
    alpha = []
    bbox = []
    dimensions = []
    location = []
    rotation_y = []
    score = []

    if not os.path.exists(path):
        # No detections for this frame.
        return dict(
            name=np.array([]),
            truncated=np.array([]),
            occluded=np.array([]),
            alpha=np.array([]),
            bbox=np.zeros((0, 4), dtype=np.float32),
            dimensions=np.zeros((0, 3), dtype=np.float32),
            location=np.zeros((0, 3), dtype=np.float32),
            rotation_y=np.array([]),
            score=np.array([]),
        )

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) < 15:
                # Standard KITTI detection format has 15+ fields.
                # Skip malformed lines rather than crashing.
                continue

            # Refer to KITTI label format:
            # type, truncated, occluded, alpha,
            # bbox_left, bbox_top, bbox_right, bbox_bottom,
            # dim_h, dim_w, dim_l,
            # loc_x, loc_y, loc_z,
            # rot_y, [score]
            names.append(fields[0])
            truncated.append(float(fields[1]))
            occluded.append(int(float(fields[2])))
            alpha.append(float(fields[3]))
            bbox.append([float(fields[4]),
                         float(fields[5]),
                         float(fields[6]),
                         float(fields[7])])
            dimensions.append(
                [float(fields[8]),
                 float(fields[9]),
                 float(fields[10])])
            location.append(
                [float(fields[11]),
                 float(fields[12]),
                 float(fields[13])])
            rotation_y.append(float(fields[14]))
            if len(fields) > 15:
                score.append(float(fields[15]))
            else:
                score.append(1.0)

    if not names:
        return dict(
            name=np.array([]),
            truncated=np.array([]),
            occluded=np.array([]),
            alpha=np.array([]),
            bbox=np.zeros((0, 4), dtype=np.float32),
            dimensions=np.zeros((0, 3), dtype=np.float32),
            location=np.zeros((0, 3), dtype=np.float32),
            rotation_y=np.array([]),
            score=np.array([]),
        )

    return dict(
        name=np.asarray(names),
        truncated=np.asarray(truncated, dtype=np.float32),
        occluded=np.asarray(occluded, dtype=np.int32),
        alpha=np.asarray(alpha, dtype=np.float32),
        bbox=np.asarray(bbox, dtype=np.float32),
        dimensions=np.asarray(dimensions, dtype=np.float32),
        location=np.asarray(location, dtype=np.float32),
        rotation_y=np.asarray(rotation_y, dtype=np.float32),
        score=np.asarray(score, dtype=np.float32),
    )


def load_dt_annos(pred_dir: str,
                  sample_ids: Sequence[str]) -> List[Dict]:
    """Load detection annotations from KITTI txt files.

    We assume one txt per frame, with filename matching the sample_idx
    (e.g., 000000.txt). If there are multiple matching layouts inside
    pred_dir (e.g., nested subdirs), users should pass the precise leaf
    directory that directly contains the txt files.
    """
    # Heuristic: if pred_dir itself has txt files, use them; otherwise,
    # allow the user to point at the parent directory and we try the
    # first "pred_instances_3d" subdirectory.
    txt_glob = os.path.join(pred_dir, '*.txt')
    found_txt = glob.glob(txt_glob)
    if not found_txt:
        candidate = os.path.join(pred_dir, 'pred_instances_3d')
        if os.path.isdir(candidate):
            pred_dir = candidate

    dt_annos: List[Dict] = []
    for sid in sample_ids:
        # Zero-pad to 6 digits by convention, but fall back to raw sid if
        # that does not exist.
        padded = sid.zfill(6)
        paths = [
            os.path.join(pred_dir, f'{padded}.txt'),
            os.path.join(pred_dir, f'{sid}.txt'),
        ]
        for path in paths:
            if os.path.exists(path):
                dt_annos.append(parse_kitti_txt_file(path))
                break
        else:
            # No prediction for this frame: append empty annotation.
            dt_annos.append(parse_kitti_txt_file(''))

    return dt_annos


def main() -> None:
    args = parse_args()

    classes = [c.strip() for c in args.classes.split(',') if c.strip()]
    metric_types = [m.strip() for m in args.metric_types.split(',') if m.strip()]

    if not classes:
        raise SystemExit('No valid classes specified via --classes.')
    if not metric_types:
        raise SystemExit('No valid metric types specified via --metric-types.')

    print('=== SiT mAP Evaluation (Python-only, KITTI-style bbox) ===')
    print(f'GT ann_file : {args.ann_file}')
    print(f'Pred dir   : {args.pred_dir}')
    print(f'Classes    : {classes}')
    print(f'MetricType : {metric_types}')
    print()

    gt_annos, sample_ids = load_gt_annos(args.ann_file)
    dt_annos = load_dt_annos(args.pred_dir, sample_ids)

    if len(gt_annos) != len(dt_annos):
        raise SystemExit(
            f'GT and prediction length mismatch: '
            f'{len(gt_annos)} GT vs {len(dt_annos)} predictions.')

    # Restrict to bbox-only evaluation by default for robustness on SiT.
    eval_types = metric_types
    print('Running KITTI evaluator with eval_types =', eval_types)
    print()

    result_str, metrics = kitti_eval(
        gt_annos=gt_annos,
        dt_annos=dt_annos,
        current_classes=classes,
        eval_types=eval_types)

    print(result_str)
    print('--- Parsed metrics (subset) ---')
    # Show compact summary focusing on 2D AP40 strict/loose for each class.
    for key in sorted(metrics.keys()):
        if '2D_AP40' in key or '2D_AP11' in key:
            print(f'{key}: {metrics[key]:.4f}')


if __name__ == '__main__':
    main()


