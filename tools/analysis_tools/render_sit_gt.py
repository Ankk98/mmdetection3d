#!/usr/bin/env python3
"""
Render 3D point cloud of SiT dataset ground truth cuboids with Three.js (WebGL/GPU).

This script loads a SiT dataset sample, extracts ground truth bounding boxes,
and visualizes them in a browser using Three.js with GPU acceleration.

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
import json
import os
import webbrowser
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import mmengine
import numpy as np


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
    """Convert boxes [x,y,z,l,w,h,yaw] to line segments for visualization."""
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


def _build_editor_frame(
    frame_id: int,
    pts: np.ndarray,
    raw_boxes: List[Dict],
) -> Dict:
    """Prepare per-frame payload for the editor."""
    pmin, pmax = _compute_scene_range(pts)
    center = ((pmin + pmax) / 2.0).tolist()
    size = (pmax - pmin).tolist()
    max_dim = max(size)
    camera_distance = max_dim * 2.0 if max_dim > 0 else 10.0

    return {
        "frameId": int(frame_id),
        "points": pts.astype(np.float32).flatten().tolist(),
        "boxes": raw_boxes,
        "center": center,
        "cameraDistance": camera_distance,
    }


def _generate_html(
    pts: np.ndarray,
    boxes_by_label: Dict[str, Tuple[np.ndarray, Tuple[float, float, float, float]]],
    title: str,
    mode: str = "visualize",
    editor_payload: Optional[Dict] = None,
) -> str:
    """Generate HTML with Three.js visualization or editor."""
    # Prepare point cloud data (flattened for Three.js BufferAttribute)
    points_data = pts.astype(np.float32).flatten().tolist()
    
    # Prepare box data
    boxes_data = {}
    for label, (boxes, color) in boxes_by_label.items():
        lines = boxes_to_lines(boxes)
        if lines.size == 0:
            continue
        # Convert to list of line segments: [[x1,y1,z1, x2,y2,z2], ...]
        line_segments = []
        for line in lines:
            line_segments.append([
                float(line[0][0]), float(line[0][1]), float(line[0][2]),
                float(line[1][0]), float(line[1][1]), float(line[1][2])
            ])
        boxes_data[label] = {
            'lines': line_segments,
            'color': [float(c) for c in color[:3]]  # RGB only
        }
    
    # Compute scene range
    pmin, pmax = _compute_scene_range(pts)
    center = ((pmin + pmax) / 2.0).tolist()
    size = (pmax - pmin).tolist()
    max_dim = max(size)
    camera_distance = max_dim * 2.0
    
    # Load templates
    script_dir = Path(__file__).parent.resolve()
    assets_dir = script_dir / 'assets'
    if mode == "edit":
        template_path = assets_dir / 'sit_editor_template.html'
        logic_path = assets_dir / 'sit_editor_logic.js'
    else:
        template_path = assets_dir / 'sit_viz_template.html'
        logic_path = assets_dir / 'sit_viz_logic.js'
    
    if not template_path.exists():
        # Fallback to checking typical locations if run from root
        potential_assets = Path('tools/analysis_tools/assets').resolve()
        if potential_assets.exists():
            if mode == "edit":
                template_path = potential_assets / 'sit_editor_template.html'
                logic_path = potential_assets / 'sit_editor_logic.js'
            else:
                template_path = potential_assets / 'sit_viz_template.html'
                logic_path = potential_assets / 'sit_viz_logic.js'
    
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found at {template_path}")
    if not logic_path.exists():
        raise FileNotFoundError(f"Logic script not found at {logic_path}")

    with open(template_path, 'r', encoding='utf-8') as f:
        html_template = f.read()
    
    with open(logic_path, 'r', encoding='utf-8') as f:
        js_logic = f.read()
        
    # Replace placeholders
    html_content = html_template.replace('__TITLE__', title)
    if mode == "edit":
        html_content = html_content.replace('__EDITOR_DATA__', json.dumps(editor_payload or {}))
        html_content = html_content.replace('__EDITOR_LOGIC__', js_logic)
    else:
        html_content = html_content.replace('__POINTS_DATA__', json.dumps(points_data))
        html_content = html_content.replace('__BOXES_DATA__', json.dumps(boxes_data))
        html_content = html_content.replace('__CENTER__', json.dumps(center))
        html_content = html_content.replace('__CAMERA_DISTANCE__', json.dumps(camera_distance))
        html_content = html_content.replace('__VISUALIZATION_LOGIC__', js_logic)

    return html_content


def render_with_threejs(
    pts: np.ndarray,
    boxes_by_label: Dict[str, Tuple[np.ndarray, Tuple[float, float, float, float]]],
    title: str,
    block: bool = True,
    output_dir: Optional[str] = None,
    mode: str = "visualize",
    editor_payload: Optional[Dict] = None,
) -> str:
    """
    Render point cloud + multiple box sets using Three.js in browser.
    
    Args:
        pts: Point cloud data
        boxes_by_label: Dictionary mapping labels to (boxes, color) tuples
        title: Title for the visualization
        block: Whether to block execution (not used in Docker-friendly mode)
        output_dir: Directory to save HTML file. If None, uses work_dirs/visualizations
    
    Returns:
        Path to the generated HTML file
    """
    html_content = _generate_html(
        pts,
        boxes_by_label,
        title,
        mode=mode,
        editor_payload=editor_payload,
    )
    
    # Determine output directory
    if output_dir is None:
        # Use work_dirs/visualizations if it exists, otherwise use current directory
        work_dirs = Path("work_dirs")
        if work_dirs.exists():
            output_dir = str(work_dirs / "visualizations")
        else:
            output_dir = "."
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename from title (sanitize for filesystem)
    safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title)
    safe_title = safe_title.replace(' ', '_')[:50]  # Limit length
    html_filename = f"sit_visualization_{safe_title}.html"
    html_path = output_path / html_filename
    
    # Write HTML file
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    html_path_abs = html_path.resolve()
    print(f"\n{'='*70}")
    print(f"Visualization saved to: {html_path_abs}")
    print(f"{'='*70}")
    
    # Check if we're in a Docker/headless environment
    in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER') == '1'
    has_display = os.environ.get('DISPLAY') is not None
    
    if in_docker or not has_display:
        # Docker/headless environment: print instructions
        print("\n📦 Docker/Headless Environment Detected")
        print("\nTo view the visualization:")
        print(f"  1. Copy the file from container to host:")
        print(f"     docker cp <container_name>:{html_path_abs} ./")
        print(f"\n  2. Or access via mounted volume:")
        print(f"     The file is saved at: {html_path_abs}")
        print(f"     If work_dirs is mounted, open it from your host machine.")
        print(f"\n  3. Or start a simple HTTP server (if port is available):")
        print(f"     cd {output_path}")
        print(f"     python -m http.server 8890")
        print(f"     Then open: http://localhost:8890/{html_filename}")
    else:
        # Local environment: try to open browser
        html_url = html_path_abs.as_uri()
        print(f"\nOpening visualization in browser: {html_url}")
        try:
            webbrowser.open(html_url)
            if block:
                print("Visualization opened in browser. Close the browser window or press Ctrl+C to exit.")
                try:
                    import time
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\nExiting...")
        except Exception as e:
            print(f"Could not open browser automatically: {e}")
            print(f"Please open the file manually: {html_path_abs}")
    
    return str(html_path_abs)


def _extract_boxes(sample: Dict, id_to_class: Dict[int, str], colors: Dict[str, Tuple[float, float, float, float]]):
    """Return boxes grouped by class for visualization and raw boxes for editing."""
    boxes_by_class: Dict[str, List] = {}
    raw_boxes: List[Dict] = []
    total_boxes = 0

    for inst in sample.get("instances", []):
        label_id = inst.get("bbox_label", -1)
        if label_id >= 0:
            class_name = id_to_class.get(label_id, "Unknown")
            boxes_by_class.setdefault(class_name, []).append(inst["bbox_3d"])
            box = inst["bbox_3d"]
            raw_boxes.append(
                {
                    "id": f"{sample.get('sample_idx', 0)}_{total_boxes}",
                    "label": class_name,
                    "center": [float(box[0]), float(box[1]), float(box[2])],
                    "dims": [float(box[3]), float(box[4]), float(box[5])],
                    "yaw": float(box[6]),
                }
            )
            total_boxes += 1

    boxes_param = {}
    for cls_name, boxes in boxes_by_class.items():
        boxes_np = np.asarray(boxes, dtype=np.float32)
        boxes_param[cls_name] = (boxes_np, colors.get(cls_name, colors["Unknown"]))

    return boxes_param, raw_boxes


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

    print(f"Found {len(gt_boxes)} GT boxes for sample {idx}")

    render_with_threejs(
        pts,
        {"gt": (gt_boxes, (0.0, 1.0, 0.0, 1.0))},
        title=f"SiT GT - Sample {idx} (Frame {frame_id})",
        block=block,
        output_dir=None,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Render SiT dataset ground truth cuboids with Three.js (browser GPU acceleration)"
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
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save HTML visualization file (default: work_dirs/visualizations)')
    parser.add_argument(
        '--find-next',
        action='store_true',
        help='Find the first sample with annotations starting from idx')
    parser.add_argument(
         '--mode',
         type=str,
         choices=['visualize', 'edit'],
         default='visualize',
         help='visualize = viewer only, edit = launch editor HTML')
    parser.add_argument(
         '--window-size',
         type=int,
         default=5,
         help='How many previous/next frames to preload for editor mode')
    args = parser.parse_args()

    print(f"Loading info file: {args.infos}")
    if not Path(args.infos).exists():
        raise FileNotFoundError(f'Info file not found: {args.infos}')

    info = mmengine.load(args.infos)
    if 'data_list' not in info:
        raise ValueError("Info file must contain 'data_list' key")

    data_list = info['data_list']
    
    current_idx = args.idx
    
    if args.find_next:
        print(f"Searching for sample with instances starting from index {current_idx}...")
        found = False
        
        # Debug statistics
        samples_checked = 0
        samples_with_empty_instances = 0
        samples_with_ignored_instances = 0
        total_ignored_instances = 0
        
        for i in range(current_idx, len(data_list)):
            samples_checked += 1
            sample = data_list[i]
            instances = sample.get('instances', [])
            
            if len(instances) == 0:
                samples_with_empty_instances += 1
                continue
                
            # Check if any instance has a valid label (>=0)
            valid_instances = [inst for inst in instances if inst.get('bbox_label', -1) >= 0]
            
            if len(valid_instances) > 0:
                print(f"Found valid sample at index {i} with {len(valid_instances)} instances.")
                current_idx = i
                found = True
                break
            else:
                samples_with_ignored_instances += 1
                total_ignored_instances += len(instances)
                
        if not found:
            print("\nDiagnosis Report:")
            print(f"  Scanned {samples_checked} samples.")
            print(f"  Samples with empty 'instances' list: {samples_with_empty_instances}")
            print(f"  Samples with instances but all ignored (label < 0): {samples_with_ignored_instances}")
            print(f"  Total ignored instances found: {total_ignored_instances}")
            print("\nConclusion: No usable ground truth found.")
            
            # Check if 'annos' exists in the last checked sample as a fallback hint
            if samples_checked > 0:
                last_sample = data_list[current_idx] # confusingly, current_idx didn't change, but it's fine for a spot check
                if 'annos' in last_sample:
                     print("  Note: Found 'annos' key in sample. The dataset might use the legacy format.")
                     print(f"  'annos' keys: {last_sample['annos'].keys()}")
            
            print("No samples with valid instances found in the rest of the dataset.")
            # Fallback to requested index even if empty
    
    if current_idx >= len(data_list):
        raise IndexError(f'Index {current_idx} out of range. Dataset has {len(data_list)} samples.')

    sample = data_list[current_idx]
    frame_id = sample.get('sample_idx', current_idx)

    bin_path = Path(args.data_root) / 'training' / 'velodyne' / f'{frame_id:06d}.bin'
    if not bin_path.exists():
        bin_path = Path(args.data_root) / 'training' / 'velodyne' / f'{frame_id}.bin'
    if not bin_path.exists():
        raise FileNotFoundError(f'Point cloud not found for frame {frame_id}: {bin_path}')

    pts = load_points(bin_path)

    # Define colors for classes
    colors = {
        'Pedestrian': (0.0, 1.0, 0.0, 1.0), # Green
        'Car': (0.0, 0.6, 1.0, 1.0),        # Blue
        'Cyclist': (1.0, 1.0, 0.0, 1.0),    # Yellow
        'Truck': (1.0, 0.0, 1.0, 1.0),      # Magenta
        'Unknown': (1.0, 0.0, 0.0, 1.0)     # Red
    }
    
    # Get categories from metainfo if available
    categories = info.get('metainfo', {}).get('categories', {})
    # Invert mapping: id -> name
    id_to_class = {v: k for k, v in categories.items()}

    boxes_param, raw_boxes = _extract_boxes(sample, id_to_class, colors)
    total_boxes = sum(len(v[0]) for v in boxes_param.values())

    print(f"Found {total_boxes} GT boxes for sample {current_idx}")
    if total_boxes == 0 and not args.find_next:
        print("Warning: No ground truth boxes found for this sample.")
        print("Try using --find-next to automatically search for a sample with annotations.")
        
        # Check for train info file as alternative
        train_infos_path = Path(args.infos.replace('_val', '_train'))
        if train_infos_path.exists():
            print(f"\nTip: Found {train_infos_path}. \n     The validation set might be unlabeled. Try using the training set:\n     python tools/analysis_tools/render_sit_gt.py --infos {train_infos_path} --idx 0")

    if args.mode == "edit":
        window = max(0, args.window_size)
        start_idx = max(0, current_idx - window)
        end_idx = min(len(data_list), current_idx + window + 1)
        editor_frames: List[Dict] = []

        for i in range(start_idx, end_idx):
            sample_i = data_list[i]
            frame_id_i = sample_i.get('sample_idx', i)
            bin_path_i = Path(args.data_root) / 'training' / 'velodyne' / f'{frame_id_i:06d}.bin'
            if not bin_path_i.exists():
                bin_path_i = Path(args.data_root) / 'training' / 'velodyne' / f'{frame_id_i}.bin'
            if not bin_path_i.exists():
                print(f"Skipping frame {frame_id_i}: point cloud not found at {bin_path_i}")
                continue

            pts_i = load_points(bin_path_i)
            boxes_param_i, raw_boxes_i = _extract_boxes(sample_i, id_to_class, colors)
            editor_frames.append(_build_editor_frame(frame_id_i, pts_i, raw_boxes_i))

            # For main frame, reuse computed boxes for visualization fallback
            if i == current_idx:
                boxes_param = boxes_param_i

        label_colors = {k: [float(c[0]), float(c[1]), float(c[2])] for k, c in colors.items()}
        editor_payload = {
            "frames": editor_frames,
            "initialFrameId": int(frame_id),
            "labelColors": label_colors,
            "labels": list(label_colors.keys()),
            "window": window,
            "title": f"SiT Editor - Frame {frame_id}",
        }

        render_with_threejs(
            pts,
            boxes_param,
            title=f"SiT GT - Sample {current_idx} (Frame {frame_id})",
            block=not args.no_block,
            output_dir=args.output_dir,
            mode="edit",
            editor_payload=editor_payload,
        )
    else:
        render_with_threejs(
            pts,
            boxes_param,
            title=f"SiT GT - Sample {current_idx} (Frame {frame_id})",
            block=not args.no_block,
            output_dir=args.output_dir,
        )


if __name__ == '__main__':
    main()
