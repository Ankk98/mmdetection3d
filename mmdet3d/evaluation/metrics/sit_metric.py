# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
from mmengine import load
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log

from mmdet3d.registry import METRICS
from mmdet3d.structures import LiDARInstance3DBoxes


@METRICS.register_module()
class SitMetric(BaseMetric):
    """SiT evaluation metric using LiDAR-space 3D IoU.

    This metric is designed for LiDAR-only datasets like SiT that don't have
    valid camera calibration. It uses LiDARInstance3DBoxes.overlaps() to compute
    3D IoU directly in LiDAR coordinate space, avoiding the camera-based
    evaluation that causes segfaults.

    Args:
        ann_file (str): Annotation file path.
        metric (str or List[str], optional): Metrics to be evaluated. This parameter
            is accepted for compatibility with MMEngine's evaluator system but is
            not used by SitMetric. Defaults to 'bbox'.
        pcd_limit_range (List[float]): The range of point cloud used to filter
            invalid predicted boxes. Defaults to [-50, -50, -5, 50, 50, 3].
        iou_thresholds (List[float]): IoU thresholds for AP calculation.
            Defaults to [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95].
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
        pklfile_prefix (str, optional): The prefix of pkl files, including the
            file path and the prefix of filename, e.g., "a/b/prefix". If not
            specified, a temp file will be created. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result to a
            specific format and submit it to the test server.
            Defaults to False.
        submission_prefix (str, optional): The prefix of submission data. If
            not specified, the submission data will not be generated.
            This parameter is accepted for compatibility with configs that
            use KittiMetric, but is not currently used by SitMetric.
            Defaults to None.
        default_cam_key (str): The default camera for lidar to camera
            conversion. This parameter is accepted for compatibility with
            KittiMetric configs, but is not used by SitMetric since it
            operates in LiDAR space only. Defaults to 'CAM2'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'bbox',
                 pcd_limit_range: Optional[List[float]] = None,
                 iou_thresholds: Optional[List[float]] = None,
                 prefix: Optional[str] = None,
                 pklfile_prefix: Optional[str] = None,
                 format_only: bool = False,
                 submission_prefix: Optional[str] = None,
                 default_cam_key: str = 'CAM2',
                 collect_device: str = 'cpu',
                 backend_args: Optional[dict] = None) -> None:
        self.default_prefix = 'Sit metric'
        super(SitMetric, self).__init__(
            collect_device=collect_device, prefix=prefix)
        if pcd_limit_range is None:
            pcd_limit_range = [-50, -50, -5, 50, 50, 3]
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

        self.pcd_limit_range = pcd_limit_range
        self.ann_file = ann_file
        self.pklfile_prefix = pklfile_prefix
        self.format_only = format_only
        if self.format_only:
            assert pklfile_prefix is not None, 'pklfile_prefix must be not '
            'None when format_only is True, otherwise the result files will '
            'be saved to a temp directory which will be cleaned up at the end.'
        self.submission_prefix = submission_prefix
        self.default_cam_key = default_cam_key
        self.iou_thresholds = iou_thresholds
        self.backend_args = backend_args

    def convert_annos_to_kitti_annos(self, data_infos: dict) -> List[dict]:
        """Convert loading annotations to KITTI-style annotations.

        Args:
            data_infos (dict): Data infos including metainfo and annotations
                loaded from ann_file.

        Returns:
            List[dict]: List of KITTI-style annotations.
        """
        data_annos = data_infos.get('data_list', [])
        if len(data_annos) == 0:
            return []
        if not self.format_only:
            cat2label = data_infos['metainfo']['categories']
            label2cat = dict((v, k) for (k, v) in cat2label.items())
            assert 'instances' in data_annos[0]
            for i, annos in enumerate(data_annos):
                if len(annos['instances']) == 0:
                    kitti_annos = {
                        'name': np.array([]),
                        'truncated': np.array([]),
                        'occluded': np.array([]),
                        'alpha': np.array([]),
                        'bbox': np.zeros([0, 4]),
                        'dimensions': np.zeros([0, 3]),
                        'location': np.zeros([0, 3]),
                        'rotation_y': np.array([]),
                        'score': np.array([]),
                    }
                else:
                    kitti_annos = {
                        'name': [],
                        'truncated': [],
                        'occluded': [],
                        'alpha': [],
                        'bbox': [],
                        'location': [],
                        'dimensions': [],
                        'rotation_y': [],
                        'score': []
                    }
                    for instance in annos['instances']:
                        # Use .get to avoid KeyError if bbox_label is missing
                        label = instance.get('bbox_label', None)
                        if label is None:
                            print_log(
                                'Instance missing bbox_label; skipping instance.',
                                logger='current',
                                level='WARNING')
                            continue
                        if label not in label2cat:
                            continue
                        kitti_annos['name'].append(label2cat[label])
                        kitti_annos['truncated'].append(
                            instance.get('truncated', 0.0))
                        kitti_annos['occluded'].append(
                            instance.get('occluded', 0))
                        kitti_annos['alpha'].append(
                            instance.get('alpha', 0.0))
                        kitti_annos['bbox'].append(
                            instance.get('bbox', [0.0, 0.0, 0.0, 0.0]))
                        bbox_3d = instance.get('bbox_3d',
                                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                0.0])
                        kitti_annos['location'].append(bbox_3d[:3])
                        kitti_annos['dimensions'].append(bbox_3d[3:6])
                        kitti_annos['rotation_y'].append(bbox_3d[6])
                        kitti_annos['score'].append(instance.get('score', 1.0))
                    
                    # Check if all instances were filtered out
                    if len(kitti_annos['name']) == 0:
                        # All instances filtered out - initialize with correct shapes
                        kitti_annos = {
                            'name': np.array([]),
                            'truncated': np.array([]),
                            'occluded': np.array([]),
                            'alpha': np.array([]),
                            'bbox': np.zeros([0, 4]),
                            'dimensions': np.zeros([0, 3]),
                            'location': np.zeros([0, 3]),
                            'rotation_y': np.array([]),
                            'score': np.array([]),
                        }
                    else:
                        # Convert lists to numpy arrays
                        for name in kitti_annos:
                            kitti_annos[name] = np.array(kitti_annos[name])
                data_annos[i]['kitti_annos'] = kitti_annos
        return data_annos

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()
            pred_3d = data_sample['pred_instances_3d']
            for attr_name in pred_3d:
                attr_value = pred_3d[attr_name]
                if hasattr(attr_value, 'to') and callable(attr_value.to):
                    pred_3d[attr_name] = attr_value.to('cpu')
                elif torch.is_tensor(attr_value):
                    pred_3d[attr_name] = attr_value.cpu()
                else:
                    pred_3d[attr_name] = attr_value
            result['pred_instances_3d'] = pred_3d
            sample_idx = data_sample['sample_idx']
            result['sample_idx'] = sample_idx
            self.results.append(result)

    def format_results(self,
                      results: List[dict],
                      pklfile_prefix: Optional[str] = None,
                      classes: Optional[List[str]] = None) -> Tuple[dict, Optional[tempfile.TemporaryDirectory]]:
        """Format the results to KITTI-style format.

        Args:
            results (List[dict]): Testing results of the dataset.
            pklfile_prefix (str, optional): The prefix of pkl files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Defaults to None.
            classes (List[str], optional): Name of classes. If None, will use
                self.dataset_meta['classes']. Defaults to None.

        Returns:
            Tuple[dict, Optional[tempfile.TemporaryDirectory]]: Formatted results
                and temporary directory.
        """
        # Use dataset_meta classes if not provided
        if classes is None:
            classes = self.dataset_meta['classes']
        
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
            mmengine.mkdir_or_exist(pklfile_prefix)

        # Format results to KITTI-style
        logger = MMLogger.get_current_instance()
        result_dict = {}
        
        # Check if results is empty
        if len(results) == 0:
            logger.warning('No results to format')
            return {}, tmp_dir
        
        for name in ['pred_instances_3d']:
            if name not in results[0]:
                continue
            
            logger.info('Converting 3D prediction to KITTI format')
            kitti_annos = []
            for result in mmengine.track_iter_progress(results):
                pred = result[name]
                if 'bboxes_3d' not in pred or len(pred['bboxes_3d']) == 0:
                    kitti_annos.append({
                        'name': np.array([]),
                        'truncated': np.array([]),
                        'occluded': np.array([]),
                        'alpha': np.array([]),
                        'bbox': np.zeros([0, 4]),
                        'dimensions': np.zeros([0, 3]),
                        'location': np.zeros([0, 3]),
                        'rotation_y': np.array([]),
                        'score': np.array([]),
                    })
                    continue

                bboxes_3d = pred['bboxes_3d']
                scores_3d = pred['scores_3d']
                labels_3d = pred['labels_3d']

                # Convert to numpy arrays if needed
                if isinstance(bboxes_3d, LiDARInstance3DBoxes):
                    bboxes_3d = bboxes_3d.tensor.detach().cpu().numpy()
                elif torch.is_tensor(bboxes_3d):
                    bboxes_3d = bboxes_3d.detach().cpu().numpy()
                else:
                    bboxes_3d = np.array(bboxes_3d)

                if torch.is_tensor(scores_3d):
                    scores_3d = scores_3d.detach().cpu().numpy()
                else:
                    scores_3d = np.array(scores_3d)

                if torch.is_tensor(labels_3d):
                    labels_3d = labels_3d.cpu().numpy().astype(np.int64)
                else:
                    labels_3d = np.array(labels_3d, dtype=np.int64)

                # Create mask for valid labels (simplified approach)
                valid_mask = np.array([0 <= label < len(classes) for label in labels_3d])
                
                # Log warnings for invalid labels
                invalid_labels = labels_3d[~valid_mask]
                if len(invalid_labels) > 0:
                    for label in invalid_labels:
                        logger.warning(
                            f'Invalid label index {label} (valid range: 0-{len(classes)-1}), '
                            f'skipping this detection')
                
                # Filter out invalid detections
                num_valid = valid_mask.sum()
                if num_valid == 0:
                    # All detections were invalid, create empty annotation
                    kitti_annos.append({
                        'name': np.array([]),
                        'truncated': np.array([]),
                        'occluded': np.array([]),
                        'alpha': np.array([]),
                        'bbox': np.zeros([0, 4]),
                        'dimensions': np.zeros([0, 3]),
                        'location': np.zeros([0, 3]),
                        'rotation_y': np.array([]),
                        'score': np.array([]),
                    })
                    continue

                # Derive valid_labels from valid_mask
                valid_labels = np.array(classes)[labels_3d[valid_mask]].tolist()
                
                kitti_anno = {
                    'name': np.array(valid_labels),
                    'truncated': np.zeros(num_valid),
                    'occluded': np.zeros(num_valid, dtype=np.int32),
                    'alpha': np.zeros(num_valid),
                    'bbox': np.zeros([num_valid, 4]),
                    'dimensions': bboxes_3d[valid_mask, 3:6],
                    'location': bboxes_3d[valid_mask, :3],
                    'rotation_y': bboxes_3d[valid_mask, 6],
                    'score': scores_3d[valid_mask],
                }
                kitti_annos.append(kitti_anno)

            result_dict[name] = kitti_annos

            # Save to pkl file
            pkl_file = osp.join(pklfile_prefix, f'{name}.pkl')
            mmengine.dump(kitti_annos, pkl_file)
            logger.info(f'Result is saved to {pkl_file}.')

        return result_dict, tmp_dir

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of the whole dataset.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        
        # Load annotations
        pkl_infos = load(self.ann_file, backend_args=self.backend_args)
        self.data_infos = self.convert_annos_to_kitti_annos(pkl_infos)
        
        # Format results
        result_dict, tmp_dir = self.format_results(
            results,
            pklfile_prefix=self.pklfile_prefix,
            classes=self.dataset_meta['classes'])

        metric_dict = {}

        if self.format_only:
            logger.info(
                f'results are saved in {osp.dirname(self.pklfile_prefix)}')
            return metric_dict

        # Build GT annotations
        gt_annos = []
        for result in results:
            sample_idx = result['sample_idx']
            if sample_idx < 0 or sample_idx >= len(self.data_infos):
                gt_annos.append({
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                })
            else:
                info = self.data_infos[sample_idx]
                if 'kitti_annos' not in info:
                    gt_annos.append({
                        'name': np.array([]),
                        'truncated': np.array([]),
                        'occluded': np.array([]),
                        'alpha': np.array([]),
                        'bbox': np.zeros([0, 4]),
                        'dimensions': np.zeros([0, 3]),
                        'location': np.zeros([0, 3]),
                        'rotation_y': np.array([]),
                        'score': np.array([]),
                    })
                else:
                    gt_annos.append(info['kitti_annos'])

        # Evaluate using LiDAR 3D IoU
        logger.info('Starting SiT evaluation using LiDAR 3D IoU...')
        ap_dict = self.sit_evaluate(
            result_dict,
            gt_annos,
            logger=logger,
            classes=self.dataset_meta['classes'])
        
        for result in ap_dict:
            metric_dict[result] = ap_dict[result]

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return metric_dict

    def sit_evaluate(self,
                     results_dict: dict,
                     gt_annos: List[dict],
                     classes: Optional[List[str]] = None,
                     logger: Optional[MMLogger] = None) -> Dict[str, float]:
        """Evaluate using LiDAR-space 3D IoU.

        Args:
            results_dict (dict): Results dictionary from format_results.
            gt_annos (List[dict]): Ground truth annotations.
            classes (List[str], optional): Class names. Defaults to None.
            logger (MMLogger, optional): Logger. Defaults to None.

        Returns:
            Dict[str, float]: Evaluation results.
        """
        if logger is None:
            logger = MMLogger.get_current_instance()

        if classes is None:
            classes = self.dataset_meta['classes']

        # Map class names to indices
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        # Get predictions
        if 'pred_instances_3d' not in results_dict:
            logger.warning('No pred_instances_3d found in results_dict')
            return {}

        dt_annos = results_dict['pred_instances_3d']

        # DIAGNOSTICS: Check initial state
        logger.info(f'=== DIAGNOSTICS: Starting evaluation ===')
        logger.info(f'Classes: {classes}')
        logger.info(f'Number of GT samples: {len(gt_annos)}')
        logger.info(f'Number of DT samples: {len(dt_annos)}')
        
        # Check first few samples
        if len(gt_annos) > 0:
            sample_gt = gt_annos[0]
            logger.info(f'Sample GT keys: {list(sample_gt.keys())}')
            logger.info(f'Sample GT name type: {type(sample_gt.get("name", None))}')
            if len(sample_gt.get('name', [])) > 0:
                logger.info(f'Sample GT names (first 5): {sample_gt["name"][:5]}')
                logger.info(f'Sample GT unique names: {np.unique(sample_gt["name"])}')
        
        if len(dt_annos) > 0:
            sample_dt = dt_annos[0]
            logger.info(f'Sample DT keys: {list(sample_dt.keys())}')
            logger.info(f'Sample DT name type: {type(sample_dt.get("name", None))}')
            if len(sample_dt.get('name', [])) > 0:
                logger.info(f'Sample DT names (first 5): {sample_dt["name"][:5]}')
                logger.info(f'Sample DT unique names: {np.unique(sample_dt["name"])}')

        # Calculate 3D IoU for each sample
        all_ious = []
        all_gt_labels = []
        all_dt_labels = []
        all_dt_scores = []

        logger.info(f'Computing 3D IoU for {len(gt_annos)} samples...')
        for i in range(len(gt_annos)):
            gt_anno = gt_annos[i]
            dt_anno = dt_annos[i]

            # Convert to LiDARInstance3DBoxes
            # NOTE: GT bbox_3d format from sit_converter: [x, y, z, l, w, h, rotation_y]
            # where dimensions are [l, w, h] = [size_x, size_y, size_z] in LiDAR format
            # and rotation_y is stored as-is (should be yaw around Z-axis if already in LiDAR space)
            if len(gt_anno['location']) == 0:
                gt_boxes = LiDARInstance3DBoxes(torch.zeros((0, 7)))
            else:
                # DIAGNOSTICS: Log box format
                if i == 0 and len(gt_anno['location']) > 0:
                    logger.info(f'=== DIAGNOSTICS: GT Box Format (sample {i}) ===')
                    logger.info(f'  Location sample: {gt_anno["location"][0]}')
                    logger.info(f'  Dimensions sample: {gt_anno["dimensions"][0]}')
                    logger.info(f'  Rotation_y sample: {gt_anno["rotation_y"][0]}')
                
                # Ensure rotation_y is at least 1D, then reshape to column vector
                rotation_y_gt = np.atleast_1d(gt_anno['rotation_y']).reshape(-1, 1)
                gt_boxes_tensor = torch.from_numpy(
                    np.concatenate([
                        gt_anno['location'],
                        gt_anno['dimensions'],
                        rotation_y_gt
                    ], axis=1).astype(np.float32))
                gt_boxes = LiDARInstance3DBoxes(gt_boxes_tensor)
                
                # DIAGNOSTICS: Log converted box
                if i == 0 and len(gt_boxes) > 0:
                    logger.info(f'  GT box tensor sample: {gt_boxes.tensor[0]}')
                    logger.info(f'  GT box center: {gt_boxes.center[0]}')
                    logger.info(f'  GT box size: {gt_boxes.tensor[0, 3:6]}')
                    logger.info(f'  GT box yaw: {gt_boxes.tensor[0, 6]}')

            if len(dt_anno['location']) == 0:
                dt_boxes = LiDARInstance3DBoxes(torch.zeros((0, 7)))
            else:
                # DIAGNOSTICS: Log prediction box format
                if i == 0 and len(dt_anno['location']) > 0:
                    logger.info(f'=== DIAGNOSTICS: DT Box Format (sample {i}) ===')
                    logger.info(f'  Location sample: {dt_anno["location"][0]}')
                    logger.info(f'  Dimensions sample: {dt_anno["dimensions"][0]}')
                    logger.info(f'  Rotation_y sample: {dt_anno["rotation_y"][0]}')
                
                # Ensure rotation_y is at least 1D, then reshape to column vector
                rotation_y_dt = np.atleast_1d(dt_anno['rotation_y']).reshape(-1, 1)
                dt_boxes_tensor = torch.from_numpy(
                    np.concatenate([
                        dt_anno['location'],
                        dt_anno['dimensions'],
                        rotation_y_dt
                    ], axis=1).astype(np.float32))
                dt_boxes = LiDARInstance3DBoxes(dt_boxes_tensor)
                
                # DIAGNOSTICS: Log converted box
                if i == 0 and len(dt_boxes) > 0:
                    logger.info(f'  DT box tensor sample: {dt_boxes.tensor[0]}')
                    logger.info(f'  DT box center: {dt_boxes.center[0]}')
                    logger.info(f'  DT box size: {dt_boxes.tensor[0, 3:6]}')
                    logger.info(f'  DT box yaw: {dt_boxes.tensor[0, 6]}')

            # Filter by point cloud range
            if len(gt_boxes) > 0:
                limit_range = torch.tensor(self.pcd_limit_range, dtype=torch.float32)
                gt_centers = gt_boxes.center
                valid_gt = ((gt_centers > limit_range[:3]) & 
                           (gt_centers < limit_range[3:])).all(dim=1)
                
                # DIAGNOSTICS: Check filtering
                if i == 0:
                    logger.info(f'=== DIAGNOSTICS: Point cloud range filtering (sample {i}) ===')
                    logger.info(f'  PCD limit range: {self.pcd_limit_range}')
                    logger.info(f'  GT boxes before filter: {len(gt_boxes)}')
                    logger.info(f'  GT centers range: X=[{gt_centers[:, 0].min():.2f}, {gt_centers[:, 0].max():.2f}], '
                              f'Y=[{gt_centers[:, 1].min():.2f}, {gt_centers[:, 1].max():.2f}], '
                              f'Z=[{gt_centers[:, 2].min():.2f}, {gt_centers[:, 2].max():.2f}]')
                    logger.info(f'  GT boxes after filter: {valid_gt.sum()}/{len(gt_boxes)}')
                
                # Ensure mask length matches original array length.
                # Torch -> numpy may yield int/uint dtypes; force boolean to avoid
                # fancy indexing being interpreted as positional indices.
                valid_gt_np = valid_gt.cpu().numpy().astype(bool, copy=False)
                if len(gt_anno['name']) > 0:
                    assert len(valid_gt_np) == len(gt_anno['name']), \
                        f"GT mask length {len(valid_gt_np)} != array length {len(gt_anno['name'])}"
                    gt_labels = gt_anno['name'][valid_gt_np]
                else:
                    gt_labels = np.array([])
                
                gt_boxes = gt_boxes[valid_gt]
            else:
                gt_labels = np.array([])

            if len(dt_boxes) > 0:
                # FIX: Don't filter predictions by point cloud range - evaluate all predictions
                # Predictions outside the valid range will be false positives if they don't match GT
                # This allows proper evaluation even if model predicts outside expected range
                dt_labels = dt_anno['name'] if len(dt_anno['name']) > 0 else np.array([])
                dt_scores = dt_anno['score'] if len(dt_anno['score']) > 0 else np.array([])
                
                # DIAGNOSTICS: Check prediction stats
                if i == 0:
                    dt_centers = dt_boxes.center
                    logger.info(f'  DT boxes (no filtering): {len(dt_boxes)}')
                    logger.info(f'  DT centers range: X=[{dt_centers[:, 0].min():.2f}, {dt_centers[:, 0].max():.2f}], '
                              f'Y=[{dt_centers[:, 1].min():.2f}, {dt_centers[:, 1].max():.2f}], '
                              f'Z=[{dt_centers[:, 2].min():.2f}, {dt_centers[:, 2].max():.2f}]')
            else:
                dt_labels = np.array([])
                dt_scores = np.array([])

            # Calculate 3D IoU
            if len(gt_boxes) > 0 and len(dt_boxes) > 0:
                ious = LiDARInstance3DBoxes.overlaps(dt_boxes, gt_boxes, mode='iou')
                ious = ious.cpu().numpy()
            else:
                ious = np.zeros((len(dt_boxes), len(gt_boxes)))

            all_ious.append(ious)
            all_gt_labels.append(gt_labels)
            all_dt_labels.append(dt_labels)
            all_dt_scores.append(dt_scores)
            
            # DIAGNOSTICS: Log first few samples
            if i < 3:
                logger.info(f'Sample {i}: GT boxes={len(gt_boxes)}, DT boxes={len(dt_boxes)}')
                logger.info(f'  GT labels: {gt_labels[:5] if len(gt_labels) > 0 else "empty"}')
                logger.info(f'  DT labels: {dt_labels[:5] if len(dt_labels) > 0 else "empty"}')
                if len(gt_boxes) > 0 and len(dt_boxes) > 0:
                    logger.info(f'  IoU matrix shape: {ious.shape}, max IoU: {ious.max():.3f}')

        # DIAGNOSTICS: Summary statistics
        total_gt = sum(len(labels) for labels in all_gt_labels)
        total_dt = sum(len(labels) for labels in all_dt_labels)
        logger.info(f'=== DIAGNOSTICS: After IoU calculation ===')
        logger.info(f'Total GT boxes (after filtering): {total_gt}')
        logger.info(f'Total DT boxes (after filtering): {total_dt}')
        
        if total_gt > 0:
            all_gt_flat = np.concatenate(all_gt_labels) if all_gt_labels else np.array([])
            unique_gt, counts_gt = np.unique(all_gt_flat, return_counts=True) if len(all_gt_flat) > 0 else (np.array([]), np.array([]))
            logger.info(f'GT class distribution: {dict(zip(unique_gt, counts_gt)) if len(unique_gt) > 0 else "none"}')
        
        if total_dt > 0:
            all_dt_flat = np.concatenate(all_dt_labels) if all_dt_labels else np.array([])
            unique_dt, counts_dt = np.unique(all_dt_flat, return_counts=True) if len(all_dt_flat) > 0 else (np.array([]), np.array([]))
            logger.info(f'DT class distribution: {dict(zip(unique_dt, counts_dt)) if len(unique_dt) > 0 else "none"}')

        # Calculate AP for each class and IoU threshold
        logger.info('Calculating AP metrics...')
        ap_results = {}
        
        for cls_idx, cls_name in enumerate(classes):
            for iou_thr in self.iou_thresholds:
                # Collect all predictions and GTs for this class
                tp_list = []
                fp_list = []
                scores_list = []
                num_gt = 0

                for i in range(len(gt_annos)):
                    gt_labels = all_gt_labels[i]
                    dt_labels = all_dt_labels[i]
                    dt_scores = all_dt_scores[i]
                    ious = all_ious[i]

                    # Filter by class
                    # DIAGNOSTICS: Check class matching
                    if i == 0 and cls_idx == 0:  # Log for first sample, first class
                        logger.info(f'=== DIAGNOSTICS: Class matching for {cls_name} ===')
                        logger.info(f'  GT labels type: {type(gt_labels)}, dtype: {gt_labels.dtype if hasattr(gt_labels, "dtype") else "N/A"}')
                        logger.info(f'  DT labels type: {type(dt_labels)}, dtype: {dt_labels.dtype if hasattr(dt_labels, "dtype") else "N/A"}')
                        logger.info(f'  cls_name type: {type(cls_name)}, value: "{cls_name}"')
                        logger.info(f'  GT labels sample: {gt_labels[:5] if len(gt_labels) > 0 else "empty"}')
                        logger.info(f'  DT labels sample: {dt_labels[:5] if len(dt_labels) > 0 else "empty"}')
                    
                    gt_mask = (gt_labels == cls_name)
                    dt_mask = (dt_labels == cls_name)
                    
                    if i == 0 and cls_idx == 0:
                        logger.info(f'  GT mask matches: {gt_mask.sum()}/{len(gt_labels)}')
                        logger.info(f'  DT mask matches: {dt_mask.sum()}/{len(dt_labels)}')

                    if not gt_mask.any() and not dt_mask.any():
                        continue

                    num_gt += gt_mask.sum()

                    if not dt_mask.any():
                        continue

                    # Get IoU matrix for this class
                    # Validate that IoU matrix dimensions match mask lengths
                    assert ious.shape[0] == len(dt_mask), \
                        f"IoU matrix rows {ious.shape[0]} != dt_mask length {len(dt_mask)}"
                    assert ious.shape[1] == len(gt_mask), \
                        f"IoU matrix cols {ious.shape[1]} != gt_mask length {len(gt_mask)}"
                    
                    if gt_mask.any():
                        class_ious = ious[dt_mask][:, gt_mask]  # [num_dt, num_gt]
                    else:
                        # No GT boxes for this class, all detections are false positives
                        class_ious = np.zeros((dt_mask.sum(), 0))

                    # Sort detections by score
                    dt_scores_class = dt_scores[dt_mask]
                    sort_idx = np.argsort(-dt_scores_class)
                    class_ious = class_ious[sort_idx]
                    dt_scores_class = dt_scores_class[sort_idx]

                    # Match detections to ground truths
                    matched_gt = np.zeros(gt_mask.sum(), dtype=bool)
                    tp = np.zeros(len(dt_scores_class), dtype=bool)
                    fp = np.zeros(len(dt_scores_class), dtype=bool)

                    for dt_idx in range(len(dt_scores_class)):
                        if class_ious.shape[1] == 0:
                            # No GT boxes for this class, all detections are false positives
                            fp[dt_idx] = True
                            continue

                        # Find best matching GT
                        best_iou = np.max(class_ious[dt_idx])
                        best_gt_idx = np.argmax(class_ious[dt_idx])

                        if best_iou >= iou_thr and not matched_gt[best_gt_idx]:
                            tp[dt_idx] = True
                            matched_gt[best_gt_idx] = True
                        else:
                            fp[dt_idx] = True

                    tp_list.extend(tp)
                    fp_list.extend(fp)
                    scores_list.extend(dt_scores_class)

                # Calculate AP
                if len(scores_list) == 0:
                    ap = 0.0
                else:
                    tp_array = np.array(tp_list)
                    fp_array = np.array(fp_list)
                    scores_array = np.array(scores_list)

                    # Sort by score
                    sort_idx = np.argsort(-scores_array)
                    tp_array = tp_array[sort_idx]
                    fp_array = fp_array[sort_idx]

                    # Calculate cumulative TP and FP
                    tp_cumsum = np.cumsum(tp_array)
                    fp_cumsum = np.cumsum(fp_array)

                    # Calculate precision and recall
                    if num_gt == 0:
                        ap = 0.0
                    else:
                        recalls = tp_cumsum / num_gt
                        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

                        # Apply 11-point interpolation (KITTI-style)
                        ap = 0.0
                        for t in np.arange(0, 1.1, 0.1):
                            if np.sum(recalls >= t) == 0:
                                p = 0
                            else:
                                p = np.max(precisions[recalls >= t])
                            ap += p / 11.0

                metric_key = f'pred_instances_3d/{cls_name}/AP_{iou_thr:.2f}'
                ap_results[metric_key] = float(ap)

        # Calculate mAP (mean over classes and IoU thresholds)
        aps_by_thr = {}
        for iou_thr in self.iou_thresholds:
            aps = [ap_results.get(f'pred_instances_3d/{cls}/AP_{iou_thr:.2f}', 0.0)
                   for cls in classes]
            mAP = np.mean(aps) if aps else 0.0
            metric_key = f'pred_instances_3d/mAP_{iou_thr:.2f}'
            ap_results[metric_key] = float(mAP)
            aps_by_thr[iou_thr] = mAP

        # Calculate overall mAP (mean over all IoU thresholds)
        overall_mAP = np.mean(list(aps_by_thr.values())) if aps_by_thr else 0.0
        ap_results['pred_instances_3d/Overall_mAP'] = float(overall_mAP)

        # Note: mAP@0.5:0.95 is identical to Overall_mAP since iou_thresholds
        # already covers the range [0.5, 0.95] with step 0.05
        # Keeping Overall_mAP as the primary metric for consistency

        logger.info(f'Evaluation completed. Overall mAP: {overall_mAP:.4f}')

        return ap_results

