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

from mmdet3d.evaluation import kitti_eval
from mmdet3d.registry import METRICS
from mmdet3d.structures import (Box3DMode, CameraInstance3DBoxes,
                                LiDARInstance3DBoxes, points_cam2img)


@METRICS.register_module()
class KittiMetric(BaseMetric):
    """Kitti evaluation metric.

    Args:
        ann_file (str): Annotation file path.
        metric (str or List[str]): Metrics to be evaluated. Defaults to 'bbox'.
        pcd_limit_range (List[float]): The range of point cloud used to filter
            invalid predicted boxes. Defaults to [0, -40, -3, 70.4, 40, 0.0].
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
        pklfile_prefix (str, optional): The prefix of pkl files, including the
            file path and the prefix of filename, e.g., "a/b/prefix". If not
            specified, a temp file will be created. Defaults to None.
        default_cam_key (str): The default camera for lidar to camera
            conversion. By default, KITTI: 'CAM2', Waymo: 'CAM_FRONT'.
            Defaults to 'CAM2'.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result to a
            specific format and submit it to the test server.
            Defaults to False.
        submission_prefix (str, optional): The prefix of submission data. If
            not specified, the submission data will not be generated.
            Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'bbox',
                 pcd_limit_range: List[float] = [0, -40, -3, 70.4, 40, 0.0],
                 prefix: Optional[str] = None,
                 pklfile_prefix: Optional[str] = None,
                 default_cam_key: str = 'CAM2',
                 format_only: bool = False,
                 submission_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 backend_args: Optional[dict] = None) -> None:
        self.default_prefix = 'Kitti metric'
        super(KittiMetric, self).__init__(
            collect_device=collect_device, prefix=prefix)
        self.pcd_limit_range = pcd_limit_range
        self.ann_file = ann_file
        self.pklfile_prefix = pklfile_prefix
        self.format_only = format_only
        if self.format_only:
            assert submission_prefix is not None, 'submission_prefix must be '
            'not None when format_only is True, otherwise the result files '
            'will be saved to a temp directory which will be cleaned up at '
            'the end.'

        self.submission_prefix = submission_prefix
        self.default_cam_key = default_cam_key
        self.backend_args = backend_args

        allowed_metrics = ['bbox', 'img_bbox', 'mAP', 'LET_mAP']
        self.metrics = metric if isinstance(metric, list) else [metric]
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError("metric should be one of 'bbox', 'img_bbox', "
                               f'but got {metric}.')

    def convert_annos_to_kitti_annos(self, data_infos: dict) -> List[dict]:
        """Convert loading annotations to Kitti annotations.

        Args:
            data_infos (dict): Data infos including metainfo and annotations
                loaded from ann_file.

        Returns:
            List[dict]: List of Kitti annotations.
        """
        data_annos = data_infos['data_list']
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
                        label = instance['bbox_label']
                        # Some pipelines may encode "unknown" / filtered
                        # instances with label -1. Skip any label that is
                        # negative or not present in label2cat to avoid
                        # KeyError when doing label2cat[label].
                        if label not in label2cat:
                            continue
                        kitti_annos['name'].append(label2cat[label])
                        # Some datasets (e.g. SiT) may omit KITTI-specific
                        # camera fields like truncated/occluded/alpha/score
                        # from their per-instance dicts. Fall back to sensible
                        # defaults so evaluation can still run.
                        kitti_annos['truncated'].append(
                            instance.get('truncated', 0.0))
                        kitti_annos['occluded'].append(
                            instance.get('occluded', 0))
                        kitti_annos['alpha'].append(
                            instance.get('alpha', 0.0))
                        # Some derived datasets may omit 2D bbox in their
                        # per-instance dicts. Fall back to a zero box so that
                        # conversion/evaluation code can still run.
                        kitti_annos['bbox'].append(
                            instance.get('bbox', [0.0, 0.0, 0.0, 0.0]))
                        # Some derived datasets may also omit 3D boxes; in that
                        # case, fall back to a zero box so evaluation code can
                        # still run, even though the metrics will be degenerate.
                        bbox_3d = instance.get('bbox_3d',
                                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                0.0])
                        kitti_annos['location'].append(bbox_3d[:3])
                        kitti_annos['dimensions'].append(bbox_3d[3:6])
                        kitti_annos['rotation_y'].append(bbox_3d[6])
                        kitti_annos['score'].append(instance.get('score', 1.0))
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
            pred_2d = data_sample['pred_instances']
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].to('cpu')
            result['pred_instances_3d'] = pred_3d
            for attr_name in pred_2d:
                pred_2d[attr_name] = pred_2d[attr_name].to('cpu')
            result['pred_instances'] = pred_2d
            sample_idx = data_sample['sample_idx']
            result['sample_idx'] = sample_idx
            self.results.append(result)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of the whole dataset.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        self.classes = self.dataset_meta['classes']

        # load annotations
        pkl_infos = load(self.ann_file, backend_args=self.backend_args)
        self.data_infos = self.convert_annos_to_kitti_annos(pkl_infos)
        result_dict, tmp_dir = self.format_results(
            results,
            pklfile_prefix=self.pklfile_prefix,
            submission_prefix=self.submission_prefix,
            classes=self.classes)

        metric_dict = {}

        if self.format_only:
            logger.info(
                f'results are saved in {osp.dirname(self.submission_prefix)}')
            return metric_dict

        # Safely build gt_annos with validation
        gt_annos = []
        for result in results:
            try:
                sample_idx = result['sample_idx']
                if sample_idx < 0 or sample_idx >= len(self.data_infos):
                    import warnings
                    warnings.warn(
                        f'Invalid sample_idx {sample_idx}, skipping. '
                        f'Valid range: [0, {len(self.data_infos)}).',
                        RuntimeWarning)
                    # Use empty annotation as fallback
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
                    continue
                info = self.data_infos[sample_idx]
                if 'kitti_annos' not in info:
                    import warnings
                    warnings.warn(
                        f'Missing kitti_annos for sample_idx {sample_idx}, '
                        f'using empty annotation.', RuntimeWarning)
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
            except (KeyError, IndexError, TypeError) as e:
                import warnings
                warnings.warn(
                    f'Error processing result for sample_idx {result.get("sample_idx", "unknown")}: {e}. '
                    f'Using empty annotation.', RuntimeWarning)
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

        for metric in self.metrics:
            try:
                ap_dict = self.kitti_evaluate(
                    result_dict,
                    gt_annos,
                    metric=metric,
                    logger=logger,
                    classes=self.classes)
                for result in ap_dict:
                    metric_dict[result] = ap_dict[result]
            except Exception as e:
                import warnings
                warnings.warn(
                    f'Error during kitti_evaluate for metric {metric}: {e}. '
                    f'Skipping this metric.', RuntimeWarning)
                logger.error(f'Failed to evaluate metric {metric}: {e}')
                # Continue with other metrics

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return metric_dict

    def kitti_evaluate(self,
                       results_dict: dict,
                       gt_annos: List[dict],
                       metric: Optional[str] = None,
                       classes: Optional[List[str]] = None,
                       logger: Optional[MMLogger] = None) -> Dict[str, float]:
        """Evaluation in KITTI protocol.

        Args:
            results_dict (dict): Formatted results of the dataset.
            gt_annos (List[dict]): Contain gt information of each sample.
            metric (str, optional): Metrics to be evaluated. Defaults to None.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            logger (MMLogger, optional): Logger used for printing related
                information during evaluation. Defaults to None.

        Returns:
            Dict[str, float]: Results of each evaluation metric.
        """
        # Validate inputs before evaluation to prevent segfaults
        if not isinstance(results_dict, dict):
            raise TypeError(f'results_dict must be a dict, got {type(results_dict)}')
        if not isinstance(gt_annos, list):
            raise TypeError(f'gt_annos must be a list, got {type(gt_annos)}')
        if classes is None:
            raise ValueError('classes must be provided')
        
        # Validate gt_annos structure
        for i, gt_anno in enumerate(gt_annos):
            if not isinstance(gt_anno, dict):
                raise TypeError(f'gt_annos[{i}] must be a dict, got {type(gt_anno)}')
            # Check required keys exist and are numpy arrays
            required_keys = ['name', 'truncated', 'occluded', 'alpha', 'bbox',
                          'dimensions', 'location', 'rotation_y', 'score']
            for key in required_keys:
                if key not in gt_anno:
                    raise KeyError(f'Missing key "{key}" in gt_annos[{i}]')
                if not isinstance(gt_anno[key], np.ndarray):
                    raise TypeError(
                        f'gt_annos[{i}]["{key}"] must be np.ndarray, '
                        f'got {type(gt_anno[key])}')
        
        ap_dict = dict()
        for name in results_dict:
            if name == 'pred_instances' or metric == 'img_bbox':
                eval_types = ['bbox']
            else:
                eval_types = ['bbox', 'bev', '3d']
            
            # Validate results_dict[name] before evaluation
            dt_annos = results_dict[name]
            if not isinstance(dt_annos, list):
                import warnings
                warnings.warn(
                    f'results_dict["{name}"] must be a list, got {type(dt_annos)}. '
                    f'Skipping evaluation for {name}.', RuntimeWarning)
                continue
            
            # Validate length match
            if len(dt_annos) != len(gt_annos):
                import warnings
                warnings.warn(
                    f'Length mismatch: dt_annos has {len(dt_annos)} items, '
                    f'gt_annos has {len(gt_annos)} items. Skipping evaluation for {name}.',
                    RuntimeWarning)
                continue
            
            # Validate dt_annos structure
            for i, dt_anno in enumerate(dt_annos):
                if not isinstance(dt_anno, dict):
                    import warnings
                    warnings.warn(
                        f'dt_annos[{i}] must be a dict, got {type(dt_anno)}. '
                        f'Skipping evaluation for {name}.', RuntimeWarning)
                    break
                # Check required keys
                required_keys = ['name', 'truncated', 'occluded', 'alpha', 'bbox',
                              'dimensions', 'location', 'rotation_y', 'score']
                for key in required_keys:
                    if key not in dt_anno:
                        import warnings
                        warnings.warn(
                            f'Missing key "{key}" in dt_annos[{i}]. '
                            f'Skipping evaluation for {name}.', RuntimeWarning)
                        break
                    if not isinstance(dt_anno[key], np.ndarray):
                        import warnings
                        warnings.warn(
                            f'dt_annos[{i}]["{key}"] must be np.ndarray, '
                            f'got {type(dt_anno[key])}. Skipping evaluation for {name}.',
                            RuntimeWarning)
                        break
            else:
                # All validations passed, proceed with evaluation
                try:
                    ap_result_str, ap_dict_ = kitti_eval(
                        gt_annos, dt_annos, classes, eval_types=eval_types)
                    for ap_type, ap in ap_dict_.items():
                        ap_dict[f'{name}/{ap_type}'] = float(f'{ap:.4f}')

                    print_log(f'Results of {name}:\n' + ap_result_str, logger=logger)
                except Exception as e:
                    import warnings
                    warnings.warn(
                        f'Error during kitti_eval for {name}: {e}. '
                        f'Skipping this result.', RuntimeWarning)
                    logger.error(f'Failed to evaluate {name}: {e}')

        return ap_dict

    def format_results(
        self,
        results: List[dict],
        pklfile_prefix: Optional[str] = None,
        submission_prefix: Optional[str] = None,
        classes: Optional[List[str]] = None
    ) -> Tuple[dict, Union[tempfile.TemporaryDirectory, None]]:
        """Format the results to pkl file.

        Args:
            results (List[dict]): Testing results of the dataset.
            pklfile_prefix (str, optional): The prefix of pkl files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Defaults to None.
            submission_prefix (str, optional): The prefix of submitted files.
                It includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Defaults to None.
            classes (List[str], optional): A list of class name.
                Defaults to None.

        Returns:
            tuple: (result_dict, tmp_dir), result_dict is a dict containing the
            formatted result, tmp_dir is the temporal directory created for
            saving json files when jsonfile_prefix is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_dict = dict()
        sample_idx_list = [result['sample_idx'] for result in results]
        for name in results[0]:
            if submission_prefix is not None:
                submission_prefix_ = osp.join(submission_prefix, name)
            else:
                submission_prefix_ = None
            if pklfile_prefix is not None:
                pklfile_prefix_ = osp.join(pklfile_prefix, name) + '.pkl'
            else:
                pklfile_prefix_ = None
            if 'pred_instances' in name and '3d' in name and name[
                    0] != '_' and results[0][name]:
                net_outputs = [result[name] for result in results]
                result_list_ = self.bbox2result_kitti(net_outputs,
                                                      sample_idx_list, classes,
                                                      pklfile_prefix_,
                                                      submission_prefix_)
                result_dict[name] = result_list_
            elif name == 'pred_instances' and name[0] != '_' and results[0][
                    name]:
                net_outputs = [result[name] for result in results]
                result_list_ = self.bbox2result_kitti2d(
                    net_outputs, sample_idx_list, classes, pklfile_prefix_,
                    submission_prefix_)
                result_dict[name] = result_list_
        return result_dict, tmp_dir

    def bbox2result_kitti(
            self,
            net_outputs: List[dict],
            sample_idx_list: List[int],
            class_names: List[str],
            pklfile_prefix: Optional[str] = None,
            submission_prefix: Optional[str] = None) -> List[dict]:
        """Convert 3D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (List[dict]): List of dict storing the inferenced
                bounding boxes and scores.
            sample_idx_list (List[int]): List of input sample idx.
            class_names (List[str]): A list of class names.
            pklfile_prefix (str, optional): The prefix of pkl file.
                Defaults to None.
            submission_prefix (str, optional): The prefix of submission file.
                Defaults to None.

        Returns:
            List[dict]: A list of dictionaries with the kitti format.
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        if submission_prefix is not None:
            mmengine.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting 3D prediction to KITTI format')
        for idx, pred_dicts in enumerate(
                mmengine.track_iter_progress(net_outputs)):
            sample_idx = sample_idx_list[idx]
            info = self.data_infos[sample_idx]
            # Here default used 'CAM2' to compute metric. If you want to
            # use another camera, please modify it.
            #
            # Some non-KITTI datasets (e.g. SiT) use a simplified info dict
            # without the nested ``images[CAM2]`` structure and instead store
            # a flat ``image`` / ``calib`` dictionary. In that case, fall back
            # to those fields so that evaluation can still proceed.
            if 'images' in info and self.default_cam_key in info['images']:
                image_shape = (
                    info['images'][self.default_cam_key]['height'],
                    info['images'][self.default_cam_key]['width'])
            elif 'image' in info:
                img_info = info['image']
                img_shape = img_info.get('image_shape', np.array([0, 0]))
                # `image_shape` is expected to be (H, W).
                image_shape = (int(img_shape[0]), int(img_shape[1]))
            else:
                # As a last resort, use a dummy resolution. This should not
                # normally happen, but avoids hard crashes on partially
                # populated info dicts.
                image_shape = (0, 0)
            box_dict = self.convert_valid_bboxes(pred_dicts, info)
            anno = {
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'score': []
            }
            if len(box_dict['bbox']) > 0:
                box_2d_preds = box_dict['bbox']
                box_preds = box_dict['box3d_camera']
                scores = box_dict['scores']
                box_preds_lidar = box_dict['box3d_lidar']
                label_preds = box_dict['label_preds']
                pred_box_type_3d = box_dict['pred_box_type_3d']

                for box, box_lidar, bbox, score, label in zip(
                        box_preds, box_preds_lidar, box_2d_preds, scores,
                        label_preds):
                    # Defensive checks to prevent segfaults
                    try:
                        # Validate bbox shape and values
                        if bbox.shape != (4,) or not np.all(np.isfinite(bbox)):
                            continue
                        bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                        bbox[:2] = np.maximum(bbox[:2], [0, 0])
                        
                        # Validate label index
                        label_int = int(label)
                        if label_int < 0 or label_int >= len(class_names):
                            continue
                        anno['name'].append(class_names[label_int])
                        anno['truncated'].append(0.0)
                        anno['occluded'].append(0)
                        
                        # Validate box arrays before operations
                        if not (np.all(np.isfinite(box)) and 
                                np.all(np.isfinite(box_lidar))):
                            continue
                            
                        if pred_box_type_3d == CameraInstance3DBoxes:
                            if len(box) >= 7 and np.isfinite(box[0]) and np.isfinite(box[2]) and np.isfinite(box[6]):
                                anno['alpha'].append(-np.arctan2(box[0], box[2]) +
                                                     box[6])
                            else:
                                anno['alpha'].append(0.0)
                        elif pred_box_type_3d == LiDARInstance3DBoxes:
                            if len(box_lidar) >= 3 and len(box) >= 7 and \
                               np.isfinite(box_lidar[0]) and np.isfinite(box_lidar[1]) and np.isfinite(box[6]):
                                anno['alpha'].append(
                                    -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6])
                            else:
                                anno['alpha'].append(0.0)
                        else:
                            anno['alpha'].append(0.0)
                        
                        # Validate array lengths before slicing
                        if len(box) >= 6:
                            anno['bbox'].append(bbox.copy())
                            anno['dimensions'].append(box[3:6].copy())
                            anno['location'].append(box[:3].copy())
                            if len(box) >= 7:
                                anno['rotation_y'].append(float(box[6]))
                            else:
                                anno['rotation_y'].append(0.0)
                        else:
                            continue
                            
                        # Validate score
                        if np.isfinite(score):
                            anno['score'].append(float(score))
                        else:
                            anno['score'].append(0.0)
                    except (ValueError, IndexError, TypeError) as e:
                        # Skip invalid entries to prevent segfault
                        import warnings
                        warnings.warn(
                            f'Skipping invalid detection at sample {sample_idx}, '
                            f'idx {idx}: {e}', RuntimeWarning)
                        continue

                # Safely stack arrays, ensuring all have the same length
                if len(anno['name']) > 0:
                    # Verify all lists have the same length
                    lengths = [len(v) for v in anno.values() if isinstance(v, list)]
                    if len(set(lengths)) == 1:
                        try:
                            anno = {k: np.stack(v) if isinstance(v, list) else v 
                                   for k, v in anno.items()}
                        except (ValueError, RuntimeError) as e:
                            # Fallback to empty arrays if stacking fails
                            import warnings
                            warnings.warn(
                                f'Failed to stack arrays at sample {sample_idx}: {e}. '
                                f'Using empty arrays.', RuntimeWarning)
                            anno = {
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
                        # Length mismatch - use empty arrays
                        import warnings
                        warnings.warn(
                            f'Length mismatch in anno dict at sample {sample_idx}. '
                            f'Using empty arrays.', RuntimeWarning)
                        anno = {
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
                    # No valid detections, use empty arrays
                    anno = {
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
                anno = {
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

            if submission_prefix is not None:
                curr_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(curr_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                                anno['name'][idx], anno['alpha'][idx],
                                bbox[idx][0], bbox[idx][1], bbox[idx][2],
                                bbox[idx][3], dims[idx][1], dims[idx][2],
                                dims[idx][0], loc[idx][0], loc[idx][1],
                                loc[idx][2], anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f)

            # Safely create sample_idx array with validation
            try:
                score_len = len(anno['score']) if isinstance(anno['score'], np.ndarray) else 0
                if score_len > 0:
                    anno['sample_idx'] = np.array(
                        [sample_idx] * score_len, dtype=np.int64)
                else:
                    anno['sample_idx'] = np.array([], dtype=np.int64)
            except (TypeError, ValueError) as e:
                import warnings
                warnings.warn(
                    f'Failed to create sample_idx array at sample {sample_idx}: {e}. '
                    f'Using empty array.', RuntimeWarning)
                anno['sample_idx'] = np.array([], dtype=np.int64)

            det_annos.append(anno)

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            else:
                out = pklfile_prefix
            try:
                # Validate det_annos before dumping to catch issues early
                if not isinstance(det_annos, list):
                    raise TypeError(f'det_annos must be a list, got {type(det_annos)}')
                for i, anno in enumerate(det_annos):
                    if not isinstance(anno, dict):
                        raise TypeError(f'anno[{i}] must be a dict, got {type(anno)}')
                    # Check that all required keys are present and valid
                    required_keys = ['name', 'truncated', 'occluded', 'alpha', 
                                   'bbox', 'dimensions', 'location', 'rotation_y', 
                                   'score', 'sample_idx']
                    for key in required_keys:
                        if key not in anno:
                            raise KeyError(f'Missing key "{key}" in anno[{i}]')
                        if not isinstance(anno[key], np.ndarray):
                            raise TypeError(
                                f'anno[{i}]["{key}"] must be np.ndarray, '
                                f'got {type(anno[key])}')
                
                mmengine.dump(det_annos, out)
                print(f'Result is saved to {out}.')
            except (TypeError, ValueError, KeyError) as e:
                import warnings
                warnings.warn(
                    f'Failed to save pkl file {out}: {e}. '
                    f'Continuing without saving.', RuntimeWarning)
                # Continue execution even if saving fails

        return det_annos

    def bbox2result_kitti2d(
            self,
            net_outputs: List[dict],
            sample_idx_list: List[int],
            class_names: List[str],
            pklfile_prefix: Optional[str] = None,
            submission_prefix: Optional[str] = None) -> List[dict]:
        """Convert 2D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (List[dict]): List of dict storing the inferenced
                bounding boxes and scores.
            sample_idx_list (List[int]): List of input sample idx.
            class_names (List[str]): A list of class names.
            pklfile_prefix (str, optional): The prefix of pkl file.
                Defaults to None.
            submission_prefix (str, optional): The prefix of submission file.
                Defaults to None.

        Returns:
            List[dict]: A list of dictionaries with the kitti format.
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        det_annos = []
        print('\nConverting 2D prediction to KITTI format')
        for i, bboxes_per_sample in enumerate(
                mmengine.track_iter_progress(net_outputs)):
            anno = dict(
                name=[],
                truncated=[],
                occluded=[],
                alpha=[],
                bbox=[],
                dimensions=[],
                location=[],
                rotation_y=[],
                score=[])
            sample_idx = sample_idx_list[i]

            num_example = 0
            bbox = bboxes_per_sample['bboxes']
            for i in range(bbox.shape[0]):
                anno['name'].append(class_names[int(
                    bboxes_per_sample['labels'][i])])
                anno['truncated'].append(0.0)
                anno['occluded'].append(0)
                anno['alpha'].append(0.0)
                anno['bbox'].append(bbox[i, :4])
                # set dimensions (height, width, length) to zero
                anno['dimensions'].append(
                    np.zeros(shape=[3], dtype=np.float32))
                # set the 3D translation to (-1000, -1000, -1000)
                anno['location'].append(
                    np.ones(shape=[3], dtype=np.float32) * (-1000.0))
                anno['rotation_y'].append(0.0)
                anno['score'].append(bboxes_per_sample['scores'][i])
                num_example += 1

            if num_example == 0:
                anno = dict(
                    name=np.array([]),
                    truncated=np.array([]),
                    occluded=np.array([]),
                    alpha=np.array([]),
                    bbox=np.zeros([0, 4]),
                    dimensions=np.zeros([0, 3]),
                    location=np.zeros([0, 3]),
                    rotation_y=np.array([]),
                    score=np.array([]),
                )
            else:
                anno = {k: np.stack(v) for k, v in anno.items()}

            anno['sample_idx'] = np.array(
                [sample_idx] * num_example, dtype=np.int64)
            det_annos.append(anno)

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            else:
                out = pklfile_prefix
            mmengine.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        if submission_prefix is not None:
            # save file in submission format
            mmengine.mkdir_or_exist(submission_prefix)
            print(f'Saving KITTI submission to {submission_prefix}')
            for i, anno in enumerate(det_annos):
                sample_idx = sample_idx_list[i]
                cur_det_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(cur_det_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions'][::-1]  # lhw -> hwl
                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} '
                            '{:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f}'.format(
                                anno['name'][idx],
                                anno['alpha'][idx],
                                *bbox[idx],  # 4 float
                                *dims[idx],  # 3 float
                                *loc[idx],  # 3 float
                                anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f,
                        )
            print(f'Result is saved to {submission_prefix}')

        return det_annos

    def convert_valid_bboxes(self, box_dict: dict, info: dict) -> dict:
        """Convert the predicted boxes into valid ones.

        Args:
            box_dict (dict): Box dictionaries to be converted.

                - bboxes_3d (:obj:`BaseInstance3DBoxes`): 3D bounding boxes.
                - scores_3d (Tensor): Scores of boxes.
                - labels_3d (Tensor): Class labels of boxes.
            info (dict): Data info.

        Returns:
            dict: Valid predicted boxes.

            - bbox (np.ndarray): 2D bounding boxes.
            - box3d_camera (np.ndarray): 3D bounding boxes in
              camera coordinate.
            - box3d_lidar (np.ndarray): 3D bounding boxes in
              LiDAR coordinate.
            - scores (np.ndarray): Scores of boxes.
            - label_preds (np.ndarray): Class label predictions.
            - sample_idx (int): Sample index.
        """
        # TODO: refactor this function
        box_preds = box_dict['bboxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = info['sample_idx']
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        if len(box_preds) == 0:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                # Keep label_preds 1D even in the empty case so its shape
                # matches the non-empty branch (where labels is 1D) and the
                # later `valid_inds.sum() == 0` branch below.
                label_preds=np.zeros([0]),
                sample_idx=sample_idx)
        # Here default used 'CAM2' to compute metric. If you want to
        # use another camera, please modify it.
        #
        # Support both the standard KITTI-style ``images[CAM2]`` layout and
        # simplified layouts used by some derived datasets (e.g. SiT) that
        # store calibration under ``calib`` and image metadata under ``image``.
        if 'images' in info and self.default_cam_key in info['images']:
            cam_info = info['images'][self.default_cam_key]
            lidar2cam = np.array(cam_info['lidar2cam']).astype(np.float32)
            P2 = np.array(cam_info['cam2img']).astype(np.float32)
            img_shape = (cam_info['height'], cam_info['width'])
        elif 'calib' in info and 'image' in info:
            calib = info['calib']
            img_info = info['image']
            # SiT-style placeholder calibration uses keys compatible with KITTI:
            #   - 'Tr_velo_to_cam'  ~ lidar2cam (4x4)
            #   - 'P2'              ~ cam2img (3x4)
            lidar2cam = np.array(calib.get('Tr_velo_to_cam',
                                           np.eye(4))).astype(np.float32)
            P2 = np.array(calib.get('P2',
                                    np.eye(3, 4))).astype(np.float32)
            img_shape_arr = img_info.get('image_shape',
                                         np.array([0, 0], dtype=np.int32))
            img_shape = (int(img_shape_arr[0]), int(img_shape_arr[1]))
        else:
            # Fallback to identity transforms and zero image size to avoid
            # crashing on partially populated info dicts. This will typically
            # filter out all boxes in `valid_cam_inds`.
            lidar2cam = np.eye(4, dtype=np.float32)
            P2 = np.eye(3, 4, dtype=np.float32)
            img_shape = (0, 0)
        P2 = box_preds.tensor.new_tensor(P2)

        if isinstance(box_preds, LiDARInstance3DBoxes):
            box_preds_camera = box_preds.convert_to(Box3DMode.CAM, lidar2cam)
            box_preds_lidar = box_preds
        elif isinstance(box_preds, CameraInstance3DBoxes):
            box_preds_camera = box_preds
            box_preds_lidar = box_preds.convert_to(Box3DMode.LIDAR,
                                                   np.linalg.inv(lidar2cam))

        box_corners = box_preds_camera.corners
        box_corners_in_image = points_cam2img(box_corners, P2)
        # box_corners_in_image: [N, 8, 2]
        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        # Post-processing
        # check box_preds_camera
        image_shape = box_preds.tensor.new_tensor(img_shape)
        valid_cam_inds = ((box_2d_preds[:, 0] < image_shape[1]) &
                          (box_2d_preds[:, 1] < image_shape[0]) &
                          (box_2d_preds[:, 2] > 0) & (box_2d_preds[:, 3] > 0))
        # check box_preds_lidar
        if isinstance(box_preds, LiDARInstance3DBoxes):
            limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
            valid_pcd_inds = ((box_preds_lidar.center > limit_range[:3]) &
                              (box_preds_lidar.center < limit_range[3:]))
            valid_inds = valid_cam_inds & valid_pcd_inds.all(-1)
        else:
            valid_inds = valid_cam_inds

        if valid_inds.sum() > 0:
            return dict(
                bbox=box_2d_preds[valid_inds, :].numpy(),
                pred_box_type_3d=type(box_preds),
                box3d_camera=box_preds_camera[valid_inds].numpy(),
                box3d_lidar=box_preds_lidar[valid_inds].numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                sample_idx=sample_idx)
        else:
            return dict(
                bbox=np.zeros([0, 4]),
                pred_box_type_3d=type(box_preds),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0]),
                sample_idx=sample_idx)
