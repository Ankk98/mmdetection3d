# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Optional, Union

import os.path as osp
import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes
from .kitti_dataset import KittiDataset


@DATASETS.register_module()
class SiTDataset(KittiDataset):
    r"""SiT Dataset.

    This class serves as the API for experiments on the `SiT Dataset
    <https://spalaboratory.github.io/SiT/>`_.

    The SiT dataset is a social navigation dataset published at NeurIPS 2023,
    focusing on pedestrian detection and tracking in crowded environments from
    a robot's perspective.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_lidar=True).
        default_cam_key (str): The default camera name adopted.
            Defaults to 'CAM2'.
        load_type (str): Type of loading mode. Defaults to 'frame_based'.
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset.
        filter_empty_gt (bool): Whether to filter the data with empty GT.
            If it's set to be True, the example with empty annotations after
            data pipeline will be dropped and a random example will be chosen
            in `__getitem__`. Defaults to True.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (List[float]): The range of point cloud used to filter
            invalid predicted boxes. Defaults to [-50, -50, -5, 50, 50, 3].
    """

    # SiT dataset classes - Pedestrian and Car (Pedestrain_sitting mapped to Pedestrian)
    METAINFO = {
        'classes': ('Pedestrian', 'Car'),
        'palette': [(106, 0, 228), (165, 42, 42)]  # Colors for visualization
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: Optional[List[Union[dict, Callable]]] = None,
                 modality: Optional[dict] = None,
                 default_cam_key: str = 'CAM2',
                 load_type: str = 'frame_based',
                 box_type_3d: str = 'LiDAR',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 pcd_limit_range: Optional[List[float]] = None,
                 **kwargs) -> None:

        # Avoid mutable default arguments by creating fresh instances here.
        if pipeline is None:
            pipeline = []
        if modality is None:
            modality = dict(use_lidar=True)
        if pcd_limit_range is None:
            pcd_limit_range = [-50, -50, -5, 50, 50, 3]

        # Call parent constructor with adjusted parameters
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            default_cam_key=default_cam_key,
            load_type=load_type,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            pcd_limit_range=pcd_limit_range,
            **kwargs)

        # SiT-specific attributes if needed
        self.sit_classes = ['Pedestrian', 'Car']

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        For SiT we follow the standard Det3DDataset path handling
        (data_root + data_prefix + paths stored in the info file),
        but early iterations of the converter stored a full
        ``'training/velodyne/xxx.bin'`` path inside the info file while
        the dataset's ``data_prefix['pts']`` was also set to
        ``'training/velodyne'``.  When combined in
        :meth:`Det3DDataset.parse_data_info`, this produced duplicated
        segments such as:

            ``data/sit/training/velodyne/training/velodyne/000001.bin``

        which in turn caused ``FileNotFoundError`` during GT database
        creation and training.

        To keep the info format backwards-compatible while avoiding I/O
        errors, we detect and collapse this specific duplication pattern
        after the base implementation has done its work.
        """
        info = super().parse_data_info(info)

        # Normalize any duplicated "training/velodyne" segments that may
        # arise when both the info dict and `data_prefix['pts']` already
        # contain this sub-path (e.g.
        # "data/sit/training/velodyne/training/velodyne/000001.bin").
        if self.modality.get('use_lidar', False) and 'lidar_points' in info:
            lidar_path = info['lidar_points'].get('lidar_path', '')
            dup_segment = osp.join('training', 'velodyne', 'training',
                                   'velodyne')
            if dup_segment in lidar_path:
                normalized = lidar_path.replace(
                    dup_segment, osp.join('training', 'velodyne'))
                info['lidar_points']['lidar_path'] = normalized
                # Keep the convenience mirror field in sync as well.
                info['lidar_path'] = normalized

        return info

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        For SiT dataset, we use LiDAR-only processing without camera data.
        We convert numpy arrays to :class:`LiDARInstance3DBoxes` objects.

        Compared to the default KITTI flow, we still need to:

        - use the base :class:`Det3DDataset` logic to build numpy arrays and
          apply ``label_mapping``; and
        - run ``_remove_dontcare`` so that any instances mapped to label ``-1``
          (e.g. unknown classes from the converter) are filtered out.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information with ``gt_bboxes_3d`` as
            :class:`LiDARInstance3DBoxes`.
        """
        # Use the generic Det3D implementation to build ann_info with numpy
        # arrays and label mapping applied. We cannot call super().parse_ann_info
        # here because KittiDataset.parse_ann_info assumes camera geometry
        # (e.g. info['images']['CAM2']['lidar2cam']), which SiT does not have.
        from mmdet3d.datasets.det3d_dataset import Det3DDataset
        ann_info = Det3DDataset.parse_ann_info(self, info)

        if ann_info is None:
            # Empty instance: mirror the parent contract by returning a complete
            # `ann_info` dict that still contains the `instances` key.
            ann_info = dict()
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)
            # For empty GT, `instances` should be an empty list rather than
            # missing entirely to stay consistent with `Det3DDataset`.
            ann_info['instances'] = info.get('instances', [])
        else:
            # Filter out "dontcare"/unknown categories where labels are -1,
            # matching the behavior in `KittiDataset.parse_ann_info`.
            ann_info = self._remove_dontcare(ann_info)

        # Convert numpy array to LiDARInstance3DBoxes for LiDAR-only dataset.
        # SiT uses LiDAR coordinates directly, so no camera-to-lidar transform
        # is required here.
        gt_bboxes_3d = LiDARInstance3DBoxes(
            ann_info['gt_bboxes_3d'],
            box_dim=ann_info['gt_bboxes_3d'].shape[-1]).convert_to(
                self.box_mode_3d)

        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info

    def _get_metainfo(self) -> dict:
        """Get meta information of dataset.

        Returns:
            dict: Meta information of dataset.
        """
        return self.METAINFO
