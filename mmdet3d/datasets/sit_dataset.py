# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Union
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
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True),
                 default_cam_key: str = 'CAM2',
                 load_type: str = 'frame_based',
                 box_type_3d: str = 'LiDAR',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 pcd_limit_range: List[float] = [-50, -50, -5, 50, 50, 3],
                 **kwargs) -> None:

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

        Convert all relative path of needed modality data file to
        the absolute path by joining with data_root.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        # First call parent to join with data_prefix
        info = super().parse_data_info(info)

        # Then make paths absolute by joining with data_root
        # The parent method already joined with data_prefix (which is empty for SiT),
        # so the path is still relative like 'training/velodyne/25.bin'
        # We need to join with data_root and make it absolute
        if self.modality['use_lidar']:
            lidar_path = info['lidar_points']['lidar_path']
            
            # Get absolute data_root for comparison
            data_root_abs = osp.abspath(self.data_root)
            
            # If path is already absolute, verify it's correct
            if osp.isabs(lidar_path):
                # Check if it already contains data_root (to detect duplication)
                if lidar_path.startswith(data_root_abs):
                    # Path is absolute and contains data_root, use as-is
                    pass
                else:
                    # Absolute path doesn't contain data_root - might be from another source
                    # Use as-is but log a warning
                    pass
            else:
                # Path is relative - need to join with data_root
                # Check if path already contains data_root as a string (to avoid double-join)
                data_root_str = str(self.data_root).rstrip('/\\')
                if data_root_str in lidar_path and lidar_path.startswith(data_root_str):
                    # Path already contains data_root string, just make absolute
                    lidar_path = osp.abspath(lidar_path)
                else:
                    # Normal case: join with data_root and make absolute
                    lidar_path = osp.join(self.data_root, lidar_path)
                    lidar_path = osp.abspath(lidar_path)
            
            info['lidar_points']['lidar_path'] = lidar_path
            info['lidar_path'] = lidar_path

        return info

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        For SiT dataset, we use LiDAR-only processing without camera data.
        We convert numpy arrays to LiDARInstance3DBoxes objects.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information with gt_bboxes_3d as LiDARInstance3DBoxes.
        """
        # Call base Det3DDataset to get numpy arrays
        from mmdet3d.datasets.det3d_dataset import Det3DDataset
        ann_info = Det3DDataset.parse_ann_info(self, info)
        
        if ann_info is None:
            # Empty instance
            ann_info = dict()
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)
        
        # Convert numpy array to LiDARInstance3DBoxes for LiDAR-only dataset
        # SiT uses LiDAR coordinates directly, so no coordinate conversion needed
        gt_bboxes_3d = LiDARInstance3DBoxes(
            ann_info['gt_bboxes_3d'],
            box_dim=ann_info['gt_bboxes_3d'].shape[-1]).convert_to(self.box_mode_3d)
        
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info

    def _get_metainfo(self) -> dict:
        """Get meta information of dataset.

        Returns:
            dict: Meta information of dataset.
        """
        return self.METAINFO
