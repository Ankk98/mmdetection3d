_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_kitti.py',
    '../_base_/datasets/sit-3d.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]

point_cloud_range = [-50, -50, -5, 50, 50, 3]  # SiT point cloud range
# dataset settings
# SiT uses normalized layout: data/sit/training/velodyne, consistent with sit-3d base config.
data_root = 'data/sit/'
class_names = ['Pedestrian', 'Car']
metainfo = dict(classes=class_names)
backend_args = None

# PointPillars adopted a different sampling strategy among classes
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'sit_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Pedestrian=5, Car=3)),
    classes=class_names,
    # Oversample Car 4x relative to Pedestrian to counter class imbalance
    sample_groups=dict(Pedestrian=15, Car=60),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    backend_args=backend_args)

# PointPillars uses different augmentation hyper parameters
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler, use_ground_plane=False),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_labels_3d', 'gt_bboxes_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

# Validation pipeline: similar to test_pipeline but loads annotations for loss computation
val_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

train_dataloader = dict(
    batch_size=8,  # Increased from 6 to better utilize 12GB VRAM
    num_workers=4,
    persistent_workers=True,
    dataset=dict(pipeline=train_pipeline, metainfo=metainfo))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline, metainfo=metainfo))
# Set test_mode=False for validation to enable annotation loading for loss computation
# Evaluation will still work correctly as the evaluator uses the annotation file directly
val_dataloader = dict(
    dataset=dict(
        pipeline=val_pipeline,
        metainfo=metainfo,
        test_mode=False))  # Set to False to load ann_info for loss computation

# Model settings for SiT dataset
# Calculate output shape based on point cloud range and voxel size
# Range: [-50, -50, -5, 50, 50, 3], Voxel size: 0.16
# X: (50 - (-50)) / 0.16 = 625, Y: (50 - (-50)) / 0.16 = 625
model = dict(
    data_preprocessor=dict(
        voxel_layer=dict(point_cloud_range=point_cloud_range)),
    voxel_encoder=dict(point_cloud_range=point_cloud_range),
    middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        # Canvas size must be compatible with the backbone/neck strides.
        # The voxel grid along X/Y is 625 (= 100 / 0.16), which is not
        # divisible by 8 (2 * 2 * 2 backbone strides). This causes the
        # three FPN paths to upsample to slightly different spatial sizes
        # (e.g. 313 vs 314) and breaks the torch.cat in SECONDFPN.
        #
        # Use the next multiple of 8 that is >= 625, so all paths align
        # while still covering the full voxelized area (extra rows/cols
        # stay empty as there are no voxels there).
        output_shape=[632, 632]),
    bbox_head=dict(
        num_classes=2,  # Pedestrian, Car
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            # Weight Car higher (index 1) to offset class imbalance
            class_weight=[1.0, 4.0],
            loss_weight=1.0),
        anchor_generator=dict(
            ranges=[
                [-50, -50, -0.6, 50, 50, -0.6],  # Pedestrian range
                [-50, -50, -0.6, 50, 50, -0.6],   # Car range
            ],
            sizes=[
                [0.8, 0.6, 1.73],  # Pedestrian (adjusted for SiT)
                [4.5, 2.03, 1.74],  # Car (width, length, height from SiT stats)
            ],
        ),
    ),
    train_cfg=dict(
        assigner=[
            dict(  # for Pedestrian
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Car
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1),
        ],
    ),
)

# In practice PointPillars also uses a different schedule
# optimizer
lr = 0.0001
epoch_num = 80
optim_wrapper = dict(
    optimizer=dict(lr=lr), clip_grad=dict(max_norm=35, norm_type=2))
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.4,
        eta_min=lr * 10,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.6,
        eta_min=lr * 1e-4,
        begin=epoch_num * 0.4,
        end=epoch_num * 1,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.4,
        eta_min=0.85 / 0.95,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.6,
        eta_min=1,
        begin=epoch_num * 0.4,
        end=epoch_num * 1,
        by_epoch=True,
        convert_to_iter_based=True)
]
# max_norm=35 is slightly better than 10 for PointPillars in the earlier
# development of the codebase thus we keep the setting. But we does not
# specifically tune this parameter.
# PointPillars usually need longer schedule than second, we simply double
# the training schedule. Do remind that since we use RepeatDataset and
# repeat factor is 2, so we actually train 160 epochs.
train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=2)
val_cfg = dict()
test_cfg = dict()

# Override val_evaluator to use SitMetric with LiDAR 3D IoU evaluation
# SitMetric uses LiDARInstance3DBoxes.overlaps() to compute 3D IoU directly
# in LiDAR coordinate space, avoiding camera-based evaluation that causes segfaults
val_evaluator = dict(
    type='SitMetric',
    ann_file=data_root + 'sit_infos_val.pkl',
    # CRITICAL: Set correct point cloud range for SiT (not KITTI default)
    # KITTI default is [0, -40, -3, 70.4, 40, 0.0], but SiT uses [-50, -50, -5, 50, 50, 3]
    pcd_limit_range=[-50, -50, -5, 50, 50, 3],
    # IoU thresholds for AP calculation (COCO-style: 0.5:0.05:0.95)
    iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    format_only=False,  # Enable mAP computation
    # Save predictions to permanent location in work_dirs
    # File will be saved as: work_dirs/predictions/val_results/pred_instances_3d.pkl
    pklfile_prefix='work_dirs/predictions/val_results',
    backend_args=backend_args)

# Enable checkpoint saving (default_runtime has interval=-1 which disables it)
# Now that we have metrics, we can save "best" checkpoint based on mAP
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=2,  # Save checkpoint every 2 epochs
        max_keep_ckpts=5,  # Keep the latest 5 checkpoints (saves disk space)
        save_optimizer=True,  # Also save optimizer state for resuming
        by_epoch=True,  # Save by epoch (not iteration)
        # Note: SitMetric uses LiDAR 3D IoU evaluation
        # The metric key format is: '{prefix}/pred_instances_3d/{metric_name}'
        # where prefix='Sit metric' and metric_name is like 'Overall_mAP' or 'mAP@0.5:0.95'
        # If this doesn't work, check the logs after first evaluation for the exact metric name
        save_best='Sit metric/pred_instances_3d/Overall_mAP',  # Save best based on overall mAP
        rule='greater'  # Higher mAP is better
    ),
    # Add validation loss hook to compute and log validation loss
    val_loss=dict(
        type='ValLossHook',
        interval=1  # Compute loss for every validation batch
    )
)

# Enable TensorBoard visualization backend for real-time monitoring
# This allows viewing training metrics, loss curves, and mAP in TensorBoard
vis_backends = [
    dict(type='LocalVisBackend'),  # Keep local logging
    dict(type='TensorboardVisBackend')  # Add TensorBoard logging
]
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

