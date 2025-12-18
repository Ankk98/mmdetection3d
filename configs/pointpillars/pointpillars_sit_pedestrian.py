_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_kitti.py',
    '../_base_/datasets/sit-3d.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]

# SiT Pedestrian-only config
# Based on official SiT implementation: https://github.com/SPALaboratory/SiT-Dataset
# Official results: PointPillars mAP = 0.319

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 5.0]  # Official SiT range (Z: -5 to 5)
voxel_size = [0.2, 0.2, 10]  # Official SiT voxel size (Z range is 10: -5 to 5)

# Dataset settings - PEDESTRIAN ONLY (matches official SiT)
data_root = 'data/sit/'
class_names = ['Pedestrian']  # Single class like official
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)
backend_args = None

# Database sampler for pedestrian only
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'sit_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Pedestrian=5)),
    classes=class_names,
    sample_groups=dict(Pedestrian=15),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    backend_args=backend_args)

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
        rot_range=[-0.3925, 0.3925],  # Official SiT rotation range
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
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
    batch_size=16,  # Official uses 16
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='SiTDataset',
        data_root=data_root,
        ann_file='sit_infos_train.pkl',
        data_prefix=dict(pts='training/velodyne'),
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='SiTDataset',
        data_root=data_root,
        ann_file='sit_infos_test.pkl',
        data_prefix=dict(pts='training/velodyne'),
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='SiTDataset',
        data_root=data_root,
        ann_file='sit_infos_val.pkl',
        data_prefix=dict(pts='training/velodyne'),
        pipeline=val_pipeline,
        modality=input_modality,
        test_mode=False,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))

# Model settings - Pedestrian only with official SiT anchor
# Output shape for voxel_size=[0.2, 0.2, 10] and range=[-51.2, 51.2]:
# (51.2 - (-51.2)) / 0.2 = 512
model = dict(
    data_preprocessor=dict(
        voxel_layer=dict(
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_num_points=20,  # Official
            max_voxels=(32000, 32000))),
    voxel_encoder=dict(
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size),
    middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=[512, 512]),  # Matches official
    bbox_head=dict(
        num_classes=1,  # Pedestrian only
        anchor_generator=dict(
            ranges=[[-51.2, -51.2, -5.0, 51.2, 51.2, 5.0]],  # Official range
            sizes=[[0.834, 0.765, 1.802]],  # Official pedestrian anchor (l, w, h)
            rotations=[0, 1.57],
            reshape_out=False),
        # Official SiT IoU thresholds
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    train_cfg=dict(
        assigner=[
            dict(  # for Pedestrian - official thresholds
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
        ],
    ),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr=0.2,
            score_thr=0.05,
            min_bbox_size=0,
            max_num=500)),
)

# Optimizer - Official SiT uses lr=1e-5 with AdamW
lr = 1e-5  # Official learning rate
epoch_num = 200  # Official trains 200 epochs

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 1000,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=epoch_num,
        by_epoch=True,
        milestones=[140, 160],
        gamma=0.1)
]

train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=10)
val_cfg = dict()
test_cfg = dict()

# Evaluator - Pedestrian only
val_evaluator = dict(
    type='SitMetric',
    ann_file=data_root + 'sit_infos_val.pkl',
    pcd_limit_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 5.0],
    iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],  # Standard COCO-style
    format_only=False,
    pklfile_prefix='work_dirs/predictions/val_results',
    backend_args=backend_args)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=5,
        save_optimizer=True,
        by_epoch=True,
        save_best='Sit metric/pred_instances_3d/Overall_mAP',
        rule='greater'
    ),
    # Add validation loss hook to compute and log validation loss
    val_loss=dict(
        type='ValLossHook',
        interval=1
    )
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

# No pretrained weights - train from scratch (official approach)
# Or uncomment to use nuScenes pretrained (better than KITTI for pedestrians):
# load_from = 'checkpoints/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus.pth'

