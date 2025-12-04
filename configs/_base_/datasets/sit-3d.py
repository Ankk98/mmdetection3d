# dataset settings
dataset_type = 'SiTDataset'
# Normalized SiT layout (mirrors KITTI):
#   data/sit/
#     ├── training/
#     │   ├── velodyne/
#     │   ├── label_2/
#     │   ├── calib/
#     │   └── image_2/
#     ├── sit_infos_*.pkl
#     └── sit_dbinfos_train.pkl
data_root = 'data/sit/'
class_names = ['Pedestrian', 'Car']
point_cloud_range = [-50, -50, -5, 50, 50, 3]  # Based on SiT data analysis
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)
backend_args = None

# Data augmentation database
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'sit_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Pedestrian=5, Car=5)),
    classes=class_names,
    sample_groups=dict(Pedestrian=15, Car=15),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    backend_args=backend_args)

# Training pipeline
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816]),
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
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

# Testing pipeline
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
                type='PointsRangeFilter',
                point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

# Evaluation pipeline
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='Pack3DDetInputs', keys=['points'])
]

# Data loaders
train_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='sit_infos_train.pkl',
        data_prefix=dict(pts='training/velodyne'),
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
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
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne'),
        ann_file='sit_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
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
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne'),
        ann_file='sit_infos_test.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))

# Evaluators
val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'sit_infos_val.pkl',
    metric='bbox',
    # SiT uses LiDAR-only placeholders for camera geometry. Running the
    # compiled KITTI evaluation on such data can lead to low-level
    # segmentation faults. Enable format_only to skip the native eval
    # while still dumping results in KITTI format if needed.
    format_only=True,
    submission_prefix=data_root + 'kitti_val_pred',
    backend_args=backend_args)
test_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'sit_infos_test.pkl',
    metric='bbox',
    format_only=True,
    submission_prefix=data_root + 'kitti_test_pred',
    backend_args=backend_args)

# Visualization configuration so demos/inferencers can create a visualizer.
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
