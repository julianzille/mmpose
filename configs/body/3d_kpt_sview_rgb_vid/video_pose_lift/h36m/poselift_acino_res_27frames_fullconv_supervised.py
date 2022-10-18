_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/acino3d.py'
]


total_epochs=400
evaluation = dict(interval=10, metric=['mpjpe', 'p-mpjpe'], save_best='MPJPE')

work_dir='work_dirs/acino_res_poselift_27frames_epoch400'
gpu_ids=range(1)
workflow=[('train',1)]
resume_from=None

# optimizer settings
optimizer = dict(
    type='Adam',
    lr=1e-4,
)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='exp',
    by_epoch=True,
    gamma=0.995,
)

log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=20,
    dataset_joints=20,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,18,19],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17,18,19
    ])

# model settings
model = dict(
    type='PoseLifter',
    pretrained=None,
    backbone=dict(
        type='TCN',
        in_channels=2 * 20,
        stem_channels=1024,
        num_blocks=2,
        kernel_sizes=(3, 3, 3),
        causal=False,
        dropout=0.25,
        use_stride_conv=True),
    keypoint_head=dict(
        type='TemporalRegressionHead',
        in_channels=1024,
        num_joints=20,
        loss_keypoint=dict(type='MPJPELoss')),
    train_cfg=dict(),
    test_cfg=dict(restore_global_position=True))

# data settings
data_root = 'data/acino_3d'
data_cfg = dict(
    num_joints=20,
    seq_len=27,
    seq_frame_interval=1,
    causal=False,
    temporal_padding=True,
    joint_2d_src='gt',
    need_camera_param=True,
    camera_param_file=f'{data_root}/annotations/cameras.pkl',
)

train_pipeline = [
    dict(
        type='GetRootCenteredPose',
        item='target',
        visible_item='target_visible',
        root_index=4,
        root_name='root_position',
        remove_root=False),
    dict(type='ImageCoordinateNormalization', item='input_2d'),
    dict(
        type='RelativeJointRandomFlip',
        item=['input_2d', 'target'],
        flip_cfg=[
            dict(center_mode='static', center_x=0.),
            dict(center_mode='root', center_index=4)
        ],
        visible_item=['input_2d_visible', 'target_visible'],
        flip_prob=0.5),
    dict(type='PoseSequenceToTensor', item='input_2d'),
    dict(
        type='Collect',
        keys=[('input_2d', 'input'), 'target'],
        meta_name='metas',
        meta_keys=['target_image_path', 'flip_pairs', 'root_position'])
]

val_pipeline = [
    dict(
        type='GetRootCenteredPose',
        item='target',
        visible_item='target_visible',
        root_index=4,
        root_name='root_position',
        remove_root=False),
    dict(type='ImageCoordinateNormalization', item='input_2d'),
    dict(type='PoseSequenceToTensor', item='input_2d'),
    dict(
        type='Collect',
        keys=[('input_2d', 'input'), 'target'],
        meta_name='metas',
        meta_keys=['target_image_path', 'flip_pairs', 'root_position'])
]

test_pipeline = val_pipeline

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=128),
    test_dataloader=dict(samples_per_gpu=128),
    train=dict(
        type='Body3DAcinoDataset',
        ann_file=f'{data_root}/annotations/Res50/acino3d_train.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='Body3DAcinoDataset',
        ann_file=f'{data_root}/annotations/Res50/acino3d_test.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='Body3DAcinoDataset',
        ann_file=f'{data_root}/annotations/Res50/acino3d_test.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
