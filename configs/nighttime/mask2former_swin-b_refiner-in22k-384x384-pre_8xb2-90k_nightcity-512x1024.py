_base_ = [
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    data_preprocessor=data_preprocessor,
    type='SparseRefiner',
    selector=dict(
        type='EntropySelector',
        threshold=0.05
    ),
    featurizer=dict(
        type='TorchSparseFeaturizer',
        features=["rgb"],
        is_half=False
    ),
    backbone=dict(
        type='TorchSparseUNet',
        in_channels=3,
        reps=[2, 2, 2, 2, 2, 2],
        nPlanes=[32, 64, 128, 256, 512, 1024]
    ),
    classifier=None,
    # classifier=dict(
    #     type='Linear',
    #     in_channels=32,
    #     out_channels=19
    # ),
    ensembler=dict(
        type='GatedEnsembler',
        num_classes=19
    ),
    loss_mask=dict(
        type='mmdet.CrossEntropyLoss',
        use_sigmoid=False,
        reduction='mean',
        ignore_index=255,
        loss_weight=1.0)
)

# By default, models are trained on 8 GPUs with 2 images per GPU
# dataset settings
train_data_root = 'data/nightcity-fine/'
test_data_root = 'data/nightcity-fine/'

crop_size = (512, 1024)
# dataset config
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadLogits', seg_logits_path='mmseg/nighttime_utils/output_logits/mask2former', data_path='data'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomChoiceResize',
        scales=[int(1024 * x * 0.1) for x in range(5, 21)],
        resize_type='ResizeShortestEdge',
        max_size=4096),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputsWithLogits')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadLogits', seg_logits_path='mmseg/nighttime_utils/output_logits/mask2former', data_path='data'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputsWithLogits')
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(
    batch_size=4,
    num_workers=16,
    dataset=dict(
        data_root=train_data_root,
        data_prefix=dict(
            img_path='train/img', seg_map_path='train/lbl'),
        img_suffix='.png',
        seg_map_suffix='_trainIds.png',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=test_data_root,
        data_prefix=dict(
            img_path='val/img', seg_map_path='val/lbl'),
        img_suffix='.png',
        seg_map_suffix='_trainIds.png',
        pipeline=test_pipeline))
test_dataloader = val_dataloader

max_epoch = 500

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0003, betas=(0.9, 0.999), weight_decay=0.05),
    constructor='LearningRateDecayOptimizerConstructor',
    loss_scale='dynamic')

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=max_epoch,
        by_epoch=True)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epoch, val_interval=max_epoch)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=50),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
