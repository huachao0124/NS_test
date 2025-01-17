_base_ = [
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
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
        threshold=0.1
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
    )
)


max_epoch = 500
optim_wrapper = dict(
    _delete_=True,
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

# By default, models are trained on 8 GPUs with 2 images per GPU
# dataset settings
train_data_root = 'data/nightcity-fine/'
test_data_root = 'data/nightcity-fine/'

crop_size = (512, 1024)
# dataset config
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadLogits', seg_logits_path='mmseg/nighttime_utils/output_logits/upernet_convnext'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomChoiceResize',
        scales=[int(1024 * x * 0.1) for x in range(5, 21)],
        resize_type='ResizeShortestEdge',
        max_size=4096),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadLogits', seg_logits_path='mmseg/nighttime_utils/output_logits/upernet_convnext'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
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
        seg_map_suffix='_trainIds.png'))
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=test_data_root,
        data_prefix=dict(
            img_path='val/img', seg_map_path='val/lbl'),
        img_suffix='.png',
        seg_map_suffix='_trainIds.png'))
test_dataloader = val_dataloader
