_base_ = '../pspnet/pspnet_r50-d8_4xb2-80k_cityscapes-512x1024.py'
model = dict(pretrained='ckpts/resnet101_v1c-e67eebb6.pth', backbone=dict(depth=101))

crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='AddNoisyImg', model='PGRU', camera='CanonEOS5D4',
         cfa='bayer', use_255=True, pre_adjust_brightness=False, mode='addnoise', dark_ratio=(1.0, 1.0), noise_ratio=(10, 100)), # here
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='HSVDarker'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
