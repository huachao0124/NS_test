_base_ = [
    '../_base_/models/upernet_convnext.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
checkpoint_file = 'ckpts/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=19),
    auxiliary_head=dict(in_channels=512, num_classes=19),
    # test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
    test_cfg=dict(mode='whole')
)

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 12
    },
    constructor='LearningRateDecayOptimizerConstructor',
    loss_scale='dynamic')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]

data_root = 'data/cityscapes_night'
# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=4,
                        num_workers=16,
                        dataset=dict(data_root=data_root))
val_dataloader = dict(batch_size=1,
                      dataset=dict(data_root=data_root))
test_dataloader = val_dataloader
