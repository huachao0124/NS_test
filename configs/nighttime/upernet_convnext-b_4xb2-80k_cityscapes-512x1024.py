_base_ = [
    '../_base_/models/upernet_convnext.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_80k.py',
    '../_base_/datasets/cityscapes.py'
]
custom_imports = dict(imports='mmpretrain.models', allow_failed_imports=False)
checkpoint_file = 'ckpts/convnext-base_3rdparty_in21k_20220301-262fd037.pth'  # noqa
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    auxiliary_head=dict(in_channels=512)
)
