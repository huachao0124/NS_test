_base_ = ['../segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py']

checkpoint = 'ckpts/mit_b5_20220624-658746d9.pth'  # noqa

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))

# dataset settings
train_data_root = 'data/nightcity-fine/'
test_data_root = 'data/nightcity-fine/'
train_dataloader = dict(
    dataset=dict(
        data_root=train_data_root,
        data_prefix=dict(
            img_path='train/img', seg_map_path='train/lbl'),
        img_suffix='.png',
        seg_map_suffix='_trainIds.png'))
val_dataloader = dict(
    dataset=dict(
        data_root=test_data_root,
        data_prefix=dict(
            img_path='val/img', seg_map_path='val/lbl'),
        img_suffix='.png',
        seg_map_suffix='_trainIds.png'))
test_dataloader = val_dataloader

default_hooks = dict(visualization=dict(type='SegVisualizationHook', draw=True, interval=20))
