_base_ = '../pspnet/pspnet_r50-d8_4xb2-80k_cityscapes-512x1024.py'
model = dict(pretrained='ckpts/resnet101_v1c-e67eebb6.pth', backbone=dict(depth=101))

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
