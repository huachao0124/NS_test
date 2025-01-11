_base_ = '../hrnet/fcn_hr18_4xb2-160k_cityscapes-512x1024.py'
model = dict(
    pretrained='ckpts/hrnetv2_w48-d2186c55.pth',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384])))

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
