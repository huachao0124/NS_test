_base_ = '../upernet/upernet_r50_4xb2-80k_cityscapes-512x1024.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))

# dataset settings
train_data_root = 'data/nightcity-fine/'
test_data_root = 'data/nightcity-fine/'
train_dataloader = dict(
    batch_size=16,
    num_workers=16,
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
