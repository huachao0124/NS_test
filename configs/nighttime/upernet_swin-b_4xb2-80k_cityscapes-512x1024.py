_base_ = [
    '../_base_/models/upernet_swin.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_80k.py',
    '../_base_/datasets/cityscapes.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
checkpoint_file = 'ckpts/swin_base_patch4_window7_224_22k_20220317-4f79f7c0.pth'
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=checkpoint_file,
    backbone=dict(embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32]),
    decode_head=dict(
        in_channels=[128, 256, 512, 1024]),
    auxiliary_head=dict(
        in_channels=512))
