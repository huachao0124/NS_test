_base_ = '../pspnet/pspnet_r50-d8_4xb2-80k_cityscapes-512x1024.py'
model = dict(pretrained='ckpts/resnet101_v1c-e67eebb6.pth', backbone=dict(depth=101))
