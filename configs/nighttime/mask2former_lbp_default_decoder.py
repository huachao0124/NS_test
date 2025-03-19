_base_ = ['../mask2former/mask2former_swin-t_8xb2-90k_cityscapes-512x1024.py']
pretrained = 'ckpts/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth'  # noqa

depths = [2, 2, 18, 2]
model = dict(
    type='EncoderDecoderLBP',
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=128,
        depths=depths,
        num_heads=[4, 8, 16, 32],
        window_size=12,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    backbone_lbp=dict(
        type='ResNet',
        in_channels=8,
        depth=18,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck_lbp=dict(
        type='mmdet.ChannelMapper',
        in_channels=[128, 256, 512],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        bias=True,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=3),
    decode_head=dict(
        type='Mask2FormerHeadLBP_D',
        in_channels=[128, 256, 512, 1024]))

# set all layers in backbone to lr_mult=0.1
# set all norm layers, position_embeding,
# query_embeding, level_embeding to decay_multi=0.0
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm': backbone_norm_multi,
    'backbone.norm': backbone_norm_multi,
    'absolute_pos_embed': backbone_embed_multi,
    'relative_position_bias_table': backbone_embed_multi,
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi
}
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})
# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0))

crop_size = (512, 1024)
# dataset config
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LBP', mode='default'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomChoiceResize',
        scales=[int(1024 * x * 0.1) for x in range(5, 21)],
        resize_type='ResizeShortestEdge',
        max_size=4096),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                                    'img_shape', 'pad_shape', 'scale_factor', 'flip',
                                    'flip_direction', 'reduce_zero_label', 'lbp'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LBP', mode='default'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                                    'img_shape', 'pad_shape', 'scale_factor', 'flip',
                                    'flip_direction', 'reduce_zero_label', 'lbp'))
]

# dataset settings
train_data_root = 'data/nightcity-fine/'
test_data_root = 'data/nightcity-fine/'
train_dataloader = dict(
    num_workers=16,
    dataset=dict(
        data_root=train_data_root,
        data_prefix=dict(
            img_path='train/img', seg_map_path='train/lbl'),
        img_suffix='.png',
        seg_map_suffix='_trainIds.png',
        pipeline=train_pipeline))
val_dataloader = dict(
    dataset=dict(
        data_root=test_data_root,
        data_prefix=dict(
            img_path='val/img', seg_map_path='val/lbl'),
        img_suffix='.png',
        seg_map_suffix='_trainIds.png',
        pipeline=test_pipeline))
test_dataloader = val_dataloader


vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ComposedVisualizer', vis_backends=vis_backends, name='visualizer')

default_hooks = dict(visualization=dict(type='SegVisualizationHook', draw=False, interval=5))
