_base_ = ['../mask2former/mask2former_swin-t_8xb2-90k_cityscapes-512x1024.py']
pretrained = 'ckpts/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth'  # noqa

depths = [2, 2, 18, 2]
model = dict(
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=128,
        depths=depths,
        num_heads=[4, 8, 16, 32],
        window_size=12,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    decode_head=dict(in_channels=[128, 256, 512, 1024]))

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
    dict(type='BitZero'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomChoiceResize',
        scales=[int(1024 * x * 0.1) for x in range(5, 21)],
        resize_type='ResizeShortestEdge',
        max_size=4096),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
train_dataloader = dict(num_workers=4,
                        dataset=dict(pipeline=train_pipeline))

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='BitZero'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ComposedVisualizer', vis_backends=vis_backends, name='visualizer')

default_hooks = dict(visualization=dict(type='SegVisualizationHook', draw=True, interval=5))
