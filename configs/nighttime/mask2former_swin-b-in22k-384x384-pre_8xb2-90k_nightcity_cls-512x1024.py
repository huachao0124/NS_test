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
    decode_head=dict(in_channels=[128, 256, 512, 1024],
        loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=10.0,
                reduction='mean',
                class_weight=[1.0] * 19 + [0.1]),
            loss_mask=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                reduction='mean',
                loss_weight=3.0),
            loss_dice=dict(
                type='mmdet.DiceLoss',
                use_sigmoid=True,
                activate=True,
                reduction='mean',
                naive_dice=True,
                eps=1.0,
                loss_weight=3.0),
            train_cfg=dict(
                num_points=12544,
                oversample_ratio=3.0,
                importance_sample_ratio=0.75,
                assigner=dict(
                    type='mmdet.HungarianAssigner',
                    match_costs=[
                        dict(type='mmdet.ClassificationCost', weight=10.0),
                        dict(
                            type='mmdet.CrossEntropyLossCost',
                            weight=3.0,
                            use_sigmoid=True),
                        dict(
                            type='mmdet.DiceCost',
                            weight=3.0,
                            pred_act=True,
                            eps=1.0)
                    ]),
                sampler=dict(type='mmdet.MaskPseudoSampler'))),)

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

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ComposedVisualizer', vis_backends=vis_backends, name='visualizer')

default_hooks = dict(visualization=dict(type='SegVisualizationHook', draw=False, interval=5))

