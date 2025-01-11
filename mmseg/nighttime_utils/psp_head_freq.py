# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from mmseg.models.utils import resize
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from .FreqFusion import FreqFusion


class PPMFreqFusion(nn.Module):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self,
                pool_scales,
                in_channels,
                channels,
                conv_cfg,
                norm_cfg,
                act_cfg,
                align_corners,
                use_high_pass=True, 
                use_low_pass=True,
                compress_ratio=8,
                semi_conv=True,
                low2high_residual=False,
                high2low_residual=False,
                lowpass_kernel=5,
                highpass_kernel=3,
                hamming_window=False,
                feature_resample=True,
                feature_resample_group=4,
                comp_feat_upsample=True,
                use_checkpoint=False,
                feature_resample_norm=True,
                **kwargs):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.pool_layers = nn.ModuleList()

        for pool_scale in pool_scales:
            self.pool_layers.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        **kwargs)))
        self.freqfusions = nn.ModuleList()
        self.feature_resample = feature_resample
        self.feature_resample_group = feature_resample_group
        self.use_checkpoint = use_checkpoint
        # from lr to hr
        pre_c = self.channels
        c = self.channels
        for pool_scale in pool_scales:
            freqfusion = FreqFusion(
                hr_channels=c, lr_channels=pre_c, scale_factor=1, lowpass_kernel=lowpass_kernel, highpass_kernel=highpass_kernel, up_group=1, 
                upsample_mode='nearest', align_corners=align_corners, 
                feature_resample=feature_resample, feature_resample_group=feature_resample_group,
                comp_feat_upsample=comp_feat_upsample,
                hr_residual=True, 
                hamming_window=hamming_window,
                compressed_channels= (pre_c + c) // compress_ratio,
                use_high_pass=use_high_pass, use_low_pass=use_low_pass, semi_conv=semi_conv, 
                feature_resample_norm=feature_resample_norm,
                )                
            self.freqfusions.append(freqfusion)
            pre_c += c

        # from lr to hr
        assert not (low2high_residual and high2low_residual)
        self.low2high_residual = low2high_residual
        self.high2low_residual = high2low_residual
        if low2high_residual:
            self.low2high_convs = nn.ModuleList()
            pre_c = in_channels[0]
            for c in in_channels[1:]:
                self.low2high_convs.append(nn.Conv2d(pre_c, c, 1))
                pre_c = c
        elif high2low_residual:
            self.high2low_convs = nn.ModuleList()
            pre_c = in_channels[0]
            for c in in_channels[1:]:
                self.high2low_convs.append(nn.Conv2d(c, pre_c, 1))
                pre_c += c

    def forward(self, x):
        """Forward function."""

        ppm_outs = []
        for ppm in self.pool_layers:
            ppm_out = ppm(x)
            ppm_outs.append(ppm_out)

        inputs = ppm_outs
        in_channels = self.channels
        lowres_feat = inputs[0]
        if self.low2high_residual:
            for hires_feat, freqfusion, low2high_conv in zip(inputs[1:], self.freqfusions, self.low2high_convs):
                pre_c = self.channels
                _, hires_feat, lowres_feat = freqfusion(hr_feat=hires_feat, lr_feat=lowres_feat, use_checkpoint=self.use_checkpoint)
                lowres_feat = torch.cat([hires_feat + low2high_conv(lowres_feat[:, :pre_c]), lowres_feat], dim=1)
            pass
        else:
            for idx, (hires_feat, freqfusion) in enumerate(zip(inputs[1:], self.freqfusions)):      
                _, hires_feat, lowres_feat = freqfusion(hr_feat=hires_feat, lr_feat=lowres_feat, use_checkpoint=self.use_checkpoint)
                if self.feature_resample:
                    b, _, h, w = hires_feat.shape
                    lowres_feat = torch.cat([hires_feat.reshape(b * self.feature_resample_group, -1, h, w), 
                                            lowres_feat.reshape(b * self.feature_resample_group, -1, h, w)], dim=1).reshape(b, -1, h, w)
                else:
                    lowres_feat = torch.cat([hires_feat, lowres_feat], dim=1)
        
        return [lowres_feat]


@MODELS.register_module()
class PSPHeadFreqAware(BaseDecodeHead):
    """Pyramid Scene Parsing Network.

    This head is the implementation of
    `PSPNet <https://arxiv.org/abs/1612.01105>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module. Default: (1, 2, 3, 6).
    """

    def __init__(self,
                pool_scales=(1, 2, 3, 6),
                use_high_pass=True, 
                use_low_pass=True,
                compress_ratio=8,
                semi_conv=True,
                low2high_residual=False,
                high2low_residual=False,
                lowpass_kernel=5,
                highpass_kernel=3,
                hamming_window=False,
                feature_resample=True,
                feature_resample_group=4,
                comp_feat_upsample=True,
                use_checkpoint=False,
                feature_resample_norm=True,
                **kwargs):
        super().__init__(**kwargs)
        assert isinstance(pool_scales, (list, tuple))
        self.pool_scales = pool_scales
        self.psp_modules = PPMFreqFusion(
            self.pool_scales,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners,
            use_high_pass=use_high_pass, 
            use_low_pass=use_low_pass,
            compress_ratio=compress_ratio,
            semi_conv=semi_conv,
            low2high_residual=low2high_residual,
            high2low_residual=high2low_residual,
            lowpass_kernel=lowpass_kernel,
            highpass_kernel=highpass_kernel,
            hamming_window=hamming_window,
            feature_resample=feature_resample,
            feature_resample_group=feature_resample_group,
            comp_feat_upsample=comp_feat_upsample,
            use_checkpoint=use_checkpoint,
            feature_resample_norm=feature_resample_norm
            )
        self.bottleneck = ConvModule(
            self.in_channels + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        print("X" * 100)
        print(len(psp_outs))
        print(psp_outs[0].shape, psp_outs[1].shape)
        psp_outs = torch.cat(psp_outs, dim=1)
        feats = self.bottleneck(psp_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
