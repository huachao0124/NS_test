import logging
from typing import List, Optional, Union, Sequence, Dict, Tuple

import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from mmseg.registry import MODELS, TASK_UTILS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from mmdet.utils import InstanceList, reduce_mean
from mmdet.models.utils import multi_apply

from mmseg.models.segmentors import EncoderDecoder
from mmseg.models.utils import resize


import copy

import torch
from mmengine.structures import InstanceData, PixelData

from mmseg.models.assigners.base_assigner import BaseAssigner
from .utils import (compute_error_matrix,
                    KeysRecorder,
                    o2o_mask_dice,
                    CodebookContrastiveHead,
                    CodebookContrastiveHead2,
                    CodebookContrastiveHead3,
                    CodebookContrastiveHead6,
                    CodebookContrastiveHead7,
                    Mask2FormerTransformerDecoderReliableMatching,
                    Mask2FormerTransformerDecoderLBP)


from mmseg.models.decode_heads import Mask2FormerHead
from mmseg.structures.seg_data_sample import SegDataSample

from mmcv.cnn import Conv2d
from mmcv.ops import point_sample
from mmengine.model import ModuleList, caffe2_xavier_init
from mmdet.models.utils import get_uncertain_point_coords_with_randomness
from info_nce import InfoNCE
from collections import defaultdict

from mmdet.models.dense_heads import AnchorFreeHead


@MODELS.register_module()
class EncoderDecoderAnalysis(EncoderDecoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.assigner = TASK_UTILS.build(dict(type='GroupMaxIoUAssigner', iou_threshold=0))
        self.assigner = TASK_UTILS.build(dict(type='MaskMaxIoUAssigner', iou_threshold=0))
        self.sampler = TASK_UTILS.build(dict(type='mmdet.MaskPseudoSampler'))
        self.ignore_index = 255

    def predict(self,
                inputs: Tensor,
                batch_data_samples: OptSampleList = None) -> SampleList:
        if batch_data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in batch_data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        
        # batch SegDataSample to InstanceDataSample
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(batch_data_samples)
        
        x = self.extract_feat(inputs)
        all_cls_scores, all_mask_preds = self.decode_head(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape']
        else:
            size = batch_img_metas[0]['img_shape']
        # upsample mask
        mask_pred_results = F.interpolate(
            mask_pred_results, size=size, mode='bilinear', align_corners=False)
        
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        # cls_score = mask_cls_results.sigmoid()
        mask_pred = mask_pred_results.sigmoid()

        # print(cls_score)

        # import numpy as np
        # np.savetxt(f'cls_score_txts/{np.random.randint(0, 100)}.txt', cls_score.squeeze(0).cpu().numpy(), fmt="%.4f")
        # mask = cls_score < 0.3
        # cls_score[mask] = 0

        # modify
        # mask_pred_binary = (mask_pred >= 0.5).long()
        mask_pred_binary = mask_pred.clone()
        ems, matched_labels, matched_masks = self.get_targets(mask_cls_results, mask_pred_binary, batch_gt_instances, batch_img_metas)
        # cls_score = torch.zeros_like(cls_score).scatter_(dim=-1, index=cls_score.max(dim=-1)[1].unsqueeze(-1), value=1)
        # cls_score[matched_labels == self.decode_head.num_classes] = 0

        cls_score_rectified = torch.zeros_like(mask_cls_results).scatter_(dim=-1, index=matched_labels.unsqueeze(-1), value=1)[..., :-1]
        # cls_score[matched_labels != 19] = cls_score_rectified[matched_labels != 19]
        cls_score = cls_score + cls_score_rectified
        # cls_score = cls_score / (torch.sum(cls_score, dim=-2, keepdim=True) + 1e-5)
        # mask_pred = matched_masks.float()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)

        return self.postprocess_result(seg_logits, ems, batch_data_samples)

    def _seg_data_to_instance_data(self, batch_data_samples: SampleList):
        batch_img_metas = []
        batch_gt_instances = []

        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            gt_sem_seg = data_sample.gt_sem_seg.data
            classes = torch.unique(
                gt_sem_seg,
                sorted=False,
                return_inverse=False,
                return_counts=False)

            # remove ignored region
            gt_labels = classes[classes != self.ignore_index]

            masks = []
            for class_id in gt_labels:
                masks.append(gt_sem_seg == class_id)

            if len(masks) == 0:
                gt_masks = torch.zeros(
                    (0, gt_sem_seg.shape[-2],
                     gt_sem_seg.shape[-1])).to(gt_sem_seg).long()
            else:
                gt_masks = torch.stack(masks).squeeze(1).long()

            instance_data = InstanceData(labels=gt_labels, masks=gt_masks)
            batch_gt_instances.append(instance_data)
        return batch_gt_instances, batch_img_metas


    def get_targets(
        self,
        cls_scores_list: List[Tensor],
        mask_preds_list: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        return_sampling_results: bool = False
    ) -> Tuple[List[Union[Tensor, int]]]:
        results = multi_apply(self._get_targets_single, cls_scores_list,
                              mask_preds_list, batch_gt_instances,
                              batch_img_metas)
        ems = results[0]
        matched_labels = results[1]
        matched_masks = results[2]

        ems = torch.stack(ems)
        matched_labels = torch.stack(matched_labels)
        matched_masks = torch.stack(matched_masks)

        return ems, matched_labels, matched_masks


    def _get_targets_single(self, cls_score: Tensor, mask_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> Tuple[Tensor]:
        gt_labels = gt_instances.labels
        gt_masks = gt_instances.masks
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        # point_coords = torch.rand((1, self.num_points, 2),
        #                           device=cls_score.device)
        # # shape (num_queries, num_points)
        # mask_points_pred = point_sample(
        #     mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1,
        #                                                 1)).squeeze(1)
        # # shape (num_gts, num_points)
        # gt_points_masks = point_sample(
        #     gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1,
        #                                                        1)).squeeze(1)

        # sampled_gt_instances = InstanceData(
        #     labels=gt_labels, masks=gt_points_masks)
        # sampled_pred_instances = InstanceData(
        #     scores=cls_score, masks=mask_points_pred)

        sampled_gt_instances = InstanceData(
            labels=gt_labels, masks=gt_masks)
        sampled_pred_instances = InstanceData(
            scores=cls_score, masks=mask_pred)
        # assign and sample
        assign_result, _ = self.assigner.assign(
            pred_instances=sampled_pred_instances,
            gt_instances=sampled_gt_instances,
            img_meta=img_meta)
        pred_instances = InstanceData(scores=cls_score, masks=mask_pred)
        sampling_result = self.sampler.sample(
            assign_result=assign_result,
            pred_instances=pred_instances,
            gt_instances=gt_instances)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        matched_labels = gt_labels.new_full((num_queries, ),
                                    self.decode_head.num_classes,
                                    dtype=torch.long)
        matched_labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]

        # pred_labels = cls_score[..., :-1].max(dim=-1)[1]
        pred_labels = cls_score.max(dim=-1)[1]

        em = compute_error_matrix(matched_labels, pred_labels, cls_score.shape[-1])
        # em = compute_error_matrix(matched_labels[overlaps_max > 0.7], pred_labels[overlaps_max > 0.7], 19)

        matched_masks = gt_masks.new_zeros((self.decode_head.num_queries, *gt_masks.shape[-2:]))
        matched_masks[pos_inds] = gt_masks[sampling_result.pos_assigned_gt_inds]

        return em, matched_labels, matched_masks


    def postprocess_result(self,
                           seg_logits: Tensor,
                           ems: Tensor,
                           data_samples: OptSampleList = None) -> SampleList:
        batch_size, C, H, W = seg_logits.shape

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom =\
                    padding_size
                # i_seg_logits shape is 1, C, H, W after remove padding
                i_seg_logits = seg_logits[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3, ))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2, ))

                # resize as original shape
                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta['ori_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False).squeeze(0)
            else:
                i_seg_logits = seg_logits[i]

            if C > 1:
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
            else:
                i_seg_logits = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits >
                              self.decode_head.threshold).to(i_seg_logits)
            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': i_seg_logits}),
                'pred_sem_seg':
                PixelData(**{'data': i_seg_pred}),
                'error_matrix':
                PixelData(**{'data': ems[i]}),
            })

        return data_samples



@MODELS.register_module()
class AlignMask2FormerHead(Mask2FormerHead):
    def __init__(self,
                 all_layers_num_gt_repeat: List[int] = None,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 tau: float = 1.5,
                 **kwargs):
        self.all_layers_num_gt_repeat = all_layers_num_gt_repeat
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.weight_table = torch.zeros(
            len(all_layers_num_gt_repeat), max(all_layers_num_gt_repeat))
        for layer_index, num_gt_repeat in enumerate(all_layers_num_gt_repeat):
            self.weight_table[layer_index][:num_gt_repeat] = torch.exp(
                -torch.arange(num_gt_repeat) / tau)
        super().__init__(**kwargs)

        assert len(self.all_layers_num_gt_repeat) == self.num_transformer_decoder_layers + 1
    
    def loss_by_feat(self, all_cls_scores: Tensor, all_mask_preds: Tensor,
                     batch_gt_instances: List[InstanceData],
                     batch_img_metas: List[dict]) -> Dict[str, Tensor]:
        num_dec_layers = len(all_cls_scores)
        all_cls_scores = KeysRecorder(all_cls_scores)
        batch_gt_instances_list = [
            batch_gt_instances for _ in range(num_dec_layers)
        ]
        img_metas_list = [batch_img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_mask, losses_dice = multi_apply(
            self._loss_by_feat_single, all_cls_scores, all_mask_preds,
            batch_gt_instances_list, img_metas_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i in zip(
                losses_cls[:-1], losses_mask[:-1], losses_dice[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            num_dec_layer += 1
        return loss_dict

    def _loss_by_feat_single(self, cls_scores: Union[KeysRecorder, Tensor],
                             mask_preds: Tensor,
                             batch_gt_instances: List[InstanceData],
                             batch_img_metas: List[dict]) -> Tuple[Tensor]:
        if isinstance(cls_scores, KeysRecorder):
            # Outputs are from decoder layer. Get layer_index from
            #   `__getitem__` keys history.
            keys = [key for key in cls_scores.keys if isinstance(key, int)]
            assert len(keys) == 1, \
                'Failed to extract key from cls_scores.keys: {}'.format(keys)
            layer_index = keys[0]
            # Get dn_cls_scores tensor.
            cls_scores = cls_scores.obj
        else:
            # Outputs are from encoder layer.
            layer_index = self.num_pred_layer - 1

        for img_meta in batch_img_metas:
            img_meta['layer_index'] = layer_index
        
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list, num_total_neg,
         avg_factor) = self.get_targets(cls_scores_list, mask_preds_list,
                                        batch_gt_instances, batch_img_metas)
        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        
        # modify
        num_total_pos = sum(
            len(gt_instances) for gt_instances in batch_gt_instances)
        self.bg_cls_weight = 0.1
        self.sync_cls_avg_factor = True
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            # avg_factor=class_weight[labels].sum())
            avg_factor=cls_avg_factor)

        num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)

        return loss_cls, loss_mask, loss_dice
    

    def get_targets(
        self,
        cls_scores_list: List[Tensor],
        mask_preds_list: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        return_sampling_results: bool = False
    ) -> Tuple[List[Union[Tensor, int]]]:
        results = multi_apply(self._get_targets_single, cls_scores_list,
                              mask_preds_list, batch_gt_instances,
                              batch_img_metas)
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])

        avg_factor = sum(
            [results.avg_factor for results in sampling_results_list])

        num_total_neg = sum((inds.numel() for inds in neg_inds_list))

        res = (labels_list, label_weights_list, mask_targets_list,
               mask_weights_list, num_total_neg, avg_factor)
        if return_sampling_results:
            res = res + (sampling_results_list)

        return res + tuple(rest_results)


    def _get_targets_single(self, cls_score: Tensor, mask_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> Tuple[Tensor]:
        gt_labels = gt_instances.labels
        gt_masks = gt_instances.masks
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2),
                                  device=cls_score.device)
        # shape (num_queries, num_points)
        mask_points_pred = point_sample(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1,
                                                        1)).squeeze(1)
        # shape (num_gts, num_points)
        gt_points_masks = point_sample(
            gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1,
                                                               1)).squeeze(1)

        sampled_gt_instances = InstanceData(
            labels=gt_labels, masks=gt_points_masks)
        sampled_pred_instances = InstanceData(
            scores=cls_score, masks=mask_points_pred)
        # assign and sample
        assign_result = self.assigner.assign(
            pred_instances=sampled_pred_instances,
            gt_instances=sampled_gt_instances,
            img_meta=img_meta)
        pred_instances = InstanceData(scores=cls_score, masks=mask_pred)
        sampling_result = self.sampler.sample(
            assign_result=assign_result,
            pred_instances=pred_instances,
            gt_instances=gt_instances)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds
        pos_gt_masks = gt_masks[pos_assigned_gt_inds.long(), :]
        layer_index = img_meta['layer_index']
        labels, label_weights, mask_weights = self._get_align_detr_targets_single(
            cls_score,
            mask_pred,
            gt_labels,
            pos_gt_masks,
            pos_inds,
            pos_assigned_gt_inds,
            layer_index,
            is_matching_queries=True)
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]

        # # label target
        # labels = gt_labels.new_full((self.num_queries, ),
        #                             self.num_classes,
        #                             dtype=torch.long)
        # labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        # label_weights = gt_labels.new_ones((self.num_queries, ))

        # # mask target
        # mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        # mask_weights = mask_pred.new_zeros((self.num_queries, ))
        # mask_weights[pos_inds] = 1.0

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds, sampling_result)

    def _get_align_detr_targets_single(self,
                                       cls_score: Tensor,
                                       mask_pred: Tensor,
                                       gt_labels: Tensor,
                                       pos_gt_masks: Tensor,
                                       pos_inds: Tensor,
                                       pos_assigned_gt_inds: Tensor,
                                       layer_index: int = -1,
                                       is_matching_queries: bool = False):
        # Classification loss
        # =           1 * BCE(prob, t * rank_weights) for positive sample;
        # = prob**gamma * BCE(prob,                0) for negative sample.
        # That is,
        # label_targets = 0                for negative sample;
        #               = t * rank_weights for positive sample.
        # label_weights = pred**gamma for negative sample;
        #               = 1           for positive sample.
        cls_prob = cls_score.sigmoid()
        label_targets = torch.zeros_like(cls_score, device=pos_gt_masks.device)
        label_weights = cls_prob**self.gamma

        mask_weights = mask_pred.new_zeros((self.num_queries, ))

        if len(pos_inds) == 0:
            return label_targets, label_weights, mask_weights

        pos_cls_score_inds = (pos_inds, gt_labels[pos_assigned_gt_inds])

        target_shape = pos_gt_masks.shape[-2:]

        mask_pred = F.interpolate(
            mask_pred.unsqueeze(1),
            target_shape,
            mode='bilinear',
            align_corners=False).squeeze(1)

        iou_scores = o2o_mask_dice(mask_pred[pos_inds], pos_gt_masks)

        # t (Tensor): The weighted geometric average of the confident score
        #   and the IoU score, to align classification and regression scores.
        #   Shape [num_positive].
        t = (
            cls_prob[pos_cls_score_inds]**self.alpha *
            iou_scores**(1 - self.alpha))
        t = torch.clamp(t, 0.01).detach()

        # Calculate rank_weights for matching queries.
        if is_matching_queries:
            # rank_weights (Tensor): Weights of each group of predictions
            #   assigned to the same positive gt bbox. Shape [num_positive].
            rank_weights = torch.zeros_like(t, dtype=self.weight_table.dtype)

            assert 0 <= layer_index < len(self.weight_table), layer_index
            rank_to_weight = self.weight_table[layer_index].to(
                rank_weights.device)
            unique_gt_inds = torch.unique(pos_assigned_gt_inds)

            # For each positive gt bbox, get all predictions assigned to it,
            #   then calculate rank weights for this group of predictions.
            for gt_index in unique_gt_inds:
                pred_group_cond = pos_assigned_gt_inds == gt_index
                # Weights are based on their rank sorted by t in the group.
                pred_group = t[pred_group_cond]
                indices = pred_group.sort(descending=True)[1]
                group_weights = torch.zeros_like(
                    indices, dtype=self.weight_table.dtype)
                group_weights[indices] = rank_to_weight[:len(indices)]
                rank_weights[pred_group_cond] = group_weights

            t = t * rank_weights
            mask_weights[pos_inds] = rank_weights
        else:
            mask_weights[pos_inds] = 1.0

        # label_targets[pos_cls_score_inds] = t
        label_targets[pos_cls_score_inds] = 1
        label_weights[pos_cls_score_inds] = 1.0

        

        return label_targets, label_weights, mask_weights
    


@MODELS.register_module()
class Mask2FormerHeadContrast(Mask2FormerHead):
    '''
    分类改为和class/background embeddings的相似度
    '''
    def __init__(self, queries_per_class=5, **kwargs):
        super().__init__(**kwargs)
        self.queries_per_class = queries_per_class
        self._init_cls_embed(queries_per_class, kwargs['feat_channels'])
        assert self.num_queries == self.num_classes * queries_per_class
    
    def _init_cls_embed(self, queries_per_class, feat_channels):
        self.cls_embed = CodebookContrastiveHead(num_classes=self.num_classes, queries_per_class=queries_per_class, embedding_dim=feat_channels)
    
    def forward(self, x: List[Tensor],
                batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        '''
        将query_feat作为返回值一并返回
        不同组的queries之间屏蔽attention
        '''
        batch_size = x[0].shape[0]
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        query_feat_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)
        query_feat_list.append(query_feat)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
            attn_mask = attn_mask & mask_sum
            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            # modify
            # 不同组的queries之间屏蔽attention
            indices = torch.arange(self.num_queries).unsqueeze(1).repeat(1, self.num_queries)
            groups = indices // self.queries_per_class
            self_attn_mask = ~(groups == groups.T).to(query_feat.device)
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                self_attn_mask=self_attn_mask,
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
            query_feat_list.append(query_feat)

        return cls_pred_list, mask_pred_list, query_feat_list

    def predict(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tuple[Tensor]:
        batch_data_samples = [
            SegDataSample(metainfo=metainfo) for metainfo in batch_img_metas
        ]

        all_cls_scores, all_mask_preds, _ = self(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape']
        else:
            size = batch_img_metas[0]['img_shape']
        # upsample mask
        mask_pred_results = F.interpolate(
            mask_pred_results, size=size, mode='bilinear', align_corners=False)
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
        return seg_logits


    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        '''
        loss中加入query_feat和class_embedding之间的loss_codebook
        '''
        # batch SegDataSample to InstanceDataSample
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples)

        # forward
        all_cls_scores, all_mask_preds, all_query_feats = self(x, batch_data_samples)

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds, all_query_feats,
                                   batch_gt_instances, batch_img_metas)

        return losses

    def loss_by_feat(self, all_cls_scores: Tensor, all_mask_preds: Tensor, all_query_feats,
                     batch_gt_instances: List[InstanceData],
                     batch_img_metas: List[dict]) -> Dict[str, Tensor]:
        '''
        加入loss_codebook
        '''
        num_dec_layers = len(all_cls_scores)
        batch_gt_instances_list = [
            batch_gt_instances for _ in range(num_dec_layers)
        ]
        img_metas_list = [batch_img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_mask, losses_dice, losses_codebook = multi_apply(
            self._loss_by_feat_single, all_cls_scores, all_mask_preds, all_query_feats,
            batch_gt_instances_list, img_metas_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        if losses_codebook[-1] is not None:
            loss_dict['loss_codebook'] = losses_codebook[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i, loss_codebook_i in zip(
                losses_cls[:-1], losses_mask[:-1], losses_dice[:-1], losses_codebook[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            if loss_codebook_i is not None:
                loss_dict[f'd{num_dec_layer}.loss_codebook'] = loss_codebook_i
            num_dec_layer += 1
        return loss_dict

    def _loss_by_feat_single(self, cls_scores: Tensor, mask_preds: Tensor, query_feat: Tensor,
                             batch_gt_instances: List[InstanceData],
                             batch_img_metas: List[dict]) -> Tuple[Tensor]:
        '''
        加入loss_codebook的计算
        '''
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        query_feats_list = [query_feat[i] for i in range(num_imgs)]
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         avg_factor) = self.get_targets(cls_scores_list, mask_preds_list,
                                        batch_gt_instances, batch_img_metas)
        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)

        # 将对应的bg embed和cls embed选出进行更新
        # loss_codebook = F.mse_loss(self.cls_embed.class_embeddings(labels), query_feat.detach().flatten(0, 1), reduction='mean')
        class_indices = torch.arange(self.num_classes).repeat_interleave(self.queries_per_class).view(1, -1).to(query_feat.device)

        # loss_codebook = F.mse_loss(self.cls_embed.class_embeddings(labels[labels < self.num_classes]), query_feat.detach()[labels < self.num_classes], reduction='mean') + \
        #     0.2 * F.mse_loss(self.cls_embed.class_embeddings((labels + class_indices)[labels == self.num_classes]), query_feat.detach()[labels == self.num_classes], reduction='mean')
        loss_codebook = None

        return loss_cls, loss_mask, loss_dice, loss_codebook
    

@MODELS.register_module()
class Mask2FormerHeadContrast2(Mask2FormerHeadContrast):
    '''
    分类改为和class/background embeddings的相似度
    添加额外的max_iou_assigner
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_iou_assigner = TASK_UTILS.build(dict(type='GroupMaxIoUAssigner', iou_threshold=0.0))
    
    def _init_cls_embed(self, queries_per_class, feat_channels):
        self.cls_embed = CodebookContrastiveHead2(num_classes=self.num_classes, queries_per_class=queries_per_class, embedding_dim=feat_channels)
    
    def _get_targets_single(self, cls_score: Tensor, mask_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> Tuple[Tensor]:
        '''
        mask全监督
        '''
        gt_labels = gt_instances.labels
        gt_masks = gt_instances.masks
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2),
                                  device=cls_score.device)
        # shape (num_queries, num_points)
        mask_points_pred = point_sample(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1,
                                                        1)).squeeze(1)
        # shape (num_gts, num_points)
        gt_points_masks = point_sample(
            gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1,
                                                               1)).squeeze(1)

        sampled_gt_instances = InstanceData(
            labels=gt_labels, masks=gt_points_masks)
        sampled_pred_instances = InstanceData(
            scores=cls_score, masks=mask_points_pred)
        # assign and sample
        assign_result = self.assigner.assign(
            pred_instances=sampled_pred_instances,
            gt_instances=sampled_gt_instances,
            img_meta=img_meta)
        pred_instances = InstanceData(scores=cls_score, masks=mask_pred)
        sampling_result = self.sampler.sample(
            assign_result=assign_result,
            pred_instances=pred_instances,
            gt_instances=gt_instances)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        labels = gt_labels.new_full((self.num_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((self.num_queries, ))

        # mask target

        if len(sampled_gt_instances) > 0:
            max_iou_assign_result, overlaps_max = self.max_iou_assigner.assign(
                pred_instances=sampled_pred_instances,
                gt_instances=sampled_gt_instances,
                img_meta=img_meta)
            max_iou_sampling_result = self.sampler.sample(
                assign_result=max_iou_assign_result,
                pred_instances=pred_instances,
                gt_instances=gt_instances)
        
        mask_targets = gt_masks.new_zeros((self.num_queries, *gt_masks.shape[-2:]))
        mask_weights = mask_pred.new_ones((self.num_queries, )) * 0.1

        if len(sampled_gt_instances) > 0:
            mask_targets[max_iou_sampling_result.pos_inds] = gt_masks[max_iou_sampling_result.pos_assigned_gt_inds]
            mask_weights[max_iou_sampling_result.pos_inds] = overlaps_max[max_iou_sampling_result.pos_inds]
        
        mask_targets[pos_inds] = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights[pos_inds] = 1.0

        sampling_result.avg_factor = mask_weights.sum()

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds, sampling_result)

    def _loss_by_feat_single(self, cls_scores: Tensor, mask_preds: Tensor, query_feat: Tensor,
                             batch_gt_instances: List[InstanceData],
                             batch_img_metas: List[dict]) -> Tuple[Tensor]:
        '''
        加入loss_codebook的计算
        '''
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        query_feats_list = [query_feat[i] for i in range(num_imgs)]
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         avg_factor) = self.get_targets(cls_scores_list, mask_preds_list,
                                        batch_gt_instances, batch_img_metas)
        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)

        # 将对应的bg embed和cls embed选出进行更新
        # loss_codebook = F.mse_loss(self.cls_embed.class_embeddings(labels), query_feat.detach().flatten(0, 1), reduction='mean')
        class_indices = torch.arange(self.num_classes).repeat_interleave(self.queries_per_class).view(1, -1).to(query_feat.device)

        # loss_codebook = F.mse_loss(self.cls_embed.class_embeddings(labels[labels < self.num_classes]), query_feat.detach()[labels < self.num_classes], reduction='mean') + \
        #     0.2 * F.mse_loss(self.cls_embed.class_embeddings((labels + class_indices)[labels == self.num_classes]), query_feat.detach()[labels == self.num_classes], reduction='mean')
        loss_codebook = None

        return loss_cls, loss_mask, loss_dice, loss_codebook


@MODELS.register_module()
class Mask2FormerHeadContrast3(Mask2FormerHeadContrast):
    '''
    分类改为 weighted averaged pooled mask features 和 class/background embeddings at first layer的相似度
    '''

    def _init_cls_embed(self, queries_per_class, feat_channels):
        self.cls_embed = CodebookContrastiveHead3(num_classes=self.num_classes, queries_per_class=queries_per_class, embedding_dim=feat_channels)

    def _forward_head(self, decoder_out: Tensor, mask_feature: Tensor,
                      attn_mask_target_size: Tuple[int, int]) -> Tuple[Tensor]:
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        # shape (num_queries, batch_size, c)
        mask_embed = self.mask_embed(decoder_out)
        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)

        # shape (num_queries, batch_size, c)
        mask_weights = mask_pred.sigmoid().detach()
        mask_weights = mask_weights / (mask_weights.sum(dim=(-1, -2), keepdim=True) + 1e-7)

        pooled_features = (mask_weights.unsqueeze(2) * mask_feature.unsqueeze(1)).sum(dim=(-1, -2)) # [b, num_queries, h, w] * [b, c, h, w]
        cls_pred = self.cls_embed(pooled_features, is_training=True)

        attn_mask = F.interpolate(
            mask_pred,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return cls_pred, mask_pred, attn_mask

    def predict(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tuple[Tensor]:
        batch_data_samples = [
            SegDataSample(metainfo=metainfo) for metainfo in batch_img_metas
        ]

        all_cls_scores, all_mask_preds, _ = self(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape']
        else:
            size = batch_img_metas[0]['img_shape']
        # upsample mask
        mask_pred_results = F.interpolate(
            mask_pred_results, size=size, mode='bilinear', align_corners=False)
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
        return seg_logits


@MODELS.register_module()
class Mask2FormerHeadContrast4(Mask2FormerHeadContrast3):
    '''
    分类改为 weighted averaged pooled image representations 和 class/background embeddings at corresponding layers的相似度
    去掉loss_cls  用loss_codebook监督
    '''
    def forward(self, x: List[Tensor],
                batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        '''
        将query_feat, mask_features 作为返回值一并返回
        '''
        batch_size = x[0].shape[0]
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        query_feat_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)
        query_feat_list.append(query_feat)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
            attn_mask = attn_mask & mask_sum
            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
            query_feat_list.append(query_feat)

        return cls_pred_list, mask_pred_list, query_feat_list, mask_features.detach()   # 一定要detach

    def predict(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tuple[Tensor]:
        batch_data_samples = [
            SegDataSample(metainfo=metainfo) for metainfo in batch_img_metas
        ]

        all_cls_scores, all_mask_preds, _, _ = self(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape']
        else:
            size = batch_img_metas[0]['img_shape']
        # upsample mask
        mask_pred_results = F.interpolate(
            mask_pred_results, size=size, mode='bilinear', align_corners=False)
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
        return seg_logits

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        '''
        loss中加入query_feat和class_embedding之间的loss_codebook
        '''
        # batch SegDataSample to InstanceDataSample
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples)

        # forward
        all_cls_scores, all_mask_preds, all_query_feats, mask_features = self(x, batch_data_samples)

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds, all_query_feats, mask_features,
                                   batch_gt_instances, batch_img_metas)

        return losses

    def loss_by_feat(self, all_cls_scores: Tensor, all_mask_preds: Tensor, all_query_feats, mask_features,
                     batch_gt_instances: List[InstanceData],
                     batch_img_metas: List[dict]) -> Dict[str, Tensor]:
        '''
        加入loss_codebook
        '''
        num_dec_layers = len(all_cls_scores)
        batch_gt_instances_list = [
            batch_gt_instances for _ in range(num_dec_layers)
        ]
        img_metas_list = [batch_img_metas for _ in range(num_dec_layers)]
        mask_features_list = [
            mask_features for _ in range(num_dec_layers)
        ]
        losses_cls, losses_mask, losses_dice, losses_codebook = multi_apply(
            self._loss_by_feat_single, all_cls_scores, all_mask_preds, all_query_feats, mask_features_list,
            batch_gt_instances_list, img_metas_list)

        loss_dict = dict()
        # loss from the last decoder layer
        # loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        if losses_codebook[-1] is not None:
            loss_dict['loss_codebook'] = losses_codebook[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i, loss_codebook_i in zip(
                losses_cls[:-1], losses_mask[:-1], losses_dice[:-1], losses_codebook[:-1]):
            # loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            if loss_codebook_i is not None:
                loss_dict[f'd{num_dec_layer}.loss_codebook'] = loss_codebook_i
            num_dec_layer += 1
        return loss_dict

    def _loss_by_feat_single(self, cls_scores: Tensor, mask_preds: Tensor, query_feat: Tensor, mask_features: Tensor,
                             batch_gt_instances: List[InstanceData],
                             batch_img_metas: List[dict]) -> Tuple[Tensor]:
        '''
        加入loss_codebook的计算
        '''
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        query_feats_list = [query_feat[i] for i in range(num_imgs)]
        mask_features_list = [mask_features[i] for i in range(num_imgs)]
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list, embedding_targets_list,
         avg_factor) = self.get_targets(cls_scores_list, mask_preds_list, mask_features_list,
                                        batch_gt_instances, batch_img_metas)
        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # 将对应的bg embed和cls embed选出进行更新
        class_indices = torch.arange(self.num_classes).repeat_interleave(self.queries_per_class).view(1, -1).repeat(num_imgs, 1).to(query_feat.device)
        embedding_targets = torch.stack(embedding_targets_list)
        cls_embeddings = self.cls_embed.class_embeddings(class_indices + self.num_classes)  # 没匹配到的是background embeddings
        cls_embeddings[labels < self.num_classes] = self.cls_embed.class_embeddings(class_indices[labels < self.num_classes])   # 匹配到的是class embeddings
        if torch.isnan(embedding_targets).any():
            print("X" * 100)
            print(embedding_targets.max(), embedding_targets.min())


        loss_codebook = F.mse_loss(cls_embeddings, embedding_targets.detach(), reduction='mean')

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        # loss_cls = self.loss_cls(
        #     cls_scores,
        #     labels,
        #     label_weights,
        #     avg_factor=class_weight[labels].sum())
        loss_cls = None

        num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)
        

        return loss_cls, loss_mask, loss_dice, loss_codebook

    def get_targets(
        self,
        cls_scores_list: List[Tensor],
        mask_preds_list: List[Tensor],
        mask_features_list: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        return_sampling_results: bool = False
    ) -> Tuple[List[Union[Tensor, int]]]:
        results = multi_apply(self._get_targets_single, cls_scores_list,
                              mask_preds_list, mask_features_list, batch_gt_instances,
                              batch_img_metas)
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list, embedding_targets_list,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:8]
        rest_results = list(results[8:])

        avg_factor = sum(
            [results.avg_factor for results in sampling_results_list])

        res = (labels_list, label_weights_list, mask_targets_list,
               mask_weights_list, embedding_targets_list, avg_factor)
        if return_sampling_results:
            res = res + (sampling_results_list)

        return res + tuple(rest_results)

    def _get_targets_single(self, cls_score: Tensor, mask_pred: Tensor, mask_feature: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> Tuple[Tensor]:
        gt_labels = gt_instances.labels
        gt_masks = gt_instances.masks
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2),
                                  device=cls_score.device)
        # shape (num_queries, num_points)
        mask_points_pred = point_sample(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1,
                                                        1)).squeeze(1)
        # shape (num_gts, num_points)
        gt_points_masks = point_sample(
            gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1,
                                                               1)).squeeze(1)

        sampled_gt_instances = InstanceData(
            labels=gt_labels, masks=gt_points_masks)
        sampled_pred_instances = InstanceData(
            scores=cls_score, masks=mask_points_pred)
        # assign and sample
        assign_result = self.assigner.assign(
            pred_instances=sampled_pred_instances,
            gt_instances=sampled_gt_instances,
            img_meta=img_meta)
        pred_instances = InstanceData(scores=cls_score, masks=mask_pred)
        sampling_result = self.sampler.sample(
            assign_result=assign_result,
            pred_instances=pred_instances,
            gt_instances=gt_instances)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        labels = gt_labels.new_full((self.num_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((self.num_queries, ))

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = 1.0

        mask_feature = mask_feature.detach()
        embedding_targets = mask_feature.mean(dim=(-1, -2)).unsqueeze(0).repeat(num_queries, 1)    # [c, h, w] -> [n, c]

        for i in range(self.num_classes):
            if (gt_labels == i).any():
                i_mask_bg = 1 - gt_masks[gt_labels == i]    # [1, h, w]
                i_mask_bg = F.interpolate(i_mask_bg.unsqueeze(0).float(), mask_feature.shape[-2:], mode='bilinear').squeeze(0)
                i_mask_bg = i_mask_bg / (i_mask_bg.sum() + 1e-7)
                embedding_targets[i*self.queries_per_class: (i+1)*self.queries_per_class] = (i_mask_bg * mask_feature).sum(dim=(-1, -2))
        mask_targets_resize = F.interpolate(mask_targets.unsqueeze(1).float(), mask_feature.shape[-2:], mode='bilinear')
        mask_targets_resize = mask_targets_resize / (mask_targets_resize.sum(dim=(-1, -2), keepdim=True) + 1e-7)
        embedding_targets[pos_inds] = (mask_targets_resize * mask_feature.unsqueeze(0)).sum(dim=(-1, -2))  # [num_masks, h, w] * [c, h, w] -> [num_masks, c] / [num_masks, 1]

        return (labels, label_weights, mask_targets, mask_weights, embedding_targets, pos_inds,
                neg_inds, sampling_result)




@MODELS.register_module()
class Mask2FormerHeadContrast5(Mask2FormerHeadContrast):
    '''
    分类改为和class/background embeddings的相似度
    class embeddins之间有loss info nce
    '''
    def _init_cls_embed(self, queries_per_class, feat_channels):
        self.cls_embed = CodebookContrastiveHead(num_classes=self.num_classes, queries_per_class=queries_per_class, embedding_dim=feat_channels)

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        '''
        loss中加入query_feat和class_embedding之间的loss_codebook
        '''
        # batch SegDataSample to InstanceDataSample
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples)

        # forward
        all_cls_scores, all_mask_preds, all_query_feats = self(x, batch_data_samples)

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds, all_query_feats,
                                   batch_gt_instances, batch_img_metas)
        
        loss_contrastive = InfoNCE(temperature=0.1, reduction='sum', negative_mode='paired')
        positive_keys = self.cls_embed.class_embeddings.weight
        negtive_indices = torch.arange(self.num_classes+1)[None, :].expand(self.num_classes+1, -1)[~torch.eye(self.num_classes+1, dtype=torch.bool)].view(self.num_classes+1, -1).to(x[0].device)
        negative_keys = self.cls_embed.class_embeddings(negtive_indices)
        losses['loss_info_nce'] = loss_contrastive(positive_keys, positive_keys, negative_keys)

        return losses

    def _loss_by_feat_single(self, cls_scores: Tensor, mask_preds: Tensor, query_feat: Tensor,
                             batch_gt_instances: List[InstanceData],
                             batch_img_metas: List[dict]) -> Tuple[Tensor]:
        '''
        加入loss_codebook的计算
        '''
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        query_feats_list = [query_feat[i] for i in range(num_imgs)]
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         avg_factor) = self.get_targets(cls_scores_list, mask_preds_list,
                                        batch_gt_instances, batch_img_metas)
        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)

        # 将对应的bg embed和cls embed选出进行更新
        # loss_codebook = F.mse_loss(self.cls_embed.class_embeddings(labels), query_feat.detach().flatten(0, 1), reduction='mean')
        loss_codebook = None

        return loss_cls, loss_mask, loss_dice, loss_codebook


@MODELS.register_module()
class Mask2FormerHeadContrast6(Mask2FormerHeadContrast):
    '''
    分类改为queries和class/background embeddings的相似度
    queries和class/background embeddings有loss info nce
    正样本：同类中相似度最高的正原型
    负样本：同类背景 + 其他类的所有原型
    '''
    def _init_cls_embed(self, queries_per_class, feat_channels):
        self.cls_embed = CodebookContrastiveHead6(num_classes=self.num_classes, queries_per_class=queries_per_class, embedding_dim=feat_channels)    

    def _loss_by_feat_single(self, cls_scores: Tensor, mask_preds: Tensor, query_feat: Tensor,
                             batch_gt_instances: List[InstanceData],
                             batch_img_metas: List[dict]) -> Tuple[Tensor]:
        '''
        加入loss_codebook的计算
        '''
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        query_feats_list = [query_feat[i] for i in range(num_imgs)]
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         avg_factor) = self.get_targets(cls_scores_list, mask_preds_list,
                                        batch_gt_instances, batch_img_metas)
        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # modify
        # 将对应的bg embed和cls embed选出进行更新
        # loss_codebook = F.mse_loss(self.cls_embed.class_embeddings(labels), query_feat.detach().flatten(0, 1), reduction='mean')
        loss_codebook = self.cls_embed.loss_only_qp(query_feat, labels)

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)

        return loss_cls, loss_mask, loss_dice, loss_codebook


@MODELS.register_module()
class Mask2FormerHeadContrast7(Mask2FormerHead):
    '''
    分类改为queries和class/background embeddings的相似度
    但不再为每个query固定匹配
    (19类 + 1bg) * 5 embeddings
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cls_embed = CodebookContrastiveHead7(num_classes=self.num_classes, prototypes_per_class=5, embedding_dim=kwargs['feat_channels'])    


@MODELS.register_module()
class Mask2FormerHeadContrast8(Mask2FormerHead):
    '''
    各层不同类之间特征有loss_info_nce
    '''
    
    def visual_representations_info_nce_loss(self, image_features, batch_gt_instances, temperature=0.1):
        """
        输入:
        mask_features: [B, C, H, W] 特征图
        all_gt_labels: List[Tensor] 每个元素是(N_i,)的标签Tensor
        all_gt_masks: List[Tensor] 每个元素是(N_i, H, W)的掩码

        输出:
        features: [Total_Instances, C] 实例特征
        labels: [Total_Instances] 对应标签
        """
        all_features = []
        all_labels = []
        all_gt_labels = [gt_instance.labels for gt_instance in batch_gt_instances]
        all_gt_masks = [gt_instance.masks for gt_instance in batch_gt_instances]

        device = image_features.device
        epsilon = 1e-12  # 更小的epsilon处理极端情况

        for b in range(image_features.shape[0]):
            feats = image_features[b]  # [C, H, W]
            masks = all_gt_masks[b]   # [N_i, H, W]

            if masks.shape[0] == 0:
                continue

            # 将原始掩码调整为特征图尺寸
            resized_masks = F.interpolate(
                masks.float().unsqueeze(1),  # 添加通道维度 [N_i, 1, H_ori, W_ori]
                size=image_features.shape[-2:],
                mode='nearest'  # 保持二值特性
            ).squeeze(1).bool()  # 移除通道维度 [N_i, H_feat, W_feat]
            
            for i, mask in enumerate(resized_masks.unbind(0)):
                # 提取掩码区域特征
                masked_feats = feats[:, mask.bool()]  # [C, K]
                if masked_feats.shape[1] == 0:
                    continue
                # 计算实例平均特征
                instance_feature = masked_feats.mean(dim=1)  # [C]
                all_features.append(instance_feature)
                all_labels.append(all_gt_labels[b][i])

        features = torch.stack(all_features)
        labels = torch.stack(all_labels)
        features = F.normalize(features, p=2, dim=1)

        sim_matrix = torch.mm(features, features.T) / temperature
        # ---- 正样本计算 ----
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~torch.eye(features.shape[0], dtype=torch.bool, device=device)
        pos_counts = pos_mask.sum(dim=1)  # 每个样本的正样本数 [N]
        valid_samples = pos_counts > 0    # 有效样本标识

        # ---- 负样本计算 ----
        neg_mask = (labels.unsqueeze(1) != labels.unsqueeze(0))  # 所有不同类样本
        # 数值稳定处理（分块计算）
        max_values = sim_matrix.max(dim=1, keepdim=True).values.detach()
        sim_matrix = (sim_matrix - max_values) / temperature  # 先做数值稳定再缩放
        # ---- 损失计算 ----
        exp_sim = torch.exp(sim_matrix)
        # 分子：正样本相似度总和（加epsilon防止log(0)）
        numerator = torch.where(
                valid_samples.unsqueeze(1), 
                (exp_sim * pos_mask.float()).sum(dim=1, keepdim=True),
                exp_sim.sum(dim=1, keepdim=True)  # 当无效时numerator=denominator
            ).squeeze(1) + epsilon
        
        # 分母：正样本 + 负样本相似度总和
        denominator = exp_sim.sum(dim=1)  # 包含自身和负样本
        
        # 计算每个样本的对比损失
        losses = -torch.log(numerator / (denominator + epsilon))  # 双重epsilon保护

        # ---- 结果聚合 ----
        if valid_samples.sum() == 0:
            return torch.tensor(0.0, device=device)  # 全batch无有效样本
        
        # 仅计算有效样本的损失
        final_loss = (losses * valid_samples.float()).sum() / (valid_samples.sum().float() + epsilon)
        
        return final_loss

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        # batch SegDataSample to InstanceDataSample
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples)

        # forward
        all_cls_scores, all_mask_preds, mask_features, multi_scale_memorys = self(x, batch_data_samples)

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                   batch_gt_instances, batch_img_metas)
        
        losses['loss_info_nce_3'] = 0.05 * self.visual_representations_info_nce_loss(mask_features, batch_gt_instances)
        assert len(multi_scale_memorys) == 3
        for i in range(len(multi_scale_memorys)):
            losses[f'loss_info_nce_{i}'] = 0.05 * self.visual_representations_info_nce_loss(multi_scale_memorys[i], batch_gt_instances)

        return losses

    def predict(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tuple[Tensor]:
        batch_data_samples = [
            SegDataSample(metainfo=metainfo) for metainfo in batch_img_metas
        ]

        all_cls_scores, all_mask_preds, _, _ = self(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape']
        else:
            size = batch_img_metas[0]['img_shape']
        # upsample mask
        mask_pred_results = F.interpolate(
            mask_pred_results, size=size, mode='bilinear', align_corners=False)
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
        return seg_logits

    def forward(self, x: List[Tensor],
                batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        batch_size = x[0].shape[0]
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
            attn_mask = attn_mask & mask_sum
            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, mask_pred_list, mask_features, multi_scale_memorys


@MODELS.register_module()
class Mask2FormerHeadContrast9(Mask2FormerHeadContrast8):
    '''
    分类改为queries和class/background embeddings的相似度
    非固定匹配
    各层不同类之间特征有loss_info_nce
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cls_embed = CodebookContrastiveHead7(num_classes=self.num_classes, prototypes_per_class=5, embedding_dim=kwargs['feat_channels'])


@MODELS.register_module()
class Mask2FormerHeadContrast10(Mask2FormerHead):
    '''
    query和类别固定匹配
    分类改为query和对应类别class/background embeddings的相似度
    各层不同类之间特征有loss_info_nce
    '''
    def __init__(self, queries_per_class=5, **kwargs):
        super().__init__(**kwargs)
        self.queries_per_class = queries_per_class
        self.cls_embed = CodebookContrastiveHead6(num_classes=self.num_classes, prototypes_per_class=5, embedding_dim=kwargs['feat_channels'])
        assert self.num_queries == self.num_classes * queries_per_class        

    def visual_representations_info_nce_loss(self, image_features, batch_gt_instances, temperature=0.1):
        """
        输入:
        mask_features: [B, C, H, W] 特征图
        all_gt_labels: List[Tensor] 每个元素是(N_i,)的标签Tensor
        all_gt_masks: List[Tensor] 每个元素是(N_i, H, W)的掩码

        输出:
        features: [Total_Instances, C] 实例特征
        labels: [Total_Instances] 对应标签
        """
        all_features = []
        all_labels = []
        all_gt_labels = [gt_instance.labels for gt_instance in batch_gt_instances]
        all_gt_masks = [gt_instance.masks for gt_instance in batch_gt_instances]

        device = image_features.device
        epsilon = 1e-12  # 更小的epsilon处理极端情况

        for b in range(image_features.shape[0]):
            feats = image_features[b]  # [C, H, W]
            masks = all_gt_masks[b]   # [N_i, H, W]

            if masks.shape[0] == 0:
                continue

            # 将原始掩码调整为特征图尺寸
            resized_masks = F.interpolate(
                masks.float().unsqueeze(1),  # 添加通道维度 [N_i, 1, H_ori, W_ori]
                size=image_features.shape[-2:],
                mode='nearest'  # 保持二值特性
            ).squeeze(1).bool()  # 移除通道维度 [N_i, H_feat, W_feat]
            
            for i, mask in enumerate(resized_masks.unbind(0)):
                # 提取掩码区域特征
                masked_feats = feats[:, mask.bool()]  # [C, K]
                if masked_feats.shape[1] == 0:
                    continue
                # 计算实例平均特征
                instance_feature = masked_feats.mean(dim=1)  # [C]
                all_features.append(instance_feature)
                all_labels.append(all_gt_labels[b][i])

        features = torch.stack(all_features)
        labels = torch.stack(all_labels)
        features = F.normalize(features, p=2, dim=1)

        sim_matrix = torch.mm(features, features.T) / temperature
        # ---- 正样本计算 ----
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~torch.eye(features.shape[0], dtype=torch.bool, device=device)
        pos_counts = pos_mask.sum(dim=1)  # 每个样本的正样本数 [N]
        valid_samples = pos_counts > 0    # 有效样本标识

        # ---- 负样本计算 ----
        neg_mask = (labels.unsqueeze(1) != labels.unsqueeze(0))  # 所有不同类样本
        # 数值稳定处理（分块计算）
        max_values = sim_matrix.max(dim=1, keepdim=True).values.detach()
        sim_matrix = (sim_matrix - max_values) / temperature  # 先做数值稳定再缩放
        # ---- 损失计算 ----
        exp_sim = torch.exp(sim_matrix)
        # 分子：正样本相似度总和（加epsilon防止log(0)）
        numerator = torch.where(
                valid_samples.unsqueeze(1), 
                (exp_sim * pos_mask.float()).sum(dim=1, keepdim=True),
                exp_sim.sum(dim=1, keepdim=True)  # 当无效时numerator=denominator
            ).squeeze(1) + epsilon
        
        # 分母：正样本 + 负样本相似度总和
        denominator = exp_sim.sum(dim=1)  # 包含自身和负样本
        
        # 计算每个样本的对比损失
        losses = -torch.log(numerator / (denominator + epsilon))  # 双重epsilon保护

        # ---- 结果聚合 ----
        if valid_samples.sum() == 0:
            return torch.tensor(0.0, device=device)  # 全batch无有效样本
        
        # 仅计算有效样本的损失
        final_loss = (losses * valid_samples.float()).sum() / (valid_samples.sum().float() + epsilon)
        
        return final_loss

    def forward(self, x: List[Tensor],
                batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        '''
        将query_feat作为返回值一并返回
        不同组的queries之间屏蔽attention
        '''
        batch_size = x[0].shape[0]
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        query_feat_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)
        query_feat_list.append(query_feat)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
            attn_mask = attn_mask & mask_sum
            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            # modify
            # 不同组的queries之间屏蔽attention
            indices = torch.arange(self.num_queries).unsqueeze(1).repeat(1, self.num_queries)
            groups = indices // self.queries_per_class
            self_attn_mask = ~(groups == groups.T).to(query_feat.device)
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                self_attn_mask=self_attn_mask,
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
            query_feat_list.append(query_feat)

        return cls_pred_list, mask_pred_list, query_feat_list, mask_features, multi_scale_memorys

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        # batch SegDataSample to InstanceDataSample
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples)

        # forward
        all_cls_scores, all_mask_preds, all_query_feats, mask_features, multi_scale_memorys = self(x, batch_data_samples)

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                   batch_gt_instances, batch_img_metas)
        
        losses['loss_info_nce_3'] = 0.05 * self.visual_representations_info_nce_loss(mask_features, batch_gt_instances)
        assert len(multi_scale_memorys) == 3
        for i in range(len(multi_scale_memorys)):
            losses[f'loss_info_nce_{i}'] = 0.05 * self.visual_representations_info_nce_loss(multi_scale_memorys[i], batch_gt_instances)

        return losses

    def predict(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tuple[Tensor]:
        """Test without augmentaton.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_img_metas (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            test_cfg (ConfigType): Test config.

        Returns:
            Tensor: A tensor of segmentation mask.
        """
        batch_data_samples = [
            SegDataSample(metainfo=metainfo) for metainfo in batch_img_metas
        ]

        all_cls_scores, all_mask_preds, _, _, _ = self(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape']
        else:
            size = batch_img_metas[0]['img_shape']
        # upsample mask
        mask_pred_results = F.interpolate(
            mask_pred_results, size=size, mode='bilinear', align_corners=False)
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
        return seg_logits






from mmdet.models.layers import SinePositionalEncoding
@MODELS.register_module()
class Mask2FormerHeadReliableMatching(Mask2FormerHead):        
    def __init__(self,
                 num_classes,
                 in_channels: List[int],
                 feat_channels: int,
                 out_channels: int,
                 num_things_classes: int = 80,
                 num_stuff_classes: int = 53,
                 num_queries: int = 100,
                 num_transformer_feat_level: int = 3,
                 pixel_decoder: ConfigType = ...,
                 enforce_decoder_input_project: bool = False,
                 transformer_decoder: ConfigType = ...,
                 positional_encoding: ConfigType = dict(
                     num_feats=128, normalize=True),
                 loss_cls: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=2.0,
                     reduction='mean',
                     class_weight=[1.0] * 133 + [0.1]),
                 loss_mask: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=5.0),
                 loss_dice: ConfigType = dict(
                     type='DiceLoss',
                     use_sigmoid=True,
                     activate=True,
                     reduction='mean',
                     naive_dice=True,
                     eps=1.0,
                     loss_weight=5.0),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 align_corners=False,
                 ignore_index=255,
                 **kwargs) -> None:
        super(AnchorFreeHead, self).__init__(init_cfg=init_cfg)
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.layer_cfg.cross_attn_cfg.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        assert pixel_decoder.encoder.layer_cfg. \
            self_attn_cfg.num_levels == num_transformer_feat_level
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels)
        self.pixel_decoder = MODELS.build(pixel_decoder_)
        self.transformer_decoder = Mask2FormerTransformerDecoderReliableMatching(
            **transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims

        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv2d(
                        feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
        self.decoder_positional_encoding = SinePositionalEncoding(
            **positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level,
                                        feat_channels)

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            self.sampler = TASK_UTILS.build(
                self.train_cfg['sampler'], default_args=dict(context=self))
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        self.class_weight = loss_cls.class_weight
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_mask = MODELS.build(loss_mask)
        self.loss_dice = MODELS.build(loss_dice)

        self.num_classes = num_classes
        self.align_corners = align_corners
        self.out_channels = num_classes
        self.ignore_index = ignore_index
        
        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)



class PrototypeBank:
    def __init__(self, num_classes, feat_dim, momentum=0.9):
        self.momentum = momentum
        self.prototypes = nn.Parameter(torch.zeros(num_classes, feat_dim))
        self.class_counts = torch.zeros(num_classes)
        
    def update(self, features, labels):
        with torch.no_grad():
            unique_labels = torch.unique(labels)
            for lbl in unique_labels:
                mask = labels == lbl
                class_feats = features[mask]
                
                # 动量更新原型
                curr_proto = class_feats.mean(dim=0)
                self.prototypes[lbl] = self.momentum * self.prototypes[lbl] + (1 - self.momentum) * curr_proto
                self.class_counts[lbl] += mask.sum().item()
                
    def get_prototypes(self):
        return F.normalize(self.prototypes, p=2, dim=1)

@MODELS.register_module()
class Mask2FormerHeadContrast11(Mask2FormerHead):
    '''
    加入视觉特征的loss info nce
    不同层有不同的prototypes
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prototypes = nn.Parameter(torch.zeros(4, self.num_classes, kwargs['feat_channels']), requires_grad=False)
    
    def visual_representations_info_nce_loss(self, image_features, batch_gt_instances, temperature=0.1, level=0):
        """
        输入:
        mask_features: [B, C, H, W] 特征图
        all_gt_labels: List[Tensor] 每个元素是(N_i,)的标签Tensor
        all_gt_masks: List[Tensor] 每个元素是(N_i, H, W)的掩码

        输出:
        features: [Total_Instances, C] 实例特征
        labels: [Total_Instances] 对应标签
        """
        all_features = []
        all_labels = []
        all_gt_labels = [gt_instance.labels for gt_instance in batch_gt_instances]
        all_gt_masks = [gt_instance.masks for gt_instance in batch_gt_instances]

        device = image_features.device
        epsilon = 1e-12  # 更小的epsilon处理极端情况

        for b in range(image_features.shape[0]):
            feats = image_features[b]  # [C, H, W]
            masks = all_gt_masks[b]   # [N_i, H, W]
            labels = all_gt_labels[b]

            if masks.shape[0] == 0:
                continue

            # 将原始掩码调整为特征图尺寸
            resized_masks = F.interpolate(
                masks.float().unsqueeze(1),  # 添加通道维度 [N_i, 1, H_ori, W_ori]
                size=image_features.shape[-2:],
                mode='nearest'  # 保持二值特性
            ).squeeze(1).bool()  # 移除通道维度 [N_i, H_feat, W_feat]
            
            for i, (lbl, mask) in enumerate(zip(labels, resized_masks)):
                # 提取掩码区域特征
                masked_feats = feats[:, mask.bool()]  # [C, K]
                if masked_feats.shape[1] == 0:
                    continue
                # 计算实例平均特征
                instance_feature = masked_feats.mean(dim=1)  # [C]
                all_features.append(instance_feature)
                all_labels.append(lbl)
                with torch.no_grad():
                    self.prototypes[level][lbl] = 0.9 * self.prototypes[level][lbl] + 0.1 * instance_feature

        features = torch.stack(all_features)
        labels = torch.stack(all_labels)
        features = F.normalize(features, p=2, dim=1)

        sim_matrix = torch.mm(features, features.T) / temperature
        # ---- 正样本计算 ----
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~torch.eye(features.shape[0], dtype=torch.bool, device=device)
        pos_counts = pos_mask.sum(dim=1)  # 每个样本的正样本数 [N]
        valid_samples = pos_counts > 0    # 有效样本标识

        # ---- 负样本计算 ----
        neg_mask = (labels.unsqueeze(1) != labels.unsqueeze(0))  # 所有不同类样本
        # 数值稳定处理（分块计算）
        max_values = sim_matrix.max(dim=1, keepdim=True).values.detach()
        sim_matrix = (sim_matrix - max_values) / temperature  # 先做数值稳定再缩放
        # ---- 损失计算 ----
        exp_sim = torch.exp(sim_matrix)
        # 分子：正样本相似度总和（加epsilon防止log(0)）
        numerator = torch.where(
                valid_samples.unsqueeze(1), 
                (exp_sim * pos_mask.float()).sum(dim=1, keepdim=True),
                exp_sim.sum(dim=1, keepdim=True)  # 当无效时numerator=denominator
            ).squeeze(1) + epsilon
        
        # 分母：正样本 + 负样本相似度总和
        denominator = exp_sim.sum(dim=1)  # 包含自身和负样本
        
        # 计算每个样本的对比损失
        loss_info_nce = -torch.log(numerator / (denominator + epsilon))  # 双重epsilon保护

        # ---- 结果聚合 ----
        if valid_samples.sum() == 0:
            # return torch.tensor(0.0, device=device)  # 全batch无有效样本
            return {f'loss_info_nce_{level}': torch.tensor(0.0, device=device), f'loss_fp_{level}': torch.tensor(0.0, device=device)}
        

        
        # 仅计算有效样本的损失
        loss_info_nce = (loss_info_nce * valid_samples.float()).sum() / (valid_samples.sum().float() + epsilon)
        
        curr_prototypes = F.normalize(self.prototypes[level], p=2, dim=1)
        sim_fp = torch.mm(features, curr_prototypes.T) / temperature  # [N, C]
        targets = F.one_hot(labels, num_classes=self.num_classes).float()
        # 采用交叉熵形式
        loss_fp = - (targets * F.log_softmax(sim_fp, dim=1)).sum(dim=1).mean()
        final_loss = {f'loss_info_nce_{level}': 0.05*loss_info_nce, f'loss_fp_{level}': 0.1*loss_fp}

        return final_loss

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        # batch SegDataSample to InstanceDataSample
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples)

        # forward
        all_cls_scores, all_mask_preds, mask_features, multi_scale_memorys = self(x, batch_data_samples)

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                   batch_gt_instances, batch_img_metas)
        
        losses.update(self.visual_representations_info_nce_loss(mask_features, batch_gt_instances, level=3))
        assert len(multi_scale_memorys) == 3
        for i in range(len(multi_scale_memorys)):
            losses.update(self.visual_representations_info_nce_loss(multi_scale_memorys[i], batch_gt_instances, level=i))

        return losses

    def predict(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tuple[Tensor]:
        batch_data_samples = [
            SegDataSample(metainfo=metainfo) for metainfo in batch_img_metas
        ]

        all_cls_scores, all_mask_preds, _, _ = self(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape']
        else:
            size = batch_img_metas[0]['img_shape']
        # upsample mask
        mask_pred_results = F.interpolate(
            mask_pred_results, size=size, mode='bilinear', align_corners=False)
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
        return seg_logits

    def forward(self, x: List[Tensor],
                batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        batch_size = x[0].shape[0]
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
            attn_mask = attn_mask & mask_sum
            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, mask_pred_list, mask_features, multi_scale_memorys



@MODELS.register_module()
class Mask2FormerHeadContrast12(Mask2FormerHeadContrast6):
    '''
    分类改为queries和class/background embeddings的相似度
    queries和class/background embeddings有loss info nce
    pooled features和class/background embeddings有loss info nce (相比于6的区别)
    正样本：同类中相似度最高的正原型
    负样本：同类背景 + 其他类的所有原型
    '''

    def visual_representations_info_nce_loss(self, image_features, batch_gt_instances, temperature=0.1):
        """
        输入:
        mask_features: [B, C, H, W] 特征图
        all_gt_labels: List[Tensor] 每个元素是(N_i,)的标签Tensor
        all_gt_masks: List[Tensor] 每个元素是(N_i, H, W)的掩码

        输出:
        features: [Total_Instances, C] 实例特征
        labels: [Total_Instances] 对应标签
        """
        all_features = []
        all_labels = []
        all_gt_labels = [gt_instance.labels for gt_instance in batch_gt_instances]
        all_gt_masks = [gt_instance.masks for gt_instance in batch_gt_instances]

        device = image_features.device
        epsilon = 1e-12  # 更小的epsilon处理极端情况

        for b in range(image_features.shape[0]):
            feats = image_features[b]  # [C, H, W]
            masks = all_gt_masks[b]   # [N_i, H, W]

            if masks.shape[0] == 0:
                continue

            # 将原始掩码调整为特征图尺寸
            resized_masks = F.interpolate(
                masks.float().unsqueeze(1),  # 添加通道维度 [N_i, 1, H_ori, W_ori]
                size=image_features.shape[-2:],
                mode='nearest'  # 保持二值特性
            ).squeeze(1).bool()  # 移除通道维度 [N_i, H_feat, W_feat]
            
            for i, mask in enumerate(resized_masks.unbind(0)):
                # 提取掩码区域特征
                masked_feats = feats[:, mask.bool()]  # [C, K]
                if masked_feats.shape[1] == 0:
                    continue
                # 计算实例平均特征
                instance_feature = masked_feats.mean(dim=1)  # [C]
                all_features.append(instance_feature)
                all_labels.append(all_gt_labels[b][i])

        features = torch.stack(all_features)
        labels = torch.stack(all_labels)
        features = F.normalize(features, p=2, dim=1)

        sim_matrix = torch.mm(features, features.T) / temperature
        # ---- 正样本计算 ----
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~torch.eye(features.shape[0], dtype=torch.bool, device=device)
        pos_counts = pos_mask.sum(dim=1)  # 每个样本的正样本数 [N]
        valid_samples = pos_counts > 0    # 有效样本标识

        # ---- 负样本计算 ----
        neg_mask = (labels.unsqueeze(1) != labels.unsqueeze(0))  # 所有不同类样本
        # 数值稳定处理（分块计算）
        max_values = sim_matrix.max(dim=1, keepdim=True).values.detach()
        sim_matrix = (sim_matrix - max_values) / temperature  # 先做数值稳定再缩放
        # ---- 损失计算 ----
        exp_sim = torch.exp(sim_matrix)
        # 分子：正样本相似度总和（加epsilon防止log(0)）
        numerator = torch.where(
                valid_samples.unsqueeze(1), 
                (exp_sim * pos_mask.float()).sum(dim=1, keepdim=True),
                exp_sim.sum(dim=1, keepdim=True)  # 当无效时numerator=denominator
            ).squeeze(1) + epsilon
        
        # 分母：正样本 + 负样本相似度总和
        denominator = exp_sim.sum(dim=1)  # 包含自身和负样本
        
        # 计算每个样本的对比损失
        losses = -torch.log(numerator / (denominator + epsilon))  # 双重epsilon保护

        # ---- 结果聚合 ----
        if valid_samples.sum() == 0:
            return torch.tensor(0.0, device=device)  # 全batch无有效样本
        
        # 仅计算有效样本的损失
        final_loss = (losses * valid_samples.float()).sum() / (valid_samples.sum().float() + epsilon)
        
        return final_loss

    def _forward_head(self, decoder_out: Tensor, mask_feature: Tensor,
                      attn_mask_target_size: Tuple[int, int]) -> Tuple[Tensor]:
        """"
        加入 masked averaged representations作为cls_logits的补充
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        # shape (num_queries, batch_size, c)
        cls_pred = self.cls_embed(decoder_out)
        # shape (num_queries, batch_size, c)
        mask_embed = self.mask_embed(decoder_out)
        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        attn_mask = F.interpolate(
            mask_pred,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return cls_pred, mask_pred, attn_mask
        
    def forward(self, x: List[Tensor],
                batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        '''
        将query_feat作为返回值一并返回
        不同组的queries之间屏蔽attention
        '''
        batch_size = x[0].shape[0]
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        query_feat_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        # shape (num_queries, batch_size, c)
        
        # mask_pred_resize = F.interpolate(mask_pred, size=multi_scale_memorys[1].shape[-2:], mode='bilinear', align_corners=False)
        # mask_weights = mask_pred_resize.sigmoid().detach()
        # mask_weights = mask_weights / (mask_weights.sum(dim=(-1, -2), keepdim=True) + 1e-7)
        # pooled_features = (mask_weights.unsqueeze(2) * multi_scale_memorys[1].unsqueeze(1)).sum(dim=(-1, -2)) # [b, num_queries, h, w] * [b, c, h, w]
        # cls_pred = cls_pred + self.cls_embed(pooled_features)
        # cls_pred = self.cls_embed(pooled_features)

        # print(cls_pred)

        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)
        query_feat_list.append(query_feat)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
            attn_mask = attn_mask & mask_sum
            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            # modify
            # 不同组的queries之间屏蔽attention
            indices = torch.arange(self.num_queries).unsqueeze(1).repeat(1, self.num_queries)
            groups = indices // self.queries_per_class
            self_attn_mask = ~(groups == groups.T).to(query_feat.device)
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                self_attn_mask=self_attn_mask,
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            # cls_pred = cls_pred + self.cls_embed(pooled_features)

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
            query_feat_list.append(query_feat)

        return cls_pred_list, mask_pred_list, query_feat_list, mask_features, multi_scale_memorys


    def predict(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tuple[Tensor]:
        batch_data_samples = [
            SegDataSample(metainfo=metainfo) for metainfo in batch_img_metas
        ]

        all_cls_scores, all_mask_preds, _, _, _ = self(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape']
        else:
            size = batch_img_metas[0]['img_shape']
        # upsample mask
        mask_pred_results = F.interpolate(
            mask_pred_results, size=size, mode='bilinear', align_corners=False)
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
        return seg_logits


    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        # batch SegDataSample to InstanceDataSample
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples)

        # forward
        all_cls_scores, all_mask_preds, all_query_feats, mask_features, multi_scale_memorys = self(x, batch_data_samples)

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds, all_query_feats, multi_scale_memorys[1],
                                   batch_gt_instances, batch_img_metas)

        return losses


    def loss_by_feat(self, all_cls_scores: Tensor, all_mask_preds: Tensor, all_query_feats, image_features,
                     batch_gt_instances: List[InstanceData],
                     batch_img_metas: List[dict]) -> Dict[str, Tensor]:
        '''
        加入loss_codebook
        '''
        num_dec_layers = len(all_cls_scores)
        batch_gt_instances_list = [
            batch_gt_instances for _ in range(num_dec_layers)
        ]
        img_metas_list = [batch_img_metas for _ in range(num_dec_layers)]
        image_features_list = [image_features for _ in range(num_dec_layers)]
        losses_cls, losses_mask, losses_dice, losses_query_info, losses_representation_info = multi_apply(
            self._loss_by_feat_single, all_cls_scores, all_mask_preds, all_query_feats, image_features_list,
            batch_gt_instances_list, img_metas_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        loss_dict['loss_query_info'] = losses_query_info[-1]
        loss_dict['loss_representation_info'] = losses_representation_info[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i, loss_query_info_i, loss_representation_info_i in zip(
                losses_cls[:-1], losses_mask[:-1], losses_dice[:-1], losses_query_info[:-1], losses_representation_info[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            loss_dict[f'd{num_dec_layer}.loss_query_info'] = loss_query_info_i
            loss_dict[f'd{num_dec_layer}.loss_representation_info'] = loss_representation_info_i
            num_dec_layer += 1
        return loss_dict


    def _loss_by_feat_single(self, cls_scores: Tensor, mask_preds: Tensor, query_feats: Tensor, image_feats: Tensor,
                             batch_gt_instances: List[InstanceData],
                             batch_img_metas: List[dict]) -> Tuple[Tensor]:
        '''
        加入loss_codebook的计算
        '''
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         avg_factor) = self.get_targets(cls_scores_list, mask_preds_list,
                                        batch_gt_instances, batch_img_metas)
        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)
        
        # modify
        # 将对应的bg embed和cls embed选出进行更新
        labels_keep = labels.clone()
        mask_preds_resize = F.interpolate(mask_preds, size=image_feats.shape[-2:], mode='bilinear', align_corners=False)
        masked_feats = mask_preds_resize.unsqueeze(2) * image_feats.unsqueeze(1)    # [B, num_masks, C, H, W]


        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)
        

        # modify
        masked_pooled_feats = point_sample(masked_feats[mask_weights > 0], points_coords).sum(dim=-1) / (mask_point_preds.sum(dim=-1) + 1e-7)
        loss_query_info, loss_representation_info = self.cls_embed.loss(query_feats, masked_pooled_feats, labels_keep)

        return loss_cls, loss_mask, loss_dice, loss_query_info, loss_representation_info





@MODELS.register_module()
class Mask2FormerHeadContrast13(Mask2FormerHeadContrast6):
    '''
    在6的基础上添加max_iou_assigner
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_iou_assigner = TASK_UTILS.build(dict(type='GroupMaxIoUAssigner', iou_threshold=0.7))
    
    def _get_targets_single(self, cls_score: Tensor, mask_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> Tuple[Tensor]:
        '''
        mask全监督
        '''
        gt_labels = gt_instances.labels
        gt_masks = gt_instances.masks
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2),
                                  device=cls_score.device)
        # shape (num_queries, num_points)
        mask_points_pred = point_sample(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1,
                                                        1)).squeeze(1)
        # shape (num_gts, num_points)
        gt_points_masks = point_sample(
            gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1,
                                                               1)).squeeze(1)

        sampled_gt_instances = InstanceData(
            labels=gt_labels, masks=gt_points_masks)
        sampled_pred_instances = InstanceData(
            scores=cls_score, masks=mask_points_pred)
        # assign and sample
        assign_result = self.assigner.assign(
            pred_instances=sampled_pred_instances,
            gt_instances=sampled_gt_instances,
            img_meta=img_meta)
        pred_instances = InstanceData(scores=cls_score, masks=mask_pred)
        sampling_result = self.sampler.sample(
            assign_result=assign_result,
            pred_instances=pred_instances,
            gt_instances=gt_instances)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        labels = gt_labels.new_full((self.num_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((self.num_queries, ))

        # mask target

        if len(sampled_gt_instances) > 0:
            max_iou_assign_result, overlaps_max = self.max_iou_assigner.assign(
                pred_instances=sampled_pred_instances,
                gt_instances=sampled_gt_instances,
                img_meta=img_meta)
            max_iou_sampling_result = self.sampler.sample(
                assign_result=max_iou_assign_result,
                pred_instances=pred_instances,
                gt_instances=gt_instances)
        
        mask_targets = gt_masks.new_zeros((self.num_queries, *gt_masks.shape[-2:]))
        mask_weights = mask_pred.new_ones((self.num_queries, )) * 0.7

        if len(sampled_gt_instances) > 0:
            mask_targets[max_iou_sampling_result.pos_inds] = gt_masks[max_iou_sampling_result.pos_assigned_gt_inds]
            mask_weights[max_iou_sampling_result.pos_inds] = overlaps_max[max_iou_sampling_result.pos_inds]
        
        mask_targets[pos_inds] = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights[pos_inds] = 1.0

        sampling_result.avg_factor = mask_weights.sum()

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds, sampling_result)







@MODELS.register_module()
class Mask2FormerHeadContrast100(Mask2FormerHeadContrast):
    '''
    分类改为 query 和 class/background embeddings的相似度 weighted averaged pooled image representations 和 class/background embeddings at corresponding layers的相似度
    '''
    def __init__():
        pass






@MODELS.register_module()
class Mask2FormerHeadLBP(Mask2FormerHead):
    def __init__(self,
                 num_classes,
                 in_channels: List[int],
                 feat_channels: int,
                 out_channels: int,
                 num_things_classes: int = 80,
                 num_stuff_classes: int = 53,
                 num_queries: int = 100,
                 num_transformer_feat_level: int = 3,
                 pixel_decoder: ConfigType = ...,
                 enforce_decoder_input_project: bool = False,
                 transformer_decoder: ConfigType = ...,
                 positional_encoding: ConfigType = dict(
                     num_feats=128, normalize=True),
                 loss_cls: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=2.0,
                     reduction='mean',
                     class_weight=[1.0] * 133 + [0.1]),
                 loss_mask: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=5.0),
                 loss_dice: ConfigType = dict(
                     type='DiceLoss',
                     use_sigmoid=True,
                     activate=True,
                     reduction='mean',
                     naive_dice=True,
                     eps=1.0,
                     loss_weight=5.0),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 align_corners=False,
                 ignore_index=255,
                 **kwargs) -> None:
        super(AnchorFreeHead, self).__init__(init_cfg=init_cfg)
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.layer_cfg.cross_attn_cfg.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        assert pixel_decoder.encoder.layer_cfg. \
            self_attn_cfg.num_levels == num_transformer_feat_level
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels)
        self.pixel_decoder = MODELS.build(pixel_decoder_)
        self.transformer_decoder = Mask2FormerTransformerDecoderLBP(
            **transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims

        self.decoder_input_projs = ModuleList()
        self.decoder_input_projs_lbp = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv2d(
                        feat_channels, self.decoder_embed_dims, kernel_size=1))
                self.decoder_input_projs_lbp.append(
                    Conv2d(
                        feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
                self.decoder_input_projs_lbp.append(nn.Identity())
        self.decoder_positional_encoding = SinePositionalEncoding(
            **positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level,
                                        feat_channels)

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            self.sampler = TASK_UTILS.build(
                self.train_cfg['sampler'], default_args=dict(context=self))
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        self.class_weight = loss_cls.class_weight
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_mask = MODELS.build(loss_mask)
        self.loss_dice = MODELS.build(loss_dice)

        self.num_classes = num_classes
        self.align_corners = align_corners
        self.out_channels = num_classes
        self.ignore_index = ignore_index
        
        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
    
    
    def forward(self, x: List[Tensor], x_lbp: List[Tensor],
                batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        batch_size = x[0].shape[0]
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_inputs_lbp = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            decoder_input_lbp = self.decoder_input_projs[i](x_lbp[-i-1])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed

            decoder_input_lbp = decoder_input_lbp.flatten(2).permute(0, 2, 1)

            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_inputs_lbp.append(decoder_input_lbp)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
            attn_mask = attn_mask & mask_sum
            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                key_lbp=decoder_inputs_lbp[level_idx],
                value_lbp=decoder_inputs_lbp[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, mask_pred_list



@MODELS.register_module()
class Mask2FormerHeadLBP_D(Mask2FormerHeadLBP):
    def forward(self, x: List[Tensor], x_lbp: List[Tensor],
                batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        batch_size = x[0].shape[0]
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_inputs_lbp = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            decoder_input_lbp = self.decoder_input_projs[i](x_lbp[-i-1])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed

            decoder_input_lbp = decoder_input_lbp.flatten(2).permute(0, 2, 1)

            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_inputs_lbp.append(decoder_input_lbp)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
            attn_mask = attn_mask & mask_sum
            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                key_lbp=decoder_inputs_lbp[level_idx],
                value_lbp=decoder_inputs_lbp[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, mask_pred_list

    def loss(self, x: Tuple[Tensor], x_lbp: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        # batch SegDataSample to InstanceDataSample
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples)

        # forward
        all_cls_scores, all_mask_preds = self(x, x_lbp, batch_data_samples)

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                   batch_gt_instances, batch_img_metas)

        return losses

    def predict(self, x: Tuple[Tensor], x_lbp: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tuple[Tensor]:
        batch_data_samples = [
            SegDataSample(metainfo=metainfo) for metainfo in batch_img_metas
        ]

        all_cls_scores, all_mask_preds = self(x, x_lbp, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape']
        else:
            size = batch_img_metas[0]['img_shape']
        # upsample mask
        mask_pred_results = F.interpolate(
            mask_pred_results, size=size, mode='bilinear', align_corners=False)
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
        return seg_logits


@MODELS.register_module()
class EncoderDecoderLBP(EncoderDecoder):
    def __init__(self,
                 backbone: ConfigType,
                 backbone_lbp: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 neck_lbp: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super(EncoderDecoder, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        self.backbone_lbp = MODELS.build(backbone_lbp)
        if neck is not None:
            self.neck = MODELS.build(neck)
        if neck_lbp is not None:
            self.neck_lbp = MODELS.build(neck_lbp)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(inputs)
        inputs_lbp = torch.stack([img_meta['lbp'] for img_meta in batch_img_metas]).to(inputs.device)
        x_lbp = self.backbone_lbp(inputs_lbp)
        x_lbp = self.neck_lbp(x_lbp)
        seg_logits = self.decode_head.predict(x, x_lbp, batch_img_metas,
                                              self.test_cfg)

        return seg_logits

    def _decode_head_forward_train(self, inputs: List[Tensor], inputs_lbp: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, inputs_lbp, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor], inputs_lbp: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, inputs_lbp, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(inputs)

        batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        inputs_lbp = torch.stack([img_meta['lbp'] for img_meta in batch_img_metas]).to(inputs.device)
        x_lbp = self.backbone_lbp(inputs_lbp)
        x_lbp = self.neck_lbp(x_lbp)
        losses = dict()

        loss_decode = self._decode_head_forward_train(x, x_lbp, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, x_lbp, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)

        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
        return self.decode_head.forward(x)
