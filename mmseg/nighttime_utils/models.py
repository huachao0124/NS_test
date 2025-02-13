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

from mmdet.models.task_modules.assigners.assign_result import AssignResult
from mmseg.models.assigners.base_assigner import BaseAssigner
from .utils import compute_error_matrix, KeysRecorder, o2o_mask_dice


from mmseg.models.decode_heads import Mask2FormerHead
from mmseg.structures.seg_data_sample import SegDataSample

from mmcv.cnn import Conv2d
from mmcv.ops import point_sample
from mmengine.model import ModuleList, caffe2_xavier_init
from mmdet.models.utils import get_uncertain_point_coords_with_randomness



@MODELS.register_module()
class EncoderDecoderAnalysis(EncoderDecoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.assigner = TASK_UTILS.build(dict(type='MaskMaxIoUAssigner'))
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
        mask_pred = mask_pred_results.sigmoid()

        # mask = cls_score < 0.3
        # cls_score[mask] = 0

        # modify
        # mask_pred_binary = (mask_pred >= 0.5).long()
        mask_pred_binary = mask_pred.clone()
        ems, matched_labels, matched_masks = self.get_targets(mask_cls_results, mask_pred_binary, batch_gt_instances, batch_img_metas)

        # cls_score = torch.zeros_like(mask_cls_results).scatter_(dim=-1, index=matched_labels.unsqueeze(-1), value=1)[..., :-1]
        mask_pred = matched_masks.float()

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
        matched_labels = gt_labels.new_full((self.decode_head.num_queries, ),
                                    self.decode_head.num_classes,
                                    dtype=torch.long)
        matched_labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]

        # pred_labels = cls_score[..., :-1].max(dim=-1)[1]
        pred_labels = cls_score.max(dim=-1)[1]

        em = compute_error_matrix(matched_labels, pred_labels, cls_score.shape[-1])

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