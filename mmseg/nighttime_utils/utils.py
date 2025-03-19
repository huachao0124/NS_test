import logging
from typing import List, Optional, Union, Sequence, Dict, Tuple, Any

import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import MMLogger, print_log
from torch import Tensor

from mmseg.registry import MODELS, TASK_UTILS, METRICS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)

from mmseg.models.segmentors import EncoderDecoder


import copy

import torch
from mmengine.structures import InstanceData

from mmdet.models.task_modules.assigners.assign_result import AssignResult
from mmseg.models.assigners.base_assigner import BaseAssigner


from mmengine.evaluator import BaseMetric
import numpy as np
from prettytable import PrettyTable
from collections import OrderedDict
import pandas as pd
from scipy.optimize import linear_sum_assignment
from mmengine import ConfigDict
from mmengine.dist import is_main_process
from mmengine.utils import mkdir_or_exist
import os.path as osp
from mmdet.models.task_modules.assigners.match_cost import BaseMatchCost
import math
import warnings
from mmcv.cnn.bricks.drop import build_dropout

from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine.model import BaseModule, ModuleList

from mmdet.models.layers.transformer.detr_layers import DetrTransformerDecoder, DetrTransformerDecoderLayer
from mmdet.models.layers.transformer.mask2former_layers import Mask2FormerTransformerDecoderLayer


def pairwise_mask_iou(masks1: torch.Tensor, masks2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # 展平为二维张量：[N, H*W] 和 [M, H*W]
    masks1_flat = masks1.view(masks1.size(0), -1).float()
    masks2_flat = masks2.view(masks2.size(0), -1).float()
    
    # 计算交集：[N, M]
    intersection = torch.mm(masks1_flat, masks2_flat.t())
    
    # 计算每个mask的面积：[N] 和 [M]
    area1 = masks1_flat.sum(dim=1)
    area2 = masks2_flat.sum(dim=1)
    
    # 计算并集：area1[:, None] + area2[None, :] - intersection
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection
    
    # 处理可能的零并集（例如两个空mask）
    iou = intersection / (union + eps)
    
    return iou

def pairwise_mask_dice(mask_preds: Tensor, gt_masks: Tensor) -> Tensor:
    eps = 1e-3
    naive_dice = True

    mask_preds = mask_preds.flatten(1)
    gt_masks = gt_masks.flatten(1).float()
    numerator = 2 * torch.einsum('nc,mc->nm', mask_preds.float(), gt_masks.float())
    if naive_dice:
        denominator = mask_preds.sum(-1)[:, None] + \
                        gt_masks.sum(-1)[None, :]
    else:
        denominator = mask_preds.pow(2).sum(1)[:, None] + \
                        gt_masks.pow(2).sum(1)[None, :]
    loss = (numerator + eps) / (denominator + eps)
    return loss

def o2o_mask_dice(mask_preds: Tensor, gt_masks: Tensor) -> Tensor:
    eps = 1e-3
    naive_dice = True

    mask_preds = mask_preds.flatten(1)
    gt_masks = gt_masks.flatten(1).float()

    a = torch.sum(mask_preds * gt_masks, 1)
    if naive_dice:
        b = torch.sum(mask_preds, 1)
        c = torch.sum(gt_masks, 1)
        d = (2 * a + eps) / (b + c + eps)
    else:
        b = torch.sum(mask_preds * mask_preds, 1) + eps
        c = torch.sum(gt_masks * gt_masks, 1) + eps
        d = (2 * a) / (b + c)
    loss = 1 - d

    return loss

def compute_error_matrix(
    matched_labels: torch.Tensor,  # 真实标签 (形状: [N])
    pred_labels: torch.Tensor,     # 预测标签 (形状: [N])
    num_classes: int,              # 类别总数 (C)
    ignore_diagonal: bool = False  # 是否排除正确分类的对角线
) -> torch.Tensor:
    assert matched_labels.shape == pred_labels.shape
    
    # 确定所有标签均在有效范围内
    valid_mask = (matched_labels >= 0) & (pred_labels >= 0) \
                & (matched_labels < num_classes) \
                & (pred_labels < num_classes)
    if not torch.all(valid_mask):
        print("WARNING: Labels exist out-of-range values. Ensure labels are in [0, num_classes-1].")
    
    # 构建混淆矩阵
    cm_indices = matched_labels * num_classes + pred_labels
    cm = torch.bincount(
        cm_indices, 
        minlength=num_classes**2
    ).reshape(num_classes, num_classes)
    
    return cm


@TASK_UTILS.register_module()
class MaskMaxIoUAssigner(BaseAssigner):

    def __init__(self, iou_threshold=0.3):
        self.iou_threshold = iou_threshold

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:
       
        pred_masks = pred_instances.masks
        gt_masks = gt_instances.masks
        gt_labels = gt_instances.labels

        num_gts, num_preds = len(gt_instances), len(pred_instances)
        device = gt_labels.device

        # 1. assign -1 by default
        assigned_gt_inds = torch.full((num_preds, ),
                                      -1,
                                      dtype=torch.long,
                                      device=device)
        assigned_labels = torch.full((num_preds, ),
                                     -1,
                                     dtype=torch.long,
                                     device=device)


        # overlaps = pairwise_mask_iou(pred_masks, gt_masks)
        overlaps = pairwise_mask_dice(pred_masks, gt_masks)

        overlaps_max, indices = torch.max(overlaps, dim=1)

        matched_row_inds = torch.arange(pred_masks.size(0)).to(device)
        matched_col_inds = indices.to(device)
        matched_row_inds = matched_row_inds.to(device)
        matched_col_inds = matched_col_inds.to(device)
        
        mask = overlaps_max >= self.iou_threshold
        matched_row_inds = matched_row_inds[mask]
        matched_col_inds = matched_col_inds[mask]


        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels), overlaps_max


@TASK_UTILS.register_module()
class GroupMaxIoUAssigner(BaseAssigner):

    def __init__(self, iou_threshold=0.3):
        self.iou_threshold = iou_threshold

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:
       
        pred_masks = pred_instances.masks
        gt_masks = gt_instances.masks
        gt_labels = gt_instances.labels

        num_gts, num_preds = len(gt_instances), len(pred_instances)
        device = gt_labels.device

        # 1. assign -1 by default
        assigned_gt_inds = torch.full((num_preds, ),
                                      -1,
                                      dtype=torch.long,
                                      device=device)
        assigned_labels = torch.full((num_preds, ),
                                     -1,
                                     dtype=torch.long,
                                     device=device)


        # overlaps = pairwise_mask_iou(pred_masks, gt_masks)
        overlaps = pairwise_mask_dice(pred_masks, gt_masks)
        for i in range(19):
            overlaps[i*5:(i+1)*5, gt_labels != i] -= 10000

        overlaps_max, indices = torch.max(overlaps, dim=1)

        matched_row_inds = torch.arange(pred_masks.size(0)).to(device)
        matched_col_inds = indices.to(device)
        matched_row_inds = matched_row_inds.to(device)
        matched_col_inds = matched_col_inds.to(device)

        mask = overlaps_max >= self.iou_threshold
        matched_row_inds = matched_row_inds[mask]
        matched_col_inds = matched_col_inds[mask]


        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels), overlaps_max


@TASK_UTILS.register_module()
class MixedHungarianAssigner(BaseAssigner):
    """Computes 1-to-k matching between ground truth and predictions.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of some components.
    For DETR the costs are weighted sum of classification cost, regression L1
    cost and regression iou cost. The targets don't include the no_object, so
    generally there are more predictions than targets. After the 1-to-k
    gt-pred matching, the un-matched are treated as backgrounds. Thus
    each query prediction will be assigned with `0` or a positive integer
    indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        match_costs (:obj:`ConfigDict` or dict or \
            List[Union[:obj:`ConfigDict`, dict]]): Match cost configs.
    """

    def __init__(
        self, match_costs: Union[List[Union[dict, ConfigDict]], dict,
                                 ConfigDict]
    ) -> None:

        if isinstance(match_costs, dict):
            match_costs = [match_costs]
        elif isinstance(match_costs, list):
            assert len(match_costs) > 0, \
                'match_costs must not be a empty list.'

        self.match_costs = [
            TASK_UTILS.build(match_cost) for match_cost in match_costs
        ]

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               img_meta: Optional[dict] = None,
               k: int = 1,
               **kwargs) -> AssignResult:
        """Computes 1-to-k gt-pred matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. Assign every prediction to -1.
        2. Compute the weighted costs, each cost has shape
            (num_preds, num_gts).
        3. Update k according to num_preds and num_gts, then repeat
            costs k times to shape: (num_preds, k * num_gts), so that each
            gt will match k predictions.
        4. Do Hungarian matching on CPU based on the costs.
        5. Assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places. It may includes ``masks``, with shape
                (n, h, w) or (n, l).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                ``labels``, with shape (k, ) and ``masks``, with shape
                (k, h, w) or (k, l).
            img_meta (dict): Image information for one image.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert isinstance(gt_instances.labels, Tensor)
        num_gts, num_preds = len(gt_instances), len(pred_instances)
        gt_labels = gt_instances.labels
        device = gt_labels.device

        # 1. Assign -1 by default.
        assigned_gt_inds = torch.full((num_preds, ),
                                      -1,
                                      dtype=torch.long,
                                      device=device)
        assigned_labels = torch.full((num_preds, ),
                                     -1,
                                     dtype=torch.long,
                                     device=device)

        if num_gts == 0 or num_preds == 0:
            # No ground truth or boxes, return empty assignment.
            if num_gts == 0:
                # No ground truth, assign all to background.
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=None,
                labels=assigned_labels)

        # 2. Compute weighted costs.
        cost_list = []
        for match_cost in self.match_costs:
            cost = match_cost(
                pred_instances=pred_instances,
                gt_instances=gt_instances,
                img_meta=img_meta)
            cost_list.append(cost)
        cost = torch.stack(cost_list).sum(dim=0)

        # 3. Update k according to num_preds and num_gts,  then
        #   repeat the ground truth k times to perform 1-to-k gt-pred
        #   matching. For example, if num_preds = 900, num_gts = 3, then
        #   there are only 3 gt-pred pairs in sum for 1-1 matching.
        #   However, for 1-k gt-pred matching, if k = 4, then each
        #   gt is assigned 4 unique predictions, so there would be 12
        #   gt-pred pairs in sum.
        k = max(1, min(k, num_preds // num_gts))
        cost = cost.repeat(1, k)

        # 4. Do Hungarian matching on CPU using linear_sum_assignment.
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')

        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(device)

        matched_col_inds = matched_col_inds % num_gts
        # 5. Assign backgrounds and foregrounds.
        # Assign all indices to backgrounds first.
        assigned_gt_inds[:] = 0
        # Assign foregrounds based on matching results.
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        assign_result = AssignResult(
            num_gts=k * num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels)

        return assign_result


@METRICS.register_module()
class BlankMetric(BaseMetric):
    def __init__(self,
                 ignore_index: int = 255,
                 nan_to_num: Optional[int] = None,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.nan_to_num = nan_to_num
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            self.results.append((data_sample['error_matrix']['data'].cpu().numpy(), data_sample['gt_sem_seg']['data'].cpu().numpy()))
        
    
    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        
        results = tuple(zip(*results))

        for k in results[0]:
            print(k.shape)

        ems = np.stack(results[0])
        ems_sum = ems.sum(axis=0).squeeze()

        

        print(ems_sum)

        labels = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
          'traffic light', 'traffic sign', 'vegetation', 'terrain',
          'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
          'motorcycle', 'bicycle', 'background']

        
        df = pd.DataFrame(ems_sum, index=labels[:ems_sum.shape[0]], columns=labels[:ems_sum.shape[0]])
        df.to_csv('error_matrix.csv')


        # np.savetxt('error_matrix_2.txt', ems_sum.squeeze(), fmt='%5d')

        prc_auc = 0
        fpr = 0
        roc_auc = 0

        # summary
        metrics = dict()
        for key, val in zip(('AUPRC', 'FPR@95TPR', 'AUROC'), (prc_auc, fpr, roc_auc)):
            metrics[key] = np.round(val * 100, 2)
        metrics = OrderedDict(metrics)
        metrics.update({'Dataset': 'RoadAnomaly'})
        metrics.move_to_end('Dataset', last=False)
        class_table_data = PrettyTable()
        for key, val in metrics.items():
            class_table_data.add_column(key, [val])

        print_log('anomaly segmentation results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics


class KeysRecorder:
    """Wrap object to record its `__getitem__` keys in the history.

    Args:
        obj (object): Any object that supports `__getitem__`.
        keys (List): List of keys already recorded. Default to None.
    """

    def __init__(self, obj: Any, keys: Optional[List[Any]] = None) -> None:
        self.obj = obj

        if keys is None:
            keys = []
        self.keys = keys

    def __getitem__(self, key: Any) -> 'KeysRecorder':
        """Wrap method `__getitem__`  to record its keys.

        Args:
            key: Key that is passed to the object.

        Returns:
            result (KeysRecorder): KeysRecorder instance that wraps sub_obj.
        """
        sub_obj = self.obj.__getitem__(key)
        keys = self.keys.copy()
        keys.append(key)
        # Create a KeysRecorder instance from the sub_obj.
        result = KeysRecorder(sub_obj, keys)
        return result


class CodebookContrastiveHead(nn.Module):
    '''
    19个class embeddings + 1个background embedding
    '''
    def __init__(self, num_classes=19, queries_per_class=5, embedding_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.queries_per_class = queries_per_class
        self.total_queries = num_classes * queries_per_class
        
        # 可学习的class embeddings (包含background)
        self.class_embeddings = nn.Embedding(num_classes + 1, embedding_dim)
        nn.init.normal_(self.class_embeddings.weight, std=0.02)

        # # 为每个query确定对应的class索引
        # if class_indices is None:  # 固定分配模式
        #     class_indices = torch.arange(0, self.num_classes)
        #     class_indices = class_indices.repeat_interleave(self.queries_per_class)
        #     self.class_indices = class_indices.view(1, -1)
        # else:
        #     self.class_indices = None

        self.register_buffer(
            "class_indices",
            torch.arange(num_classes).repeat_interleave(queries_per_class).view(1, -1),
            persistent=False
        )

    def forward(self, query_features):
        """
        query_features: [batch_size, num_queries, embed_dim]
        输出: [batch_size, num_queries, 2]
        """
        batch_size, num_queries, _ = query_features.shape
        
        # 获取对应类的embedding [batch, num_queries, embed_dim]
        class_emb = self.class_embeddings(self.class_indices.expand(batch_size, -1))
        
        # 获取background embedding [1, 1, embed_dim]
        bg_emb = self.class_embeddings(
            torch.full((1,), self.num_classes, device=query_features.device)
        ).unsqueeze(0)
        
        # 计算相似度
        query_features = F.normalize(query_features, p=2, dim=2)
        class_emb = F.normalize(class_emb, p=2, dim=2)
        bg_emb = F.normalize(bg_emb, p=2, dim=2)
        class_sim = torch.einsum('bqd,bqd->bq', query_features, class_emb).unsqueeze(-1)  # [B,Q,1]
        bg_sim = torch.einsum('bqd,bqd->bq', query_features, bg_emb).unsqueeze(-1)         # [B,Q,1]
        
        # logits = torch.cat([class_sim, bg_sim], dim=-1)  # [B, Q, 2]

        logits = torch.full(
            (batch_size, num_queries, self.num_classes + 1),
            float('-inf'),
            device=query_features.device
        )
        # 填充正类位置（需要处理batch维度）
        logits.scatter_(
            dim=2,
            index=self.class_indices.view(1, -1, 1).expand(batch_size, -1, 1),
            src=class_sim
        )
        # 填充background位置（共享最后一列）
        logits[:, :, self.num_classes] = bg_sim.squeeze(-1)

        return logits


class CodebookContrastiveHead2(nn.Module):
    '''
    19个class embeddings + 19个background embedding
    '''
    def __init__(self, num_classes=19, queries_per_class=5, embedding_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.queries_per_class = queries_per_class
        self.total_queries = num_classes * queries_per_class
        
        # 可学习的class embeddings (包含background)
        self.class_embeddings = nn.Embedding(num_classes * 2, embedding_dim)
        nn.init.normal_(self.class_embeddings.weight, std=0.02)

        # # 为每个query确定对应的class索引
        # if class_indices is None:  # 固定分配模式
        #     class_indices = torch.arange(0, self.num_classes)
        #     class_indices = class_indices.repeat_interleave(self.queries_per_class)
        #     self.class_indices = class_indices.view(1, -1)
        # else:
        #     self.class_indices = None

        self.register_buffer(
            "class_indices",
            torch.arange(num_classes).repeat_interleave(queries_per_class).view(1, -1),
            persistent=False
        )

    def forward(self, query_features):
        """
        query_features: [batch_size, num_queries, embed_dim]
        输出: [batch_size, num_queries, 2]
        """
        batch_size, num_queries, _ = query_features.shape
        
        # 获取对应类的embedding [batch, num_queries, embed_dim]
        class_emb = self.class_embeddings(self.class_indices.expand(batch_size, -1))
        # 获取background embedding [batch, num_queries, embed_dim]
        bg_emb = self.class_embeddings(self.class_indices.expand(batch_size, -1) + self.num_classes)
        
        # 计算相似度
        
        class_sim = torch.einsum('bqd,bqd->bq', query_features, class_emb).unsqueeze(-1)  # [B,Q,1]
        bg_sim = torch.einsum('bqd,bqd->bq', query_features, bg_emb).unsqueeze(-1)         # [B,Q,1]
        
        # logits = torch.cat([class_sim, bg_sim], dim=-1)  # [B, Q, 2]

        logits = torch.full(
            (batch_size, num_queries, self.num_classes + 1),
            float('-inf'),
            device=query_features.device
        )
        # 填充正类位置（需要处理batch维度）
        logits.scatter_(
            dim=2,
            index=self.class_indices.view(1, -1, 1).expand(batch_size, -1, 1),
            src=class_sim
        )
        # 填充background位置（共享最后一列）
        logits[:, :, self.num_classes] = bg_sim.squeeze(-1)

        return logits


class CodebookContrastiveHead3(nn.Module):
    '''
    19个class embeddings (for weighted averaged mask features) + 19个background embeddings ((for weighted averaged mask features)
    '''
    def __init__(self, num_classes=19, queries_per_class=5, embedding_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.queries_per_class = queries_per_class
        self.total_queries = num_classes * queries_per_class
        
        # 可学习的class embeddings (包含background)
        self.class_embeddings = nn.Embedding(num_classes * 2, embedding_dim)
        nn.init.normal_(self.class_embeddings.weight, std=0.02)

        # # 为每个query确定对应的class索引
        # if class_indices is None:  # 固定分配模式
        #     class_indices = torch.arange(0, self.num_classes)
        #     class_indices = class_indices.repeat_interleave(self.queries_per_class)
        #     self.class_indices = class_indices.view(1, -1)
        # else:
        #     self.class_indices = None

        self.register_buffer(
            "class_indices",
            torch.arange(num_classes).repeat_interleave(queries_per_class).view(1, -1),
            persistent=False
        )

    def forward(self, pooled_features, is_training=True):
        """
        query_features: [batch_size, num_queries, embed_dim]
        输出: [batch_size, num_queries, 2]
        """
        batch_size, num_queries, _ = pooled_features.shape
        

        if is_training:
            # 获取对应类的embedding [batch, num_queries, embed_dim]
            class_emb = self.class_embeddings(self.class_indices.expand(batch_size, -1))
            # 获取background embedding [batch, num_queries, embed_dim]
            bg_emb = self.class_embeddings(self.class_indices.expand(batch_size, -1) + self.num_classes)
            
            # 计算相似度  
            class_sim = torch.einsum('bqd,bqd->bq', pooled_features, class_emb).unsqueeze(-1)  # [B,Q,1]
            bg_sim = torch.einsum('bqd,bqd->bq', pooled_features, bg_emb).unsqueeze(-1)         # [B,Q,1]
            
            # logits = torch.cat([class_sim, bg_sim], dim=-1)  # [B, Q, 2]

            logits = torch.full(
                (batch_size, num_queries, self.num_classes + 1),
                float('-inf'),
                device=pooled_features.device
            )
            # 填充正类位置（需要处理batch维度）
            logits.scatter_(
                dim=2,
                index=self.class_indices.view(1, -1, 1).expand(batch_size, -1, 1),
                src=class_sim
            )
            # 填充background位置（共享最后一列）
            logits[:, :, self.num_classes] = bg_sim.squeeze(-1)
        else:
            logits = torch.einsum('bqd,nd->bqn', pooled_features, self.class_embeddings.weight[:self.num_classes])

        return logits


class CodebookContrastiveHead6(nn.Module):
    '''
    每个类有5个prototypes和1个background embedding
    '''
    def __init__(self, num_classes=19, queries_per_class=5, prototypes_per_class=5, embedding_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.queries_per_class = queries_per_class
        self.prototypes_per_class = prototypes_per_class
        
        self.class_embeddings = nn.Embedding(
            num_embeddings=num_classes * (prototypes_per_class + 1),
            embedding_dim=embedding_dim
        )

        # --- 获取每个query对应的class所有prototypes ---
        # class_indices: [1, num_queries]
        self.register_buffer(
            "class_indices",
            torch.arange(num_classes).repeat_interleave(queries_per_class).view(1, -1),
            persistent=False
        )   # [1, num_queries]
        # 每个类的prototype起始索引 [1, num_classes*queries_per_class]
        proto_start = self.class_indices * (self.prototypes_per_class + 1)  # [1, num_queries]
        offset = torch.arange(prototypes_per_class + 1) # [prototypes_per_class + 1]
        self.register_buffer(
            "prototype_indices",
            proto_start.unsqueeze(-1) + offset.unsqueeze(0).unsqueeze(0),
            persistent=False
        )   # [1, num_queries, 1] + [1, 1, prototypes_per_class + 1] = [1, num_queries, prototypes_per_class + 1]

        # self.class_embeddings.weight.requires_grad_(False)

    def forward(self, query_features):
        """
        Args:
            query_features: [batch_size, num_queries, embed_dim]
        Returns:
            logits: [batch_size, num_queries, num_classes + 1]
        """
        batch_size, num_queries, _ = query_features.shape
        
        # --- 获取每个query对应的class所有prototypes ---
        all_indices = self.prototype_indices
        proto_features = self.class_embeddings(all_indices).expand(batch_size, -1, -1, -1)  # [B, Q, K+1, D]

        # --- 计算相似度 ---
        # 正prototypes相似度 [B, Q, K]
        pos_sim = torch.einsum('bqd,bqkd->bqk', 
                              query_features, 
                              proto_features[:, :, :-1, :])
        # 背景相似度 [B, Q, 1]
        bg_sim = torch.einsum('bqd,bqkd->bqk',
                             query_features,
                             proto_features[:, :, -1:, :])
        
        # --- 构造logits矩阵 ---
        # 初始化全-inf [B, Q, C+1]
        logits = torch.full((batch_size, num_queries, self.num_classes + 1),
                          float('-inf'),
                          device=query_features.device)
        
        # 填充正类相似度（取每个类多个prototype的最大值）
        class_indices = self.class_indices.expand(batch_size, -1)  # [B, Q]
        logits.scatter_(2, class_indices.unsqueeze(-1), pos_sim.max(dim=-1)[0].unsqueeze(-1))
        
        # 填充背景相似度到最后一列
        logits[:, :, -1:] = bg_sim

        return logits
    
    def loss_only_qp(self, query_features, matched_labels):
        batch_size, num_queries, feat_dim = query_features.shape
        device = query_features.device

        # --- 分离背景Query ---
        is_bg = (matched_labels == self.num_classes)    # 背景标记, [batch_size, num_queries]
        valid_labels = matched_labels[~is_bg]           # 有效Query的标签, [M]
        M = len(valid_labels)
        
        # 仅处理有效Query（非背景）
        if valid_labels.numel() == 0:
            return torch.tensor(0.0, device=device)
        
        valid_queries = query_features[~is_bg]  # [M, D], M=有效Query数
        
        # --- 获取每个有效Query对应类的所有原型（含背景）---
        all_indices = self.prototype_indices.expand(batch_size, -1, -1) # [batch_size, num_queries, prototypes_per_class + 1]
        proto_features = self.class_embeddings(all_indices)[~is_bg].view(M, self.prototypes_per_class+1, feat_dim)  # [batch_size, num_queries, prototypes_per_class+1, D] -> [M, D]
        
        # --- 计算相似度 ---
        sim = torch.einsum('md,mkd->mk', valid_queries, proto_features)  # [M, K+1]
        
        # --- 正负样本定义 ---
        # 正样本：同类中相似度最高的正原型
        pos_sim, pos_idx = sim[:, :self.prototypes_per_class].max(dim=1)  # [M]
        
        # 负样本：同类背景 + 其他类的所有原型
        # 同类背景相似度
        bg_sim = sim[:, self.prototypes_per_class]  # [M]

        # 生成其他类原型索引（排除背景）
        other_proto_indices = []
        for lbl in valid_labels:
            lbl = lbl.item()
            class_start = lbl * (self.prototypes_per_class + 1)
            class_end = (lbl + 1) * (self.prototypes_per_class + 1)
            
            # 获取所有非当前类的索引
            indices = torch.cat([
                torch.arange(0, class_start),
                torch.arange(class_end, self.num_classes * (self.prototypes_per_class + 1))
            ])
            
            # 过滤其他类的背景原型
            mask = (indices % (self.prototypes_per_class + 1)) != self.prototypes_per_class
            filtered_indices = indices[mask]
            other_proto_indices.append(filtered_indices)
        
        other_proto_indices = torch.stack(other_proto_indices).to(device)  # [M, (num_classes-1)*prototypes_per_class]
        
        # 获取其他类原型特征并计算相似度
        other_protos = self.class_embeddings(other_proto_indices)  # [M, (C-1)*(K+1), D]
        other_sim = torch.einsum('md,mkd->mk', valid_queries, other_protos)  # [M, (C-1)*K]
        # 合并负样本logits
        neg_sim = torch.cat([bg_sim.unsqueeze(1), other_sim], dim=1)  # [M, 1 + (C-1)(K+1)]
        # --- 计算对比损失 ---
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)    # [M, 1 + N_neg]
        loss = F.cross_entropy(logits, torch.zeros(logits.size(0), dtype=torch.long, device=device))
        
        return loss


    def loss(self, query_features, masked_pooled_features, matched_labels):
        batch_size, num_queries, feat_dim = query_features.shape
        device = query_features.device

        # --- 分离背景Query ---
        is_bg = (matched_labels == self.num_classes)    # 背景标记, [batch_size, num_queries]
        valid_labels = matched_labels[~is_bg]           # 有效Query的标签, [M]
        M = len(valid_labels)
        
        # 仅处理有效Query（非背景）
        if valid_labels.numel() == 0:
            return torch.tensor(0.0, device=device)
        
        valid_queries = query_features[~is_bg]  # [M, D], M=有效Query数
        
        # --- 获取每个有效Query对应类的所有原型（含背景）---
        all_indices = self.prototype_indices.expand(batch_size, -1, -1) # [batch_size, num_queries, prototypes_per_class + 1]
        proto_features = self.class_embeddings(all_indices)[~is_bg].view(M, self.prototypes_per_class+1, feat_dim)  # [batch_size, num_queries, prototypes_per_class+1, D] -> [M, D]
        
        # --- 计算相似度 ---
        sim = torch.einsum('md,mkd->mk', valid_queries, proto_features)  # [M, K+1]
        
        # --- 正负样本定义 ---
        # 正样本：同类中相似度最高的正原型
        pos_sim, pos_idx = sim[:, :self.prototypes_per_class].max(dim=1)  # [M]
        
        # 负样本：同类背景 + 其他类的所有原型
        # 同类背景相似度
        bg_sim = sim[:, self.prototypes_per_class]  # [M]

        # 生成其他类原型索引（排除背景）
        other_proto_indices = []
        for lbl in valid_labels:
            lbl = lbl.item()
            class_start = lbl * (self.prototypes_per_class + 1)
            class_end = (lbl + 1) * (self.prototypes_per_class + 1)
            
            # 获取所有非当前类的索引
            indices = torch.cat([
                torch.arange(0, class_start),
                torch.arange(class_end, self.num_classes * (self.prototypes_per_class + 1))
            ])
            
            # 过滤其他类的背景原型
            mask = (indices % (self.prototypes_per_class + 1)) != self.prototypes_per_class
            filtered_indices = indices[mask]
            other_proto_indices.append(filtered_indices)
        
        other_proto_indices = torch.stack(other_proto_indices).to(device)  # [M, (num_classes-1)*prototypes_per_class]
        
        # 获取其他类原型特征并计算相似度
        other_protos = self.class_embeddings(other_proto_indices)  # [M, (C-1)*(K+1), D]
        other_sim = torch.einsum('md,mkd->mk', valid_queries, other_protos)  # [M, (C-1)*K]
        # 合并负样本logits
        neg_sim = torch.cat([bg_sim.unsqueeze(1), other_sim], dim=1)  # [M, 1 + (C-1)(K+1)]
        # --- 计算对比损失 ---
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)    # [M, 1 + N_neg]
        loss_qp = F.cross_entropy(logits, torch.zeros(logits.size(0), dtype=torch.long, device=device))




        # --- 计算相似度 ---
        sim_image_proto = torch.einsum('md,mkd->mk', masked_pooled_features, proto_features)  # [M, K+1]
        pos_sim, pos_idx = sim_image_proto[:, :self.prototypes_per_class].max(dim=1)  # [M]
        bg_sim = sim_image_proto[:, self.prototypes_per_class]
        other_sim = torch.einsum('md,mkd->mk', masked_pooled_features, other_protos)  # [M, (C-1)*K]
        neg_sim = torch.cat([bg_sim.unsqueeze(1), other_sim], dim=1)  # [M, 1 + (C-1)(K+1)]
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)    # [M, 1 + N_neg]
        loss_mp = F.cross_entropy(logits, torch.zeros(logits.size(0), dtype=torch.long, device=device))
        
        return loss_qp, loss_mp



class CodebookContrastiveHead7(nn.Module):
    '''
    不再为每个query固定匹配
    (19类 + 1bg) * 5 embeddings
    '''
    def __init__(self, num_classes=19, prototypes_per_class=5, embedding_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.prototypes_per_class = prototypes_per_class
        
        self.class_embeddings = nn.Parameter(
            torch.randn(num_classes + 1, prototypes_per_class, embedding_dim)
        )

    def forward(self, query_features):
        """
        Args:
            query_features: [batch_size, num_queries, embed_dim]
        Returns:
            logits: [batch_size, num_queries, num_classes + 1]
        """
        batch_size, num_queries, _ = query_features.shape
        
        sim = torch.einsum('bqd,npd->bqnp', query_features, self.class_embeddings)

        logits = sim.max(dim=-1)[0]

        return logits
    
    def loss(self, query_features, matched_labels):
        batch_size, num_queries, feat_dim = query_features.shape
        device = query_features.device

        # --- 分离背景Query ---
        is_bg = (matched_labels == self.num_classes)    # 背景标记, [batch_size, num_queries]
        valid_labels = matched_labels[~is_bg]           # 有效Query的标签, [M]
        M = len(valid_labels)
        
        # 仅处理有效Query（非背景）
        if valid_labels.numel() == 0:
            return torch.tensor(0.0, device=device)
        
        valid_queries = query_features[~is_bg]  # [M, D], M=有效Query数
        
        # --- 获取每个有效Query对应类的所有原型（含背景）---
        all_indices = self.prototype_indices.expand(batch_size, -1, -1) # [batch_size, num_queries, prototypes_per_class + 1]
        proto_features = self.class_embeddings(all_indices)[~is_bg].view(M, self.prototypes_per_class+1, feat_dim)  # [batch_size, num_queries, prototypes_per_class+1, D] -> [M, D]
        
        # --- 计算相似度 ---
        sim = torch.einsum('md,mkd->mk', valid_queries, proto_features)  # [M, K+1]
        
        # --- 正负样本定义 ---
        # 正样本：同类中相似度最高的正原型
        pos_sim, pos_idx = sim[:, :self.prototypes_per_class].max(dim=1)  # [M]
        
        # 负样本：同类背景 + 其他类的所有原型
        # 同类背景相似度
        bg_sim = sim[:, self.prototypes_per_class]  # [M]

        # 生成其他类原型索引（排除背景）
        other_proto_indices = []
        for lbl in valid_labels:
            lbl = lbl.item()
            class_start = lbl * (self.prototypes_per_class + 1)
            class_end = (lbl + 1) * (self.prototypes_per_class + 1)
            
            # 获取所有非当前类的索引
            indices = torch.cat([
                torch.arange(0, class_start),
                torch.arange(class_end, self.num_classes * (self.prototypes_per_class + 1))
            ])
            
            # 过滤其他类的背景原型
            mask = (indices % (self.prototypes_per_class + 1)) != self.prototypes_per_class
            filtered_indices = indices[mask]
            other_proto_indices.append(filtered_indices)
        
        other_proto_indices = torch.stack(other_proto_indices).to(device)  # [M, (num_classes-1)*prototypes_per_class]
        
        # 获取其他类原型特征并计算相似度
        other_protos = self.class_embeddings(other_proto_indices)  # [M, (C-1)*(K+1), D]
        other_sim = torch.einsum('md,mkd->mk', valid_queries, other_protos)  # [M, (C-1)*K]
        # 合并负样本logits
        neg_sim = torch.cat([bg_sim.unsqueeze(1), other_sim], dim=1)  # [M, 1 + (C-1)(K+1)]
        # --- 计算对比损失 ---
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)    # [M, 1 + N_neg]
        loss = F.cross_entropy(logits, torch.zeros(logits.size(0), dtype=torch.long, device=device))
        
        return loss



class CodebookContrastiveHead34567(nn.Module):
    '''
    每个类有5个prototypes和1个background embedding
    '''
    def __init__(self, num_classes=19, queries_per_class=5, prototypes_per_class=5, embedding_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.queries_per_class = queries_per_class
        self.prototypes_per_class = prototypes_per_class
        
        self.class_embeddings = nn.Embedding(
            num_embeddings=num_classes * (prototypes_per_class + 1),
            embedding_dim=embedding_dim
        )

        # --- 获取每个query对应的class所有prototypes ---
        # class_indices: [batch_size, num_queries]
        self.register_buffer(
            "class_indices",
            torch.arange(num_classes).repeat_interleave(queries_per_class).view(1, -1),
            persistent=False
        )
        # 每个类的prototype起始索引 [1, num_classes*queries_per_class]
        proto_start = self.class_indices * (self.prototypes_per_class + 1)
        offset = torch.arange(prototypes_per_class + 1)
        self.all_indices = proto_start.unsqueeze(-1) + offset.unsqueeze(0).unsqueeze(0)

    def forward(self, query_features):
        """
        Args:
            query_features: [batch_size, num_queries, embed_dim]
        Returns:
            logits: [batch_size, num_queries, num_classes + 1]
        """
        batch_size, num_queries, _ = query_features.shape
        
        # --- 获取每个query对应的class所有prototypes ---
        all_indices = self.all_indices.expand(batch_size, -1)
        proto_features = self.class_embeddings(all_indices)  # [B, Q, K+1, D]

        # --- 计算相似度 ---
        # 正prototypes相似度 [B, Q, K]
        pos_sim = torch.einsum('bqd,bqkd->bqk', 
                              query_features, 
                              proto_features[:, :, :-1, :])
        # 背景相似度 [B, Q, 1]
        bg_sim = torch.einsum('bqd,bqkd->bqk',
                             query_features,
                             proto_features[:, :, -1:, :])
        
        # --- 构造logits矩阵 ---
        # 初始化全-inf [B, Q, C+1]
        logits = torch.full((batch_size, num_queries, self.num_classes + 1),
                          float('-inf'),
                          device=query_features.device)
        
        # 填充正类相似度（取每个类多个prototype的最大值）
        logits[:, :, :self.num_classes] = pos_sim.max(dim=-1)[0].unsqueeze(-1) 
        
        # 填充背景相似度到最后一列
        logits[:, :, -1:] = bg_sim

        return logits
    
    def loss(self, query_features, matched_labels):
        batch_size, num_queries, _ = query_features.shape
        device = query_features.device

        sim = torch.einsum('bqd,nd->bqn', query_features, self.class_embeddings.weight)  # [B, Q, C*(K+1)]

        # 正Prototype掩码: [B, Q, C*(K+1)]
        pos_mask = torch.zeros_like(sim, dtype=torch.bool)
        # 负Prototype掩码: 其他类的所有正Prototype + 当前类的背景
        neg_mask = torch.zeros_like(sim, dtype=torch.bool)

        # --- 构造正负样本掩码 ---
        # 每个类别的Prototype索引范围: [C*(K+1)]
        proto_indices = torch.arange(self.num_classes*(self.prototypes_per_class+1), device=device)
        class_ranges = proto_indices.view(self.num_classes, self.prototypes_per_class+1)  # [C, K+1]
        
        # 正Prototype掩码: [B, Q, C*(K+1)]
        pos_mask = torch.zeros_like(sim, dtype=torch.bool)
        # 负Prototype掩码: 其他类的所有正Prototype + 当前类的背景
        neg_mask = torch.zeros_like(sim, dtype=torch.bool)
        
        # 遍历每个类别，批量生成掩码
        for c in range(self.num_classes):
            # 当前类的正Prototype索引 [K]
            pos_proto = class_ranges[c, :self.prototypes_per_class]
            # 当前类的背景索引 [1]
            bg_proto = class_ranges[c, self.prototypes_per_class]
            
            # 属于当前类的Query掩码 [B, Q]
            class_mask = (matched_labels == c)
            
            # --- 正样本掩码 ---
            pos_mask[:, :, pos_proto] |= class_mask.unsqueeze(-1)
            
            # --- 负样本掩码 ---
            # 其他类的所有正Prototype
            other_pos_proto = torch.cat([class_ranges[:c, :self.prototypes_per_class], class_ranges[c+1:, :self.prototypes_per_class]], dim=0).flatten()
            # 当前类的背景
            current_bg_proto = bg_proto.unsqueeze(0)
            
            # 合并负样本索引 [其他类正Prototype + 当前类背景]
            neg_indices = torch.cat([other_pos_proto, current_bg_proto])
            
            # 设置负样本掩码
            neg_mask[:, :, neg_indices] |= class_mask.unsqueeze(-1)
        
        # --- 计算对比损失 ---
        pos_sim = sim.masked_select(pos_mask).view(batch_size, num_queries, -1)  # [B, Q, K]
        neg_sim = sim.masked_select(neg_mask).view(batch_size, num_queries, -1)  # [B, Q, (C-1)*K+1]
        
        # 合并正负样本logits [B, Q, K + (C-1)*K +1]
        logits = torch.cat([pos_sim, neg_sim], dim=-1)
        


        # 目标标签：正样本位于前K个位置
        target = torch.zeros(batch_size, num_queries, dtype=torch.long, device=device)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                              target.view(-1), 
                              reduction='mean')
        
        return loss


class CodebookLoss(nn.Module):
    def __init__(self, commitment_cost=0.25):
        super().__init__()
        self.commitment_cost = commitment_cost

    def forward(self, query_features, class_embeddings, class_indices):
        """
        query_features: [Q, D]
        class_embeddings: [num_classes+1, D]
        class_indices: [Q]
        """
        # 将query绑定到对应class
        selected_embeddings = class_embeddings[class_indices]  # [B, Q, D]
        
        # Codebook Loss (仅更新codebook)
        codebook_loss = F.mse_loss(selected_embeddings.detach(), query_features.detach(), reduction='none').mean()
        
        # Commitment Loss (仅更新encoder)
        commitment_loss = F.mse_loss(query_features, selected_embeddings.detach(), reduction='none').mean()
        
        return codebook_loss + self.commitment_cost * commitment_loss


@TASK_UTILS.register_module()
class GroupMatchingFixedCost(BaseMatchCost):
    def __init__(self, num_classes=19, queries_per_class=5, penalty=1e7) -> None:
        super().__init__(weight=1.)
        self.num_classes = num_classes
        self.queries_per_class = queries_per_class
        self.penalty = penalty
        # 注册每个query所属的类别索引，例如假设query是按照类别顺序分配的
        # self.register_buffer(
        #     "class_indices",
        #     torch.arange(num_classes).repeat_interleave(queries_per_class),
        #     persistent=False
        # )
        self.class_indices = torch.arange(num_classes).repeat_interleave(queries_per_class)

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        gt_labels = gt_instances.labels  # [num_gt]
        pred_class_idx = self.class_indices.to(gt_labels.device)
        with torch.no_grad():
            # 对比维度扩展：[num_pred, 1] vs [1, num_gt]
            mismatch_mask = (
                pred_class_idx[:, None] != gt_labels[None, :]
            )  # 当query类别与GT类别不同时为True

        # 对于不匹配的位置施加极大惩罚，确保匈牙利算法不会选这些匹配
        return self.penalty * mismatch_mask.float()  # [num_pred, num_gt]












class ReliableAttention(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):        
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.embed_dim = embed_dims
        
        self.q_proj_weight = nn.Linear(embed_dims, embed_dims)
        self.k_proj_weight = nn.Linear(embed_dims, embed_dims)
        self.v_proj_weight = nn.Linear(embed_dims, embed_dims)
        self.out_proj_linear = nn.Linear(embed_dims, embed_dims)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super().__setstate__(state)

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # 线性变换
        q = self.q_proj_weight(query)  # (batch_size, seq_len, embed_dim)
        k = self.k_proj_weight(key)    # (batch_size, seq_len, embed_dim)
        v = self.v_proj_weight(value)  # (batch_size, seq_len, embed_dim)

        scaling = math.sqrt(self.embed_dims)

        sim_nm = (q @ k.transpose(-1, -2)) / scaling    # [b, n, m]
        score_m = sim_nm.sum(dim=1) # [b, m]
        

        top_values, top_indices = torch.topk(score_m, k=40, dim=1)  # dim=0表示在每一列中寻找
        top_indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, self.embed_dims)
        selected_features = torch.gather(k, 1, top_indices_expanded)

        sim_qn = F.softmax((q @ selected_features.transpose(-1, -2)) / scaling, dim=-1) # [b, n, k]
        sim_km = F.softmax((k @ selected_features.transpose(-1, -2)) / scaling, dim=-1) # [b, m, k]
        sim_qk = sim_qn @ sim_km.transpose(-1, -2)

        out = torch.einsum('bnm,bmc->bnc', sim_qk, v)
        out = self.out_proj_linear(out)
       
        return identity + self.dropout_layer(self.proj_drop(out))


class MultiheadReliableAttention(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 top_k=40,
                 **kwargs):        
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.head_dim = embed_dims // num_heads
        self.top_k = top_k
        assert self.head_dim * num_heads == embed_dims, "embed_dims must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        
        # 多头投影矩阵
        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.out_proj = nn.Linear(embed_dims, embed_dims)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    def __setstate__(self, state):
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True
        super().__setstate__(state)

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # 获取batch_size和序列长度
        if self.batch_first:
            batch_size, query_len = query.shape[0], query.shape[1]
            key_len = key.shape[1]
        else:
            batch_size, query_len = query.shape[1], query.shape[0]
            key_len = key.shape[0]
            # 转换为batch_first
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        # 线性投影
        q = self.q_proj(query)  # (batch_size, seq_len_q, embed_dims)
        k = self.k_proj(key)    # (batch_size, seq_len_k, embed_dims)
        v = self.v_proj(value)  # (batch_size, seq_len_v, embed_dims)

        # 重塑为多头形式
        q = q.reshape(batch_size, query_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, key_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, key_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # shape: [batch_size, num_heads, seq_len, head_dim]

        # 计算相似度矩阵
        sim_nm = torch.matmul(q, k.transpose(-1, -2)) * self.scaling  # [batch_size, num_heads, query_len, key_len]
        
        # 计算每个键向量的总得分 (跨查询累加)
        score_m = sim_nm.sum(dim=2)  # [batch_size, num_heads, key_len]
        
        # 为每个头选择top-k键
        top_k = min(self.top_k, key_len)  # 确保k不超过序列长度
        top_values, top_indices = torch.topk(score_m, k=top_k, dim=2)  # [batch_size, num_heads, top_k]
        
        # 正确收集选定的键向量
        # 创建索引用于gather操作
        batch_size_idx = torch.arange(batch_size, device=k.device).view(batch_size, 1, 1, 1)
        batch_indices = batch_size_idx.expand(batch_size, self.num_heads, top_k, self.head_dim)
        
        head_idx = torch.arange(self.num_heads, device=k.device).view(1, self.num_heads, 1, 1)
        head_indices = head_idx.expand(batch_size, self.num_heads, top_k, self.head_dim)
        
        dim_idx = torch.arange(self.head_dim, device=k.device).view(1, 1, 1, self.head_dim)
        dim_indices = dim_idx.expand(batch_size, self.num_heads, top_k, self.head_dim)
        
        # 扩展top_indices以用于索引
        top_indices_expanded = top_indices.unsqueeze(-1).expand(batch_size, self.num_heads, top_k, self.head_dim)
        
        # 使用高级索引选择top-k的键向量
        selected_k = k[batch_indices, head_indices, top_indices_expanded, dim_indices]  # [batch_size, num_heads, top_k, head_dim]
        
        # 计算查询与选定键之间的相似度
        sim_qn = torch.matmul(q, selected_k.transpose(-1, -2)) * self.scaling  # [batch_size, num_heads, query_len, top_k]
        sim_qn = F.softmax(sim_qn, dim=-1)  # 对top_k维度做softmax
        
        # 计算所有键与选定键之间的相似度
        sim_km = torch.matmul(k, selected_k.transpose(-1, -2)) * self.scaling  # [batch_size, num_heads, key_len, top_k]
        sim_km = F.softmax(sim_km, dim=-1)  # 对top_k维度做softmax
        
        # 计算最终的注意力权重矩阵
        sim_qk = torch.matmul(sim_qn, sim_km.transpose(-1, -2))  # [batch_size, num_heads, query_len, key_len]
        sim_qk = self.attn_drop(sim_qk)
        
        # 应用注意力权重到值向量
        out = torch.matmul(sim_qk, v)  # [batch_size, num_heads, query_len, head_dim]
        
        # 重新组合多头输出
        out = out.permute(0, 2, 1, 3).reshape(batch_size, query_len, self.embed_dims)
        
        # 输出投影
        out = self.out_proj(out)
        
        # 如果最初不是batch_first，则转换回来
        if not self.batch_first:
            out = out.transpose(0, 1)
            
        # 残差连接和dropout
        return identity + self.dropout_layer(self.proj_drop(out))




class Mask2FormerTransformerDecoderReliableMatching(DetrTransformerDecoder):
    """Decoder of Mask2Former."""

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            Mask2FormerTransformerDecoderLayerReliableMatching(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        self.post_norm = build_norm_layer(self.post_norm_cfg,
                                          self.embed_dims)[1]


class Mask2FormerTransformerDecoderLayerReliableMatching(DetrTransformerDecoderLayer):
    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn = MultiheadReliableAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                **kwargs) -> Tensor:
        query = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs)
        query = self.norms[0](query)
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            **kwargs)
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        return query


class Mask2FormerTransformerDecoderLBP(DetrTransformerDecoder):
    """Decoder of Mask2Former."""

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            Mask2FormerTransformerDecoderLayerLBP(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        self.post_norm = build_norm_layer(self.post_norm_cfg,
                                          self.embed_dims)[1]

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_lbp: Tensor, value_lbp: Tensor,
                query_pos: Tensor, key_pos: Tensor, key_padding_mask: Tensor,
                **kwargs) -> Tensor:
        intermediate = []
        for layer in self.layers:
            query = layer(
                query,
                key=key,
                value=value,
                key_lbp=key_lbp,
                value_lbp = value_lbp,
                query_pos=query_pos,
                key_pos=key_pos,
                key_padding_mask=key_padding_mask,
                **kwargs)
            if self.return_intermediate:
                intermediate.append(self.post_norm(query))
        query = self.post_norm(query)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return query.unsqueeze(0)

class Mask2FormerTransformerDecoderLayerLBP(Mask2FormerTransformerDecoderLayer):
    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn = MultiheadAttention(**self.cross_attn_cfg)
        self.cross_attn_lbp = MultiheadAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(4)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                key_lbp: Tensor = None,
                value_lbp: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                **kwargs) -> Tensor:
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            **kwargs)
        query = self.norms[0](query)
        query = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs)
        query = self.norms[1](query)
        query = self.cross_attn_lbp(
            query=query,
            key=key_lbp,
            value=value_lbp,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs)
        query = self.norms[2](query)
        query = self.ffn(query)
        query = self.norms[3](query)

        return query