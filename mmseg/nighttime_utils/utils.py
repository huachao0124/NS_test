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
    numerator = 2 * torch.einsum('nc,mc->nm', mask_preds, gt_masks)
    if naive_dice:
        denominator = mask_preds.sum(-1)[:, None] + \
                        gt_masks.sum(-1)[None, :]
    else:
        denominator = mask_preds.pow(2).sum(1)[:, None] + \
                        gt_masks.pow(2).sum(1)[None, :]
    loss = 1 - (numerator + eps) / (denominator + eps)
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

    def __init__(self):
        pass

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


        overlaps = pairwise_mask_iou(pred_masks, gt_masks)

        overlaps_max, indices = torch.max(overlaps, dim=1)

        matched_row_inds = torch.arange(pred_masks.size(0)).to(device)
        matched_col_inds = indices.to(device)
        matched_row_inds = matched_row_inds.to(device)
        matched_col_inds = matched_col_inds.to(device)
        
        mask = overlaps_max > 0.1
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
            labels=assigned_labels)


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
        df.to_csv('error_matrix_0.3.csv')


        np.savetxt('error_matrix_2.txt', ems_sum.squeeze(), fmt='%5d')

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