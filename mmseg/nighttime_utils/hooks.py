# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Optional, Sequence

import torch
from torch.nn import functional as F
import mmcv
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer

from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample
from mmengine.dist import is_main_process, master_only
import torch.distributed as dist

@HOOKS.register_module()
class ClsEmbSimHook(Hook):
    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch = None,
                         outputs: Optional[dict] = None) -> None:
        if batch_idx % 50 == 0 and is_main_process():
            model = runner.model.module
            class_embeddings = model.decode_head.cls_embed.class_embeddings.weight
            class_embeddings = F.normalize(class_embeddings, p=2, dim=1)
            sim_matrix = torch.mm(class_embeddings, class_embeddings.t()).detach().cpu()
            print(sim_matrix)


@HOOKS.register_module()
class ClsEmbSwitchHook(Hook):

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:
        current_iter = runner.iter

        # 动态冻结参数（全局第5000次迭代触发）
        if current_iter == 5000:
            model = runner.model.module if hasattr(runner.model, 'module') else runner.model
            param = model.decode_head.cls_embed.class_embeddings.weight

            # 主进程设置参数状态并广播
            if is_main_process():
                param.requires_grad_(False)

            # 同步requires_grad状态到所有进程
            if dist.is_available() and dist.is_initialized():
                requires_grad_tensor = torch.tensor(param.requires_grad, device=param.device)
                dist.broadcast(requires_grad_tensor, src=0)
                param.requires_grad_(requires_grad_tensor.item())

            # 主进程打印日志
            if is_main_process():
                print(f"Iteration {current_iter}: Set class_embeddings.requires_grad to {param.requires_grad}")

            # 确保所有进程同步完成
            dist.barrier()

        # 监控部分（每5000次全局迭代）
        if current_iter % 5000 == 0:
            model = runner.model.module if hasattr(runner.model, 'module') else runner.model
            class_embeddings = model.decode_head.cls_embed.class_embeddings.weight

            if is_main_process():
                print(f"Iteration {current_iter}: class_embeddings.requires_grad = {class_embeddings.requires_grad}")

                # 计算相似度矩阵
                normalized_embeddings = F.normalize(class_embeddings, p=2, dim=1)
                sim_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t()).detach().cpu()
                print(f"Similarity Matrix (subset):\n{sim_matrix[:5, :5]}")

                # 检查梯度
                if class_embeddings.grad is not None:
                    print(f"Gradient norm: {class_embeddings.grad.norm().item()}")
                else:
                    print("Gradient is None")
