import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
import torchsparse

from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from .FreqFusion import FreqFusion

from typing import List, Optional, Dict, Any
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from mmseg.models.segmentors.base import BaseSegmentor
from torch import Tensor


@MODELS.register_module()
class EntropyEnsembler(nn.Module):
    def forward(self, l1: torch.Tensor, l2: torch.Tensor) -> torch.Tensor:
        p1 = torch.softmax(l1, dim=-1)
        e1 = -torch.sum(p1 * torch.log(p1.clamp(min=1e-5)), dim=-1, keepdim=True)
        p2 = torch.softmax(l2, dim=-1)
        e2 = -torch.sum(p2 * torch.log(p2.clamp(min=1e-5)), dim=-1, keepdim=True)
        return torch.where(e1 < e2, l1, l2)


@MODELS.register_module()
class GatedEnsembler(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear((num_classes + 1) * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.fuser = nn.Sequential(
            nn.Linear(num_classes, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, l1: torch.Tensor, l2: torch.Tensor) -> torch.Tensor:
        p1 = torch.softmax(l1, dim=-1)
        e1 = -torch.sum(p1 * torch.log(p1.clamp(min=1e-5)), dim=-1, keepdim=True)
        x1 = torch.cat([p1, e1], dim=-1)

        p2 = torch.softmax(l2, dim=-1)
        e2 = -torch.sum(p2 * torch.log(p2.clamp(min=1e-5)), dim=-1, keepdim=True)
        x2 = torch.cat([p2, e2], dim=-1)

        x = torch.cat([x1, x2], dim=-1)
        w = torch.softmax(self.attn(x), dim=-1)
        y = torch.sum(torch.stack([l1, l2], dim=-1) * w.unsqueeze(1), dim=-1)
        return self.fuser(y)


@MODELS.register_module()
class TorchSparseFeaturizer(nn.Module):
    def __init__(self, features: List[str], is_half: bool = True) -> None:
        super().__init__()
        self.features = features
        self.is_half = is_half

    def forward(self, inputs: Dict[str, Any], mask: torch.Tensor) -> torchsparse.SparseTensor:
        batch_size, *spatial_shape = mask.shape
        indices = mask.nonzero()

        features = []
        for name in self.features:
            if name == "xy":
                feature = indices[:, 1:3].float() / torch.tensor(spatial_shape, device=indices.device).float()
            elif name == "rgb":
                feature = inputs["image"][mask]
            elif name.startswith("p"):
                feature = torch.softmax(inputs["logits"][mask], dim=-1)
                if name.startswith("p>"):
                    feature = (feature > float(name[2:])).float()
            else:
                raise ValueError(f"Unknown feature: '{name}'")
            features.append(feature)
        features = torch.cat(features, dim=-1)

        # Pad coordinates to 3D
        indices = torch.cat([indices, torch.zeros(indices.shape[0], 1, device=indices.device)], dim=-1)

        if self.is_half:
            sp_tensor = torchsparse.SparseTensor(features.half(), indices.int())
        else:
            sp_tensor = torchsparse.SparseTensor(features.float(), indices.int())
        return sp_tensor


@MODELS.register_module()
class EntropySelector(nn.Module):
    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold


    def forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        probs = torch.softmax(inputs["logits"], dim=-1)
        entropy = -torch.sum(probs * torch.log(probs.clamp(1e-5)), dim=-1)
        return entropy > self.threshold


@MODELS.register_module()
class LearnableSelector(nn.Module):
    def __init__(self, threshold: float, num_classes: int, percentage: float = 0.118) -> None:
        super().__init__()
        self.threshold = threshold
        self.mlp_hidden_size = 32
        self.out_channels = 16
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(num_classes+self.out_channels, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        
        image_features = self.conv(inputs["image"].permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        
        combined_input = torch.cat((inputs["logits"],image_features), dim=-1)

        logit_mask = self.mlp(combined_input).squeeze(-1)

        return self.sigmoid(logit_mask) > self.threshold


@MODELS.register_module()
class SparseRefiner(BaseSegmentor):
    def __init__(
        self,
        selector,
        featurizer,
        backbone,
        classifier,
        ensembler,
        loss_mask,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.selector = MODELS.build(selector)
        self.featurizer = MODELS.build(featurizer)
        self.backbone = MODELS.build(backbone)
        # self.classifier = MODELS.build(classifier)
        self.classifier = nn.Linear(32, 19)
        self.ensembler = MODELS.build(ensembler)
        self.loss_mask = MODELS.build(loss_mask)

    def _forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        mask = self.selector(inputs)
        x = self.featurizer(inputs, mask)
        x = self.backbone(x).F
        yo = self.classifier(x)

        logits = inputs["logits"]
        yi = logits[mask]
        ye = self.ensembler(yi, yo)

        outputs = {}
        outputs["logits/i/mask"] = yi
        outputs["logits/o/mask"] = yo
        outputs["logits/e/mask"] = ye

        for name in ["logits/i", "logits/o", "logits/e"]:
            y = logits.clone()
            y[mask] = outputs[name + "/mask"].to(dtype=y.dtype)
            outputs[name + "/full"] = y.permute(0, 3, 1, 2)

        if "label" in inputs:
            outputs["label/full"] = inputs["label"]
            outputs["label/mask"] = inputs["label"][mask]

        return outputs
    
    def loss(self, inputs: torch.Tensor, batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        inputs_dict = dict()
        inputs_dict["logits"] = torch.stack([data_sample.logits.data for data_sample in batch_data_samples])
        inputs_dict["image"] = inputs.permute(0, 2, 3, 1)
        inputs_dict["label"] = torch.stack([data_sample.gt_sem_seg.data for data_sample in batch_data_samples]).squeeze(1)
        outputs = self._forward(inputs_dict)

        losses = dict()
        losses['loss_mask_output'] = self.loss_mask(outputs['logits/o/mask'], outputs['label/mask'])
        losses['loss_mask_ensemble'] = self.loss_mask(outputs['logits/e/mask'], outputs['label/mask'])

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        outputs = self._forward(inputs)
        seg_logits = outputs['logits/e/full']
        return self.postprocess_result(seg_logits, data_samples)

    def extract_feat(self, inputs: Tensor) -> bool:
        """Placeholder for extract features from images."""
        pass

    def encode_decode(self, inputs: Tensor, batch_data_samples: SampleList):
        """Placeholder for encode images with backbone and decode into a
        semantic segmentation map of the same size as input."""
        outputs = self._forward(inputs)
        seg_logits = outputs['logits/e/full']
        return seg_logits
    