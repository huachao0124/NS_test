from mmseg.registry import DATASETS, TRANSFORMS
from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.datasets.cityscapes import CityscapesDataset

import mmcv
from mmcv.transforms.base import BaseTransform
import mmengine
import mmengine.fileio as fileio
from mmengine.fileio import list_from_file, get_local_path
from mmengine.config import Config
from mmengine.logging import print_log
from mmengine.dataset import BaseDataset, Compose, force_full_init

import random
import os
import os.path as osp
import pickle
import copy
import cv2
import math
import numpy as np
import logging
from collections.abc import Mapping
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from torchvision import transforms
from torch.utils.data import Dataset
import json
import glob
from PIL import Image


@TRANSFORMS.register_module()
class HSVDarker(BaseTransform):
    def transform(self, results: dict) -> dict:
        image = results['img']
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 2] = hsv_image[:, :, 2] * 0.4
        dark_image_bgr = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        results['img'] = dark_image_bgr
        return results


@TRANSFORMS.register_module()
class MotionBlur(BaseTransform):
    def __init__(self):
        self.param_map = dict()

    def transform(self, results: dict) -> dict:
        image = results['img']
        if results['sample_idx'] not in self.param_map:
            self.param_map['sample_idx'] = dict(degree=random.randint(5, 20),
                                                angle=random.uniform(0, 360))
        results['img'] = self.apply_motion_blur(image, **self.param_map['sample_idx'])
        return results

    def apply_motion_blur(self, img, degree=15, angle=45):
        # 生成运动模糊核
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        kernel_motion_blur = np.diag(np.ones(degree))
        kernel_motion_blur = cv2.warpAffine(kernel_motion_blur, M, (degree, degree))
        kernel_motion_blur = kernel_motion_blur / degree
        
        # 对图像应用滤波
        blurred = cv2.filter2D(img, -1, kernel_motion_blur)
        return blurred
