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

from mmcv.transforms import to_tensor
from mmengine.structures import PixelData

from mmseg.structures import SegDataSample
from mmseg.datasets.transforms import PackSegInputs

from skimage.feature import local_binary_pattern
import torch


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
class BetaDarker(BaseTransform):
    def transform(self, results: dict, beta: int = -40) -> dict:
        image = results['img']
        dark_image_bgr = cv2.convertScaleAbs(image, alpha=1, beta=beta)
        results['img'] = dark_image_bgr
        return results


@TRANSFORMS.register_module()
class MotionBlur(BaseTransform):
    def __init__(self, json_file='mmseg/nighttime_utils/motion_blur_params.json'):
        with open(json_file, 'r', encoding='utf-8') as file:
            self.motion_blur_params = json.load(file)

    def transform(self, results: dict) -> dict:
        image = results['img']
        if 'train' in results['img_path']:
            results['img'] = self.apply_motion_blur(image, **self.motion_blur_params[results['sample_idx']])
        else:
            results['img'] = self.apply_motion_blur(image, **self.motion_blur_params[results['sample_idx'] + 5000])
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


@TRANSFORMS.register_module()
class LoadLogits(BaseTransform):
    def __init__(self, seg_logits_path, data_path):
        self.seg_logits_path = seg_logits_path
        self.data_path = data_path

    def transform(self, results: dict) -> dict:
        logits = np.load(results['img_path'].replace('.png', '.npy').replace(self.data_path, self.seg_logits_path)).transpose(1, 2, 0)
        logits = cv2.resize(logits, (results['img'].shape[1], results['img'].shape[0]))
        results['logits'] = logits
        results['seg_fields'].append('logits')
        return results

@TRANSFORMS.register_module()
class BitZero(BaseTransform):
    def __init__(self, num_bits = 3):
        self.num_bits = num_bits

    def transform(self, results: dict) -> dict:
        results['img'] &= np.uint8(0xFF << self.num_bits)
        return results


@TRANSFORMS.register_module()
class UpBitZero(BaseTransform):
    def __init__(self, num_bits = 3):
        self.num_bits = num_bits

    def transform(self, results: dict) -> dict:
        results['img'] &= ~np.uint8(0xFF << (8 - self.num_bits))
        return results

@TRANSFORMS.register_module()
class EnhanceEdge(BaseTransform):
    def transform(self, results: dict) -> dict:
        edges = cv2.Canny(results['gt_seg_map'].astype(np.uint8), threshold1=100, threshold2=200)
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        mask = edges > 0
        results['img'][mask] = 255
        return results
    

def lbp_to_6channels_unpackbits(lbp):
    # 确保数据类型是uint8
    lbp_uint8 = lbp.astype(np.uint8)
    # 展开为二进制，然后取后6位
    binary = np.unpackbits(lbp_uint8[...,None], axis=2)
    return binary[...,-6:].astype(np.uint8)

def lbp_to_8channels_unpackbits(lbp):
    # 确保数据类型是uint8
    lbp_uint8 = lbp.astype(np.uint8)
    # 展开为二进制，然后取后6位
    binary = np.unpackbits(lbp_uint8[...,None], axis=2)
    return binary[...,-8:].astype(np.uint8)

def nrlbp(image, P=8, R=1, epsilon=3):
    rows, cols = image.shape
    lbp = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(R, rows-R):
        for j in range(R, cols-R):
            center = image[i,j]
            code = 0
            for k in range(P):
                x = i + R * np.cos(2*np.pi*k/P)
                y = j - R * np.sin(2*np.pi*k/P)
                x, y = int(x), int(y)
                if abs(int(image[x,y]) - int(center)) <= epsilon:
                    bit = 0
                elif image[x,y] > center:
                    bit = 1
                else:
                    bit = 0
                code |= (bit << k)
            lbp[i,j] = code  # 将计算的code赋值给result
    return lbp

def nrlbp_fast(image, P=8, R=1, epsilon=3):
    """快速NRLBP实现"""
    rows, cols = image.shape
    # 预计算采样点的坐标偏移
    angles = 2 * np.pi * np.arange(P) / P
    x_offsets = R * np.cos(angles)
    y_offsets = -R * np.sin(angles)  # 注意图像坐标系y轴向下
    # 创建结果数组
    result = np.zeros_like(image)
    # 计算有效区域
    valid_rows = slice(R, rows-R)
    valid_cols = slice(R, cols-R)
    # 中心像素值
    center = image[valid_rows, valid_cols]
    # 初始化编码
    codes = np.zeros_like(center, dtype=np.uint8)
    
    # 对每个采样点进行计算
    for k in range(P):
        # 计算采样点坐标
        x = np.round(np.arange(R, rows-R)[:, None] + x_offsets[k]).astype(np.int32)
        y = np.round(np.arange(R, cols-R)[None, :] + y_offsets[k]).astype(np.int32)
        # 获取采样点的值
        neighbors = image[x, y]
        # 计算差值
        diff = neighbors - center
        # NRLBP编码规则
        codes = codes.astype(np.uint8) | ((diff > epsilon).astype(np.uint8) << k)
    
    # 保存结果
    result[valid_rows, valid_cols] = codes
    
    return result


@TRANSFORMS.register_module()
class LBP(BaseTransform):
    def __init__(self, mode: str = 'uniform'):
        self.mode = mode
    
    def transform(self, results: dict) -> dict:
        image = results['img']
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if self.mode == 'nrlbp':
            lbp = nrlbp_fast(hsv_image[:, :, 2])
        else:
            lbp = local_binary_pattern(hsv_image[:, :, 2], P=8, R=1, method=self.mode)
        if self.mode == 'uniform' or self.mode == 'ror':
            lbp = lbp_to_6channels_unpackbits(lbp)
        elif self.mode == 'default' or self.mode == 'nrlbp':
            lbp = lbp_to_8channels_unpackbits(lbp)
        results['lbp'] = torch.from_numpy(lbp).permute(2, 0, 1).float()
        return results


@TRANSFORMS.register_module()
class MultiScaleLBP(BaseTransform):
    def __init__(self, mode: str = 'uniform', num_scales=3):
        self.mode = mode
        self.num_scales = num_scales
    
    def transform(self, results: dict) -> dict:
        image = results['img']
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lbps = []
        for i in range(self.num_scales):
            if self.mode == 'nrlbp':
                lbp = nrlbp_fast(hsv_image[:, :, 2], P=8, R=i+1)
            else:
                lbp = local_binary_pattern(hsv_image[:, :, 2], P=8, R=i+1, method=self.mode)
            if self.mode == 'uniform' or self.mode == 'ror':
                lbp = lbp_to_6channels_unpackbits(lbp)
            elif self.mode == 'default' or self.mode == 'nrlbp':
                lbp = lbp_to_8channels_unpackbits(lbp)
            lbps.append(lbp)
        lbp_res = np.concatenate(lbps, axis=-1)
        results['lbp'] = torch.from_numpy(lbp_res).permute(2, 0, 1).float()
        return results

@TRANSFORMS.register_module()
class PackSegInputsWithLogits(PackSegInputs):

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results['inputs'] = img

        data_sample = SegDataSample()
        if 'gt_seg_map' in results:
            if len(results['gt_seg_map'].shape) == 2:
                data = to_tensor(results['gt_seg_map'][None,
                                                       ...].astype(np.int64))
            else:
                warnings.warn('Please pay attention your ground truth '
                              'segmentation map, usually the segmentation '
                              'map is 2D, but got '
                              f'{results["gt_seg_map"].shape}')
                data = to_tensor(results['gt_seg_map'].astype(np.int64))
            gt_sem_seg_data = dict(data=data)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        if 'gt_edge_map' in results:
            gt_edge_data = dict(
                data=to_tensor(results['gt_edge_map'][None,
                                                      ...].astype(np.int64)))
            data_sample.set_data(dict(gt_edge_map=PixelData(**gt_edge_data)))

        if 'gt_depth_map' in results:
            gt_depth_data = dict(
                data=to_tensor(results['gt_depth_map'][None, ...]))
            data_sample.set_data(dict(gt_depth_map=PixelData(**gt_depth_data)))
        
        if 'logits' in results:
            logits_data = dict(
                data=to_tensor(results['logits']))
            data_sample.set_data(dict(logits=PixelData(**logits_data)))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


