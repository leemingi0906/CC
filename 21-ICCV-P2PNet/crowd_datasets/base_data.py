import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob
import scipy.io as io

# NPoint 모듈 임포트
try:
    from npoint_aug import apply_npoint
except ImportError:
    from .npoint_aug import apply_npoint

class BaseData(Dataset):
    def __init__(self, data_root, 
                 transform=None, train=False, patch=False, flip=False, 
                 use_npoint=False, alpha=0.2, adaptive_npoint=0):
        self.data_root = data_root
        self.train = train
        self.patch = patch
        self.flip = flip
        self.transform = transform
        
        self.use_npoint = use_npoint
        self.alpha = alpha
        self.adaptive_npoint = adaptive_npoint

        self.img_map = {} # img-gt 연결
        self.img_list = [] # img 경로 리스트

        self.build_img_map()

        self.nSamples = len(self.img_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        img_path = self.img_list[index]
        gt_path = self.img_map[img_path]
        
        img, point = self.load_data(img_path, gt_path)
        
        if self.train and self.use_npoint:
            w, h = img.size 
            n_point = apply_npoint(point, (h, w), alpha=self.alpha, k=6)

        if self.transform is not None:
            img = self.transform(img)

        if self.train:
            img, point = self.apply_train_augmentation(img, point, n_point)

        if not self.train:
            point = [point]

        target = self.build_target(point)
        return img, target


    def build_img_map(self):
        raise NotImplementedError("build_img_map() must be implemented in child dataset class.")


    def load_data(self, img_path, img_gt_path):
        raise NotImplementedError("load_data() must be implemented in child dataset class.")


    def apply_train_augmentation(self, img, point, n_point):
        # scale
        scale_range = [0.7, 1.3]
        min_size = min(img.shape[1:])
        scale = random.uniform(*scale_range)
        if scale * min_size > 128:
            img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale, mode='bilinear').squeeze(0)
            point = point * scale
            n_point = n_point * scale

        # crop (patch일 때만)
        if self.patch:
            if self.adaptive_npoint:
                # child가 adaptive를 쓰고 싶으면 여기 연결
                img, point = self.random_crop_for_adaptive_npoint(img, point, n_point, 
                                                                  threshold=self.adaptive_npoint)
            else:
                img, point = self.random_crop(img, n_point)
        else:
            point = n_point

        # flip
        if self.flip and random.random() > 0.5:
            img = torch.flip(img, dims=[-1])
            if isinstance(point, list):
                for p in point:
                    p[:, 0] = 128 - p[:, 0]
            else:
                point[:, 0] = img.shape[-1] - point[:, 0]

        return img, point

    def build_target(self, point):
        target = []
        for i in range(len(point)):
            d = {}
            p = torch.tensor(point[i], dtype=torch.float32)
            d['point'] = p
            d['labels'] = torch.ones([p.shape[0]], dtype=torch.int64)
            target.append(d)
        return target

    def random_crop(self, img, den, num_patch=4):
        half_h, half_w = 128, 128
        result_img = torch.zeros([num_patch, img.shape[0], half_h, half_w])
        result_den = []
        c, h, w = img.shape
        for i in range(num_patch):
            start_h = random.randint(0, max(0, h - half_h))
            start_w = random.randint(0, max(0, w - half_w))
            result_img[i] = img[:, start_h:start_h+half_h, start_w:start_w+half_w]
            idx = (den[:, 0] >= start_w) & (den[:, 0] <= start_w + half_w) & \
                (den[:, 1] >= start_h) & (den[:, 1] <= start_h + half_h)
            record_den = den[idx].copy()
            record_den[:, 0] -= start_w
            record_den[:, 1] -= start_h
            result_den.append(record_den)
        return result_img, result_den

    def random_crop_for_adaptive_npoint(self, img, den_o, den_npoint, threshold=5, num_patch=4):
        half_h, half_w = 128, 128
        result_img = torch.zeros([num_patch, img.shape[0], half_h, half_w])
        result_den = []
        c, h, w = img.shape
        for i in range(num_patch):
            start_h = random.randint(0, max(0, h - half_h))
            start_w = random.randint(0, max(0, w - half_w))
            result_img[i] = img[:, start_h:start_h+half_h, start_w:start_w+half_w]
            idx = (den_o[:, 0] >= start_w) & (den_o[:, 0] <= start_w + half_w) & \
                (den_o[:, 1] >= start_h) & (den_o[:, 1] <= start_h + half_h)
            if np.sum(idx) < threshold:
                record_den = den_o[idx].copy()
            else:
                record_den = den_npoint[idx].copy()
            record_den[:, 0] -= start_w
            record_den[:, 1] -= start_h
            result_den.append(record_den)
        return result_img, result_den