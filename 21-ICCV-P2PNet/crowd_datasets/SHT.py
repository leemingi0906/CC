try:
    from base_data import BaseData
except ImportError:
    from .base_data import BaseData


import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob
import scipy.io as io


class SHHA(BaseData):
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False, 
                 use_npoint=False, alpha=0.2, adaptive_npoint=False):
        super(SHHA, self).__init__(
            data_root, 
            transform=transform, 
            train=train, 
            patch=patch, 
            flip=flip, 
            use_npoint=use_npoint, 
            alpha=alpha
        )

    def build_img_map(self):
        if self.train:
            self.img_lists = glob.glob(f'{self.data_root}/part_A_final/train_data/images/*.jpg')
            self.gt_lists = glob.glob(f'{self.data_root}/part_A_final/train_data/ground_truth/*.txt')
        else:
            self.img_lists = glob.glob(f'{self.data_root}/part_A_final/test_data/images/*.jpg')
            self.gt_lists = glob.glob(f'{self.data_root}/part_A_final/test_data/ground_truth/*.txt')
        
        gt_dict = {
            os.path.basename(p).replace('GT_', '').replace('.txt', ''): p
            for p in self.gt_lists
        }
        for img_path in self.img_lists:
            img_key = os.path.basename(img_path).replace('.jpg', '')

            if img_key not in gt_dict:
                print(f"⚠️ GT not found for image: {img_path}")
                continue

            self.img_map[img_path] = gt_dict[img_key]

        self.img_list = sorted(self.img_map.keys())

        
    def load_data(self, img_path, gt_path):
        img = cv2.imread(img_path)
        if img is None: raise FileNotFoundError(f"❌ 이미지 읽기 실패: {img_path}")
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        points = []
        with open(gt_path, 'r', errors='ignore') as f_label:
            for line in f_label:
                line = line.strip().split()
                if not line: continue
                points.append([float(line[0]), float(line[1])])
        return img, np.array(points)



class SHHB(BaseData):
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False, use_npoint=False, alpha=0.2):
        super(SHHA, self).__init__(
            data_root, 
            transform=transform, 
            train=train, 
            patch=patch, 
            flip=flip, 
            use_npoint=use_npoint, 
            alpha=alpha
        )

    def build_img_map(self):
        if self.train:
            self.img_lists = glob.glob(f'{self.data_root}/part_B_final/train_data/images/*.jpg')
            self.gt_lists = glob.glob(f'{self.data_root}/part_B_final/train_data/ground_truth/*.txt')
        else:
            self.img_lists = glob.glob(f'{self.data_root}/part_B_final/test_data/images/*.jpg')
            self.gt_lists = glob.glob(f'{self.data_root}/part_B_final/test_data/ground_truth/*.txt')
        
        gt_dict = {
            os.path.basename(p).replace('GT_', '').replace('.txt', ''): p
            for p in self.gt_lists
        }
        for img_path in self.img_lists:
            img_key = os.path.basename(img_path).replace('.jpg', '')

            if img_key not in gt_dict:
                print(f"⚠️ GT not found for image: {img_path}")
                continue

            self.img_map[img_path] = gt_dict[img_key]

        self.img_list = sorted(self.img_map.keys())

        
    def load_data(self, img_path, gt_path):
        img = cv2.imread(img_path)
        if img is None: raise FileNotFoundError(f"❌ 이미지 읽기 실패: {img_path}")
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        points = []
        with open(gt_path, 'r', errors='ignore') as f_label:
            for line in f_label:
                line = line.strip().split()
                if not line: continue
                points.append([float(line[0]), float(line[1])])
        return img, np.array(points)