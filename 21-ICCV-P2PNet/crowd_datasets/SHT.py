import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob

# BaseData 임포트 (경로 문제 방지)
try:
    from base_data import BaseData
except ImportError:
    from .base_data import BaseData

class SHHA(BaseData):
    """
    ShanghaiTech Part A 데이터셋 클래스 
    """
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False, 
                 use_npoint=False, alpha=0.2, adaptive_npoint=0):
        # BaseData의 __init__에 모든 인자를 정확히 전달하여 에러 방지
        super(SHHA, self).__init__(
            data_root, 
            transform=transform, 
            train=train, 
            patch=patch, 
            flip=flip, 
            use_npoint=use_npoint, 
            alpha=alpha,
            adaptive_npoint=adaptive_npoint
        )

    def build_img_map(self):
        mode = 'train_data' if self.train else 'test_data'
        # 이미지 파일 리스트업
        self.img_lists = sorted(glob.glob(os.path.join(self.data_root, 'part_A_final', mode, 'images', '*.jpg')))
        
        # 오직 .txt 정답지만 검색
        self.gt_lists = glob.glob(os.path.join(self.data_root, 'part_A_final', mode, 'ground_truth', 'GT_*.txt'))
        
        gt_dict = {
            os.path.basename(p).replace('GT_', '').replace('.txt', ''): p
            for p in self.gt_lists
        }
        
        for img_path in self.img_lists:
            img_key = os.path.basename(img_path).replace('.jpg', '')
            if img_key in gt_dict:
                self.img_map[img_path] = gt_dict[img_key]
            else:
                print(f"⚠️ [SHHA] GT를 찾을 수 없음: {img_key}")

        self.img_list = sorted(list(self.img_map.keys()))

    def load_data(self, img_path, gt_path):
        img = cv2.imread(img_path)
        if img is None: raise FileNotFoundError(f"❌ 이미지 읽기 실패: {img_path}")
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        points = []
        with open(gt_path, 'r', errors='ignore') as f_label:
            for line in f_label:
                line = line.strip().replace(',', ' ').split()
                if not line: continue
                try:
                    points.append([float(line[0]), float(line[1])])
                except (ValueError, IndexError):
                    continue
                    
        return img, np.array(points)

class SHHB(BaseData):
    """
    ShanghaiTech Part B 데이터셋 클래스 
    """
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False, 
                 use_npoint=False, alpha=0.2, adaptive_npoint=0):
        # [수정] super(SHHA, self) -> super(SHHB, self)로 오타 수정
        # [수정] adaptive_npoint 인자 추가
        super(SHHB, self).__init__(
            data_root, 
            transform=transform, 
            train=train, 
            patch=patch, 
            flip=flip, 
            use_npoint=use_npoint, 
            alpha=alpha,
            adaptive_npoint=adaptive_npoint
        )

    def build_img_map(self):
        mode = 'train_data' if self.train else 'test_data'
        # [수정] SHHB 경로로 정확히 지정 (part_B_final)
        self.img_lists = sorted(glob.glob(os.path.join(self.data_root, 'part_B_final', mode, 'images', '*.jpg')))
        self.gt_lists = glob.glob(os.path.join(self.data_root, 'part_B_final', mode, 'ground_truth', 'GT_*.txt'))
        
        gt_dict = {
            os.path.basename(p).replace('GT_', '').replace('.txt', ''): p
            for p in self.gt_lists
        }
        
        for img_path in self.img_lists:
            img_key = os.path.basename(img_path).replace('.jpg', '')
            if img_key in gt_dict:
                self.img_map[img_path] = gt_dict[img_key]
            else:
                # 패턴 다양성 대응
                alt_names = [f"GT_{img_key}.txt", f"{img_key}.txt"]
                found = False
                for alt in alt_names:
                    p = os.path.join(self.data_root, 'part_B_final', mode, 'ground_truth', alt)
                    if os.path.exists(p):
                        self.img_map[img_path] = p
                        found = True
                        break
                if not found:
                    print(f"⚠️ [SHHB] GT를 찾을 수 없음: {img_key}")

        self.img_list = sorted(list(self.img_map.keys()))

    def load_data(self, img_path, gt_path):
        img = cv2.imread(img_path)
        if img is None: raise FileNotFoundError(f"❌ 이미지 읽기 실패: {img_path}")
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        points = []
        with open(gt_path, 'r', errors='ignore') as f_label:
            for line in f_label:
                line = line.strip().replace(',', ' ').split()
                if not line: continue
                try:
                    points.append([float(line[0]), float(line[1])])
                except (ValueError, IndexError):
                    continue
                    
        return img, np.array(points)