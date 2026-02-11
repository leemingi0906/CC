import torch.utils.data as data
import os
from glob import glob
import torch
from torchvision import transforms
import random
import numpy as np
import cv2
from PIL import Image
import scipy.io as sio

# [핵심] NPoint 모듈 임포트
try:
    from npoint_aug import apply_npoint
except ImportError:
    # 파일이 없을 경우 대비 (베이스라인 모드)
    def apply_npoint(points, *args, **kwargs): return points

def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w

class Crowd(data.Dataset):
    def __init__(self, root_path, crop_size=256,
                 downsample_ratio=8,
                 method='train',
                 alpha=0.0,
                 adaptive_npoint=7):
        
        self.root_path = root_path
        self.crop_size = crop_size
        self.downsample_ratio = downsample_ratio
        self.method = method
        
        # NPoint 설정
        self.alpha = alpha
        self.adaptive_npoint = adaptive_npoint
        self.use_npoint = (self.method == 'train' and self.alpha > 0)
        
        # 이미지 경로 자동 탐색 (ShanghaiTech 구조 대응)
        # SHT/part_A_final/train_data/images/*.jpg 패턴
        if 'part_' in root_path:
            self.im_list = sorted(glob(os.path.join(self.root_path, 'images', '*.jpg')))
        else:
            self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))
        
        if method == 'train':
            print(f"📊 [DM-Count] {len(self.im_list)} images loaded | NPoint: {self.use_npoint} (α={self.alpha})")
        
        # 전처리 (ImageNet Mean/Std)
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        # GT 경로 자동 매핑 (images -> ground_truth)
        # 파일명 패턴: IMG_1.jpg -> GT_IMG_1.mat
        if 'images' in img_path:
            gd_path = img_path.replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_').replace('.jpg', '.mat')
        else:
            # 단순 폴더 구조일 경우
            dirname = os.path.dirname(img_path)
            basename = os.path.basename(img_path)
            gd_path = os.path.join(dirname.replace('images', 'ground_truth'), 'GT_' + basename.replace('.jpg', '.mat'))
        
        img = Image.open(img_path).convert('RGB')
        
        # GT 로드 (.mat 또는 .txt)
        if os.path.exists(gd_path):
            mat = sio.loadmat(gd_path)
            try:
                points = mat['image_info'][0, 0][0, 0][0]
            except:
                points = mat['location']
        else:
            # Fallback for .txt
            txt_path = gd_path.replace('.mat', '.txt')
            points = []
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    for line in f:
                        line = line.strip().replace(',', ' ').split()
                        if line: points.append([float(line[0]), float(line[1])])
            points = np.array(points)

        # -----------------------------------------------------------
        # [핵심 로직] NPoint 적용 (Train 단계에서 Crop 하기 전에 전체 이미지 기준 적용)
        # -----------------------------------------------------------
        if self.use_npoint:
            w, h = img.size
            # 적응형 조건 (나 포함 7명 이상일 때만)
            if self.adaptive_npoint == 0 or len(points) >= self.adaptive_npoint:
                points = apply_npoint(points, (h, w), alpha=self.alpha, k=6)
        # -----------------------------------------------------------

        if self.method == 'train':
            return self.train_transform(img, points)
        elif self.method == 'val':
            return self.val_transform(img, points, img_path)

    def train_transform(self, img, keypoints):
        w, h = img.size
        st_size = 1.0 * min(w, h)
        
        # Random Resize (Scale Augmentation)
        if st_size < self.crop_size:
            scale = 1.0 + (self.crop_size / st_size)
            w, h = int(w*scale), int(h*scale)
            img = img.resize((w, h), Image.BICUBIC)
            keypoints = keypoints * scale
        
        # Random Crop
        i, j, h, w = random_crop(h, w, self.crop_size, self.crop_size)
        img = transforms.functional.crop(img, i, j, h, w)
        
        # Crop 범위 내 포인트만 필터링
        if len(keypoints) > 0:
            idx = (keypoints[:, 0] >= j) & (keypoints[:, 0] < j + w) & \
                  (keypoints[:, 1] >= i) & (keypoints[:, 1] < i + h)
            keypoints = keypoints[idx]
            
            # 좌표를 Crop 기준 상대 좌표로 변환
            keypoints[:, 0] -= j
            keypoints[:, 1] -= i
        
        # Random Flip
        if random.random() > 0.5:
            img = transforms.functional.hflip(img)
            if len(keypoints) > 0:
                keypoints[:, 0] = w - keypoints[:, 0]
            
        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.tensor([1.0]) # Dummy scale

    def val_transform(self, img, keypoints, img_path):
        # Validation은 원본 크기 사용
        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), img_path