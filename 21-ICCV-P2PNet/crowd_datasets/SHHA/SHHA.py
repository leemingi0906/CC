import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob
import scipy.io as io

# [추가 1] NPoint 모듈 임포트
try:
    from npoint_aug import apply_npoint
except ImportError:
    # 경로 문제 발생 시 상위 폴더 추가
    import sys
    sys.path.append("..") 
    from npoint_aug import apply_npoint

class SHHA(Dataset):
    # [수정 2] __init__에 use_npoint 추가
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False, use_npoint=False):
        self.root_path = data_root
        self.train_lists = "shanghai_tech_part_a_train.list"
        self.eval_list = "shanghai_tech_part_a_test.list"
        
        # use_npoint 옵션 저장
        self.use_npoint = use_npoint 

        # ... (기존 파일 리스트 로딩 코드 유지) ...
        self.img_list_file = self.train_lists.split(',')
        if train:
            self.img_list_file = self.train_lists.split(',')
        else:
            self.img_list_file = self.eval_list.split(',')

        self.img_map = {}
        self.img_list = []
        for _, train_list in enumerate(self.img_list_file):
            train_list = train_list.strip()
            with open(os.path.join(self.root_path, train_list)) as fin:
                for line in fin:
                    if len(line) < 2: 
                        continue
                    line = line.strip().split()
                    self.img_map[os.path.join(self.root_path, line[0].strip())] = \
                                    os.path.join(self.root_path, line[1].strip())
        self.img_list = sorted(list(self.img_map.keys()))
        self.nSamples = len(self.img_list)
        
        self.transform = transform
        self.train = train
        self.patch = patch
        self.flip = flip

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.img_list[index]
        gt_path = self.img_map[img_path]
        
        # load image and ground truth
        img, point = load_data((img_path, gt_path), self.train)
        
        # ========================================================
        # [추가 3] NPoint 적용 (학습 모드 & 옵션 켜짐 확인)
        # ========================================================
        if self.train and self.use_npoint:
            # 이미지 크기 (H, W) -> load_data에서 img는 PIL Image임
            w, h = img.size 
            
            # NPoint 적용 
            # point는 numpy array여야 함 (load_data가 numpy로 반환함)
            point = apply_npoint(point, (h, w), k=6)
        # ========================================================

        # apply augmentation (기존 Transform)
        if self.transform is not None:
            img = self.transform(img)

        if self.train:
            # data augmentation -> random scale
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            
            # scale the image and points
            if scale * min_size > 128:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                point *= scale
                
            # random crop augmentation
            if self.train and self.patch:
                img, point = random_crop(img, point)
                for i, _ in enumerate(point):
                    point[i] = torch.Tensor(point[i])
                    
            # random flipping
            if random.random() > 0.5 and self.train and self.flip:
                img = torch.Tensor(img[:, :, :, ::-1].copy())
                for i, _ in enumerate(point):
                    point[i][:, 0] = 128 - point[i][:, 0]

        if not self.train:
            point = [point]

        img = torch.Tensor(img)
        
        # pack up related infos
        target = [{} for i in range(len(point))]
        for i, _ in enumerate(point):
            target[i]['point'] = torch.Tensor(point[i])
            image_id = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            target[i]['labels'] = torch.ones([point[i].shape[0]]).long()

        return img, target

# ... (load_data, random_crop 함수 등 기존 코드 유지) ...
def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    # load the images
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # load ground truth points
    points = []
    with open(gt_path) as f_label:
        for line in f_label:
            line = line.strip().split()
            x = float(line[0])
            y = float(line[1])
            points.append([x, y])

    return img, np.array(points)

def random_crop(img, den, num_patch=4):
    half_h = 128
    half_w = 128
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
    result_den = []
    for i in range(num_patch):
        start_h = random.randint(0, img.size(1) - half_h)
        start_w = random.randint(0, img.size(2) - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        result_img[i] = img[:, start_h:end_h, start_w:end_w]
        idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
        record_den = den[idx]
        record_den[:, 0] -= start_w
        record_den[:, 1] -= start_h
        result_den.append(record_den)
    return result_img, result_den

