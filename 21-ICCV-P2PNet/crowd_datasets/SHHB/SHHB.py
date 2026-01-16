import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import scipy.io as io

# NPoint 모듈 임포트
try:
    from npoint_aug import apply_npoint
except ImportError:
    import sys
    sys.path.append("..") 
    from npoint_aug import apply_npoint

class SHHB(Dataset):
    """
    ShanghaiTech Part B 데이터셋 로더
    - 저밀도/희소 구역(거리 풍경 등)에 최적화됨
    """
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False, use_npoint=False, alpha=0.2):
        self.root_path = data_root
        
        # Part B 전용 설정
        self.train_lists = "shanghai_tech_part_b_train.list"
        self.eval_list = "shanghai_tech_part_b_test.list"
        self.sub_path = 'part_B_final'
        
        self.use_npoint = use_npoint 
        self.alpha = alpha
        self.train = train
        self.patch = patch
        self.flip = flip
        self.transform = transform

        # 경로 결정
        mode_path = 'train_data' if train else 'test_data'
        target_list = self.train_lists if train else self.eval_list
        
        self.img_map = {}
        self.img_list = []
        
        list_path = os.path.join(self.root_path, target_list)
        if not os.path.exists(list_path):
            img_dir = os.path.join(self.root_path, self.sub_path, mode_path, 'images')
            gt_dir = os.path.join(self.root_path, self.sub_path, mode_path, 'ground_truth')
            import glob
            img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
            for img_p in img_files:
                bname = os.path.basename(img_p)
                gt_p = os.path.join(gt_dir, "GT_" + bname.replace(".jpg", ".mat"))
                self.img_map[img_p] = gt_p
        else:
            with open(list_path) as fin:
                for line in fin:
                    if len(line) < 2: continue
                    line = line.strip().split()
                    img_name = os.path.basename(line[0].strip())
                    gt_name = os.path.basename(line[1].strip())
                    real_img_path = os.path.join(self.root_path, self.sub_path, mode_path, 'images', img_name)
                    real_gt_path = os.path.join(self.root_path, self.sub_path, mode_path, 'ground_truth', gt_name)
                    self.img_map[real_img_path] = real_gt_path
                    
        self.img_list = sorted(list(self.img_map.keys()))
        self.nSamples = len(self.img_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        img_path = self.img_list[index]
        gt_path = self.img_map[img_path]
        
        img, point = load_data((img_path, gt_path))
        
        if self.train and self.use_npoint:
            w, h = img.size 
            point = apply_npoint(point, (h, w), alpha=self.alpha, k=4)

        if self.transform is not None:
            img = self.transform(img)

        if self.train:
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            if scale * min_size > 128:
                img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale, mode='bilinear').squeeze(0)
                point *= scale
            if self.patch:
                img, point = random_crop(img, point)
            if self.flip and random.random() > 0.5:
                img = torch.flip(img, dims=[-1])
                if isinstance(point, list):
                    for p in point: p[:, 0] = 128 - p[:, 0]
                else:
                    point[:, 0] = img.shape[-1] - point[:, 0]

        if not self.train:
            point = [point]

        target = []
        for i in range(len(point)):
            d = {'point': torch.tensor(point[i], dtype=torch.float32),
                 'labels': torch.ones([point[i].shape[0]], dtype=torch.int64)}
            target.append(d)
        return img, target

def load_data(img_gt_path):
    img_path, gt_path = img_gt_path
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    points = []
    try:
        mat = io.loadmat(gt_path)
        if 'image_info' in mat:
            points = mat['image_info'][0, 0][0, 0][0]
        elif 'location' in mat:
            points = mat['location']
    except:
        with open(gt_path) as f:
            for line in f:
                line = line.strip().split()
                if line: points.append([float(line[0]), float(line[1])])
    return img, np.array(points)

def random_crop(img, den, num_patch=4):
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