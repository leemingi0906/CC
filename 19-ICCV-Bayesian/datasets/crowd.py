from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np
import scipy.io as sio
from scipy.spatial import KDTree

try:
    from .npoint_aug import apply_npoint
except ImportError:
    def apply_npoint(points, *args, **kwargs): return points

def get_nn_distance(points):
    if len(points) <= 1: return np.zeros(len(points))
    tree = KDTree(points)
    distances, _ = tree.query(points, k=2)
    return distances[:, 1]

def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w

def cal_innner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right-inner_left, 0.0) * np.maximum(inner_down-inner_up, 0.0)
    return inner_area

class Crowd(data.Dataset):
    def __init__(self, args, method='train'):
        self.root_path = args.data_root
        self.dataset_name = args.dataset.upper()
        self.method = method
        
        self.c_size = args.crop_size
        self.d_ratio = args.downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio
        
        self.alpha = getattr(args, 'alpha', 0.0)
        self.use_npoint = (self.alpha > 0 and method == 'train')

        if args.is_gray:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # -----------------------------------------------------------
        # [데이터셋 패스 탐색]
        # -----------------------------------------------------------
        self.im_list = []
        if self.dataset_name in ['SHA', 'SHB', 'SHT']:
            sub_dir = 'train_data' if method == 'train' else 'test_data'
            # Check for standard paths vs native paths
            part = self.dataset_name[-1] if self.dataset_name in ['SHA', 'SHB'] else 'B'
            opt1 = os.path.join(self.root_path, f'part_{part}_final', sub_dir, 'images')
            opt2 = os.path.join(self.root_path, sub_dir, 'images')
            target_dir = opt1 if os.path.exists(opt1) else (opt2 if os.path.exists(opt2) else self.root_path)
            self.im_list = sorted(glob(os.path.join(target_dir, '*.jpg')))
            
        elif self.dataset_name == 'QNRF':
            sub_dir = 'Train' if method == 'train' else 'Test'
            opt1 = os.path.join(self.root_path, sub_dir)
            target_dir = opt1 if os.path.exists(opt1) else self.root_path
            self.im_list = sorted(glob(os.path.join(target_dir, '*.jpg')))
            
        elif self.dataset_name == 'CC50':
            all_files = sorted(glob(os.path.join(self.root_path, '*.jpg')))
            if not all_files:
                all_files = sorted(glob(os.path.join(self.root_path, 'UCF_CC_50', '*.jpg')))
                
            fold_size = max(1, len(all_files) // 5)
            test_fold = min(getattr(args, 'test_fold', 0), 4)
            start_idx = test_fold * fold_size
            end_idx = start_idx + fold_size if test_fold < 4 else len(all_files)

            if method == 'train': 
                self.im_list = all_files[:start_idx] + all_files[end_idx:]
            else: 
                self.im_list = all_files[start_idx:end_idx]
                
        elif self.dataset_name == 'JHU':
            mode_name = method
            opt1 = os.path.join(self.root_path, mode_name)
            target_dir = opt1 if os.path.exists(opt1) else os.path.join(self.root_path, 'JHU_Processed', mode_name)
            self.im_list = sorted(glob(os.path.join(target_dir, '*.jpg')))

        print(f"📊 [Bayesian] ({method}) {self.dataset_name} Loaded: {len(self.im_list)} images")

    def __len__(self):
        return len(self.im_list)

    def load_gt(self, img_path):
        points = []
        if self.dataset_name in ['SHA', 'SHB', 'SHT']:
            if 'images' in img_path:
                gt_path = img_path.replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_').replace('.jpg', '.mat')
            else:
                gt_path = img_path.replace('.jpg', '.mat')
                
            if os.path.exists(gt_path):
                try: points = sio.loadmat(gt_path)['image_info'][0, 0][0, 0][0]
                except: pass
                
        elif self.dataset_name == 'QNRF':
            gt_path = img_path.replace('.jpg', '_ann.mat')
            if os.path.exists(gt_path):
                try: points = sio.loadmat(gt_path)['annPoints']
                except: pass
                
        elif self.dataset_name == 'CC50':
            gt_path = img_path.replace('.jpg', '_ann.mat')
            if not os.path.exists(gt_path): gt_path = img_path.replace('.jpg', '.mat')
            if os.path.exists(gt_path):
                try: points = sio.loadmat(gt_path)['annPoints']
                except: pass
                
        elif self.dataset_name == 'JHU':
            gt_path = img_path.replace('.jpg', '.npy')
            if os.path.exists(gt_path):
                try:
                    points = np.load(gt_path)
                    if len(points.shape) == 1:
                        points = points.reshape(-1, 2) if len(points) > 0 else np.zeros((0, 2))
                    elif len(points.shape) >= 2:
                        points = points[:, :2]
                except: pass

        points = np.array(points) if len(points) > 0 else np.zeros((0, 2))
        return points

    def __getitem__(self, item):
        img_path = self.im_list[item]
        img = Image.open(img_path).convert('RGB')
        points = self.load_gt(img_path)

        # -----------------------------------------------------------
        # [NPoint 증강 적용] - 항상 Float를 유지
        # -----------------------------------------------------------
        if self.use_npoint and len(points) > 0:
            w, h = img.size
            points = apply_npoint(points, (h, w), alpha=float(self.alpha), k=6)
            points = np.array(points, dtype=np.float32)

        # -----------------------------------------------------------
        # [Bayesian 전용 전처리]
        # -----------------------------------------------------------
        if len(points) > 0:
            distances = get_nn_distance(points)
            keypoints = np.concatenate([points, distances[:, np.newaxis]], axis=1)
        else:
            keypoints = np.zeros((0, 3))

        if self.method == 'train':
            return self.train_transform(img, keypoints)
        elif self.method in ['val', 'test']:
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, len(keypoints), name

    def train_transform(self, img, keypoints):
        wd, ht = img.size
        st_size = min(wd, ht)
        
        # 작은 이미지 방어 코드 (QNRF/CC50 대비)
        if st_size < self.c_size:
            scale = 1.0 + (self.c_size / st_size)
            wd, ht = int(wd * scale), int(ht * scale)
            img = img.resize((wd, ht), Image.BICUBIC)
            if len(keypoints) > 0:
                keypoints[:, :2] = keypoints[:, :2] * scale
                keypoints[:, 2] = keypoints[:, 2] * scale # 거리도 비례 스케일링
                
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        
        if len(keypoints) == 0:
            return self.trans(img), torch.zeros(0, 2), torch.zeros(0), st_size

        nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)
        points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
        points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
        
        bbox = np.concatenate((points_left_up, points_right_down), axis=1)
        inner_area = cal_innner_area(j, i, j+w, i+h, bbox)
        origin_area = nearest_dis * nearest_dis
        ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
        
        mask = (ratio >= 0.3)
        target = ratio[mask]
        keypoints = keypoints[mask]
        
        # 좌표 이동
        keypoints = keypoints[:, :2] - [j, i]
        
        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                
        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), \
               torch.from_numpy(target.copy()).float(), st_size
