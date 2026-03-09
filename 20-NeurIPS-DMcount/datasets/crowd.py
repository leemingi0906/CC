import torch.utils.data as data
import os
from glob import glob
import torch
from torchvision import transforms
import random
import numpy as np
from PIL import Image
import scipy.io as sio

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
                 dataset_name='sha',
                 alpha=0.0,
                 adaptive_npoint=7,
                 file_list=None): # [추가] 외부에서 파일 리스트를 직접 받을 수 있음
        
        self.root_path = root_path
        self.crop_size = crop_size
        self.method = method
        self.dataset_name = dataset_name
        self.alpha = alpha
        self.adaptive_npoint = adaptive_npoint
        self.use_npoint = (self.method == 'train' and self.alpha > 0)

        # -----------------------------------------------------------
        # [이미지 리스트 로드 로직 개선]
        # -----------------------------------------------------------
        if file_list is not None:
            # 1. 외부에서 리스트를 직접 받은 경우 (UCF-CC-50 5-fold 등)
            self.im_list = file_list
            print(f"📊 [{self.dataset_name.upper()}] Loaded {len(self.im_list)} images from provided list.")
        else:
            # 2. 기존 방식 (폴더 탐색)
            # 대상 폴더 결정
            if self.dataset_name in ['sha', 'shb'] and os.path.exists(os.path.join(self.root_path, 'images')):
                target_dir = os.path.join(self.root_path, 'images')
            else:
                target_dir = self.root_path

            # 이미지 검색
            extensions = ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg']
            self.im_list = []
            for ext in extensions:
                self.im_list.extend(glob(os.path.join(target_dir, ext)))
            self.im_list = sorted(list(set(self.im_list)))

            if method == 'train':
                print(f"📊 [{self.dataset_name.upper()}] Loaded {len(self.im_list)} images from {os.path.abspath(target_dir)}")

        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        
        points = None
        
        # 1. UCF-QNRF (원래는 .npy를 기대하나, .mat도 유연하게 지원)
        if self.dataset_name == 'qnrf':
            base_name = os.path.splitext(img_path)[0]
            gt_path = base_name + '_ann.mat'
            if not os.path.exists(gt_path): 
                gt_path = base_name + '.npy'
                
            if os.path.exists(gt_path):
                if gt_path.endswith('.npy'):
                    points = np.load(gt_path)
                else:
                    try:
                        mat = sio.loadmat(gt_path)
                        points = mat['annPoints']
                    except:
                        points = np.array([])
            else:
                points = np.array([])

        # 2. UCF-CC-50
        elif self.dataset_name == 'cc50':
            base_name = os.path.splitext(img_path)[0]
            gt_path = base_name + '_ann.mat'
            if not os.path.exists(gt_path):
                gt_path = base_name + '.mat'
                
            if os.path.exists(gt_path):
                try:
                    mat = sio.loadmat(gt_path)
                    points = mat['annPoints']
                except:
                    points = np.array([])
            else:
                points = np.array([])

        # 3. JHU++
        elif self.dataset_name == 'jhu':
            gt_path = os.path.splitext(img_path)[0] + '.npy'
            if os.path.exists(gt_path):
                try:
                    points = np.load(gt_path)
                    if len(points.shape) == 1:
                        points = points.reshape(-1, 2) if len(points) > 0 else np.zeros((0, 2))
                    elif len(points.shape) >= 2:
                        points = points[:, :2]
                except:
                    points = np.array([])
            else:
                points = np.array([])

        # 4. Others (ShanghaiTech)
        else:
            if 'images' in img_path:
                gt_path = img_path.replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_').replace('.jpg', '.mat').replace('.JPG', '.mat')
            else:
                gt_path = img_path.replace('.jpg', '.mat').replace('.JPG', '.mat')
                
            if os.path.exists(gt_path):
                try:
                    mat = sio.loadmat(gt_path)
                    if 'image_info' in mat:
                        points = mat['image_info'][0, 0][0, 0][0]
                    elif 'location' in mat:
                        points = mat['location']
                    elif 'annPoints' in mat:
                        points = mat['annPoints']
                except:
                    points = np.array([])
            else:
                points = np.array([])

        points = np.array(points)
        
        # NPoint Augmentation
        try:
            from npoint_aug import apply_npoint
            if self.use_npoint:
                img_temp = Image.open(img_path)
                w, h = img_temp.size
                # 포인트가 없거나 너무 적으면 적용 스킵
                if len(points) >= self.adaptive_npoint:
                     points = apply_npoint(points, (h, w), alpha=self.alpha, k=6)
        except ImportError:
            pass

        img = Image.open(img_path).convert('RGB')

        if self.method == 'train':
            return self.train_transform(img, points)
        elif self.method == 'val':
            return self.val_transform(img, points, img_path)

    def train_transform(self, img, keypoints):
        w, h = img.size
        st_size = 1.0 * min(w, h)
        
        if st_size < self.crop_size:
            scale = 1.0 + (self.crop_size / st_size)
            w, h = int(w*scale), int(h*scale)
            img = img.resize((w, h), Image.BICUBIC)
            keypoints = keypoints * scale
        
        i, j, h, w = random_crop(h, w, self.crop_size, self.crop_size)
        img = transforms.functional.crop(img, i, j, h, w)
        
        if len(keypoints) > 0:
            idx = (keypoints[:, 0] >= j) & (keypoints[:, 0] < j + w) & \
                  (keypoints[:, 1] >= i) & (keypoints[:, 1] < i + h)
            keypoints = keypoints[idx]
            keypoints[:, 0] -= j
            keypoints[:, 1] -= i
        
        if random.random() > 0.5:
            img = transforms.functional.hflip(img)
            if len(keypoints) > 0:
                keypoints[:, 0] = w - keypoints[:, 0]
            
        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.tensor([1.0])

    def val_transform(self, img, keypoints, img_path):
        w, h = img.size
        # [최적화] 거대한 이미지의 경우 Test 시 줄여 OOM 예방
        # 주의: 16의 배수로 맞춰서 네트워크를 안 터지게 해야함
        scale = 1.0
        if max(h, w) > 1500:
            scale = 1500.0 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            # 16의 배수로 강제 정렬 (Pooling Layer Issue)
            new_w = (new_w // 16) * 16
            new_h = (new_h // 16) * 16
            # 비율 재조정
            scale_w = new_w / w
            scale_h = new_h / h
            img = img.resize((new_w, new_h), Image.BICUBIC)
            keypoints[:, 0] = keypoints[:, 0] * scale_w
            keypoints[:, 1] = keypoints[:, 1] * scale_h

        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), img_path