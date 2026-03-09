import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob
import scipy.io as io
from scipy.ndimage import gaussian_filter
from torchvision import transforms

# NPoint 모듈 임포트
try:
    from npoint_aug import apply_npoint
except ImportError:
    def apply_npoint(points, *args, **kwargs):
        return points.copy()

class CSRNet_Dataset(Dataset):
    def __init__(self, data_root, dataset_name='SHT', part='B', phase='train', transform=None, 
                 use_npoint=False, alpha=0.0, adaptive_npoint=7, crop_size=400, max_shift=25, test_fold=0):
        
        self.data_root = data_root
        self.dataset_name = dataset_name.upper()
        self.phase = str(phase).strip().lower()
        self.part = part
        self.transform = transform
        self.test_fold = test_fold
        
        # 최적화된 NPoint 설정
        self.use_npoint = use_npoint
        self.alpha = alpha 
        self.adaptive_npoint = adaptive_npoint
        self.crop_size = crop_size
        self.max_shift = max_shift
        
        self.downsample_large_images = True
        self.img_list = []

        # 데이터셋 경로 설정 로직
        if self.dataset_name == 'SHT':
            part_name = f'part_{part}_final'
            mode_name = f'{self.phase}_data'
            
            # [수정] data_root가 이미 SHT 폴더 자체일 때 대비
            if os.path.exists(os.path.join(data_root, part_name, mode_name)):
                base_dir = os.path.join(data_root, part_name, mode_name)
            elif os.path.exists(os.path.join(data_root, f'part_{part}', mode_name)):
                base_dir = os.path.join(data_root, f'part_{part}', mode_name)
            else:
                base_dir = os.path.join(data_root, 'SHT', part_name, mode_name)
                if not os.path.exists(base_dir):
                    base_dir = os.path.join(data_root, 'SHT', f'part_{part}', mode_name)
                    
            self.img_dir = os.path.join(base_dir, 'images')
            self.gt_dir = os.path.join(base_dir, 'ground_truth')
            self.img_list = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))
            self.dataset_type = 'sht'
            
        elif self.dataset_name == 'QNRF':
            mode_name = 'Train' if self.phase == 'train' else 'Test'
            
            # [수정] data_root가 이미 QNRF 폴더를 가리키는 경우 직결 (예: ./preprocess/UCF-QNRF_Processed)
            if os.path.exists(os.path.join(data_root, mode_name)):
                base_dir = os.path.join(data_root, mode_name)
            else:
                base_dir = os.path.join(data_root, 'UCF', 'UCF-QNRF_ECCV18', mode_name)
                if not os.path.exists(base_dir):
                    base_dir = os.path.join(data_root, 'UCF', 'UCF-QNRF', mode_name)
                    
            self.img_list = sorted(glob.glob(os.path.join(base_dir, '*.jpg')))
            self.dataset_type = 'qnrf'
            
        elif self.dataset_name == 'JHU':
            mode_name = self.phase
            
            if os.path.exists(os.path.join(data_root, mode_name)):
                base_dir = os.path.join(data_root, mode_name)
            else:
                base_dir = os.path.join(data_root, 'JHU_Processed', mode_name)
                
            self.img_list = sorted(glob.glob(os.path.join(base_dir, '*.jpg')))
            self.dataset_type = 'jhu'
            
        elif self.dataset_name == 'CC50':
            # [수정] data_root가 직접 CC50 폴더를 가리킬 때 대비
            if any(fname.endswith('.jpg') for fname in os.listdir(data_root)):
                base_dir = data_root
            else:
                base_dir = os.path.join(data_root, 'UCF', 'UCF_CC_50')
                
            all_images = sorted(glob.glob(os.path.join(base_dir, '*.jpg')))
            
            # [기능추가] 5-Fold Cross Validation 지원
            fold_size = max(1, len(all_images) // 5)
            # test_fold 속성 확인 (train.py에서 전달됨, 없으면 0)
            test_fold = min(getattr(self, 'test_fold', 0), 4)
            start_idx = test_fold * fold_size
            end_idx = start_idx + fold_size if test_fold < 4 else len(all_images)

            if self.phase == 'train': 
                self.img_list = all_images[:start_idx] + all_images[end_idx:]
            else: 
                self.img_list = all_images[start_idx:end_idx]
                
            self.dataset_type = 'cc50'

        print(f"📊 [{self.dataset_name}] ({self.phase}) Loaded: {len(self.img_list)} images.")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        
        if self.dataset_type == 'sht':
            gt_path = os.path.join(self.gt_dir, 'GT_' + os.path.basename(img_path).replace('.jpg', '.mat'))
        elif self.dataset_type == 'jhu':
            gt_path = img_path.replace('.jpg', '.npy')
        else:
            gt_path = img_path.replace('.jpg', '_ann.mat')

        try:
            img_pil = Image.open(img_path).convert('RGB')
        except:
            img_pil = Image.new('RGB', (1024, 768), (0,0,0))

        img_raw = np.array(img_pil)
        h, w, _ = img_raw.shape
        points = self.load_gt(gt_path)

        # CSRNet Output Stride (다운샘플링 비율)
        ds_factor = 8

        if self.phase == 'train':
            # [최적화] 거대한 이미지의 경우 Train 시에도 Crop 전에 1차로 줄여 OOM 예방
            scale = 1.0
            if self.downsample_large_images:
                long_side = max(h, w)
                if long_side > 2000: scale = 2000.0 / long_side
                
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                img_raw = cv2.resize(img_raw, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # 가로 세로 각각의 정확한 실수 스케일 계산 후 포인트 축소 적용
                scale_w = new_w / w
                scale_h = new_h / h
                points[:, 0] *= scale_w
                points[:, 1] *= scale_h
                h, w, _ = img_raw.shape

            pad_h, pad_w = max(0, self.crop_size - h), max(0, self.crop_size - w)
            if pad_h > 0 or pad_w > 0:
                img_raw = cv2.copyMakeBorder(img_raw, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
                h, w, _ = img_raw.shape

            y1, x1 = random.randint(0, h - self.crop_size), random.randint(0, w - self.crop_size)
            img_patch = img_raw[y1:y1+self.crop_size, x1:x1+self.crop_size]
            
            if len(points) > 0:
                idx = (points[:, 0] >= x1) & (points[:, 0] < x1+self.crop_size) & \
                      (points[:, 1] >= y1) & (points[:, 1] < y1+self.crop_size)
                points_in_patch = points[idx].copy()
                points_in_patch[:, 0] -= x1
                points_in_patch[:, 1] -= y1
            else:
                points_in_patch = np.zeros((0, 2))

            # 최적화된 apply_npoint 호출
            if self.use_npoint and len(points_in_patch) >= self.adaptive_npoint:
                points_in_patch = apply_npoint(points_in_patch, (self.crop_size, self.crop_size), 
                                               alpha=self.alpha, max_shift=self.max_shift)
            
            final_img, final_points = Image.fromarray(img_patch), points_in_patch
            cur_h, cur_w = self.crop_size, self.crop_size
        else:
            # Test 시 OOM 방지 리사이징
            scale = 1.0
            if self.downsample_large_images:
                long_side = max(h, w)
                if long_side > 2500: scale = 2000.0 / long_side
            
            if scale < 1.0:
                # [개선 2] 리사이즈 후의 크기를 ds_factor(8)의 배수로 강제 정렬
                new_w = int(w * scale)
                new_h = int(h * scale)
                new_w = (new_w // ds_factor) * ds_factor
                new_h = (new_h // ds_factor) * ds_factor
                
                # 비율이 미세하게 바뀌었으므로 scale 재계산
                scale_w = new_w / w
                scale_h = new_h / h
                
                img_raw = cv2.resize(img_raw, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # 포인트 좌표도 각 축의 실제 스케일에 맞게 조절
                points[:, 0] *= scale_w
                points[:, 1] *= scale_h
                
                cur_h, cur_w = new_h, new_w
            else:
                # 스케일 변환이 없더라도 8의 배수 강제 맞춤 (Padding 대신 잘라내기)
                cur_h = (h // ds_factor) * ds_factor
                cur_w = (w // ds_factor) * ds_factor
                img_raw = img_raw[:cur_h, :cur_w]
                
                # 잘려나간 영역의 포인트 제거
                idx = (points[:, 0] < cur_w) & (points[:, 1] < cur_h)
                points = points[idx].copy()
                
            final_img, final_points = Image.fromarray(img_raw), points

        
        gt_map = self.generate_density_map(cur_h//ds_factor, cur_w//ds_factor, final_points/ds_factor)
        
        t = self.transform if self.transform else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return t(final_img), torch.from_numpy(gt_map).unsqueeze(0)

    def load_gt(self, gt_path):
        if not os.path.exists(gt_path): return np.array([])
        if gt_path.endswith('.npy'):
            points = np.load(gt_path)
            if len(points.shape) == 1:
                return points.reshape(-1, 2) if len(points) > 0 else np.zeros((0, 2))
            elif len(points.shape) >= 2:
                return points[:, :2]
            return np.zeros((0, 2))
        try:
            mat = io.loadmat(gt_path)
            if 'annPoints' in mat: return mat['annPoints']
            if 'image_info' in mat: return mat['image_info'][0,0][0,0][0]
            if 'location' in mat: return mat['location']
            for k in mat:
                if not k.startswith('__') and hasattr(mat[k], 'shape') and len(mat[k].shape)==2 and mat[k].shape[1]==2:
                    return mat[k]
        except: return np.array([])
        return np.array([])

    def generate_density_map(self, h, w, points, sigma=4):
        d_map = np.zeros((h, w), dtype=np.float32)
        for p in points:
            # [개선 1] 반올림을 사용하여 다운샘플링 시 위치 정확도 향상
            x, y = int(round(p[0])), int(round(p[1]))
            # [수정 완료] 누적 덧셈
            if 0 <= x < w and 0 <= y < h: d_map[y, x] += 1.0
        return gaussian_filter(d_map, sigma=sigma)
