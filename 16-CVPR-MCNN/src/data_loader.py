import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob
import scipy.io as io
from scipy.ndimage import gaussian_filter

# NPoint 모듈 임포트 (npoint_aug.py가 src 폴더에 있어야 함)
try:
    from .npoint_aug import apply_npoint
except ImportError:
    try:
        from npoint_aug import apply_npoint
    except ImportError:
        # 파일이 없을 경우를 대비한 더미 함수
        def apply_npoint(points, *args, **kwargs): return points

class MCNN_SHT_Dataset(Dataset):
    """
    MCNN 전용 데이터 로더 (클래스명: MCNN_SHT_Dataset)
    """
    def __init__(self, data_root, part='B', phase='train', transform=None, 
                 use_npoint=False, alpha=0.2, adaptive_npoint=7):
        self.data_root = data_root
        self.use_npoint = use_npoint
        self.alpha = alpha
        self.adaptive_npoint = adaptive_npoint
        self.transform = transform
        self.phase = phase
        
        # 상하이텍 표준 경로 설정
        part_name = f'part_{part}_final'
        mode_name = f'{phase}_data'
        
        # 이미지와 GT 경로 조합
        self.img_dir = os.path.join(data_root, part_name, mode_name, 'images')
        self.gt_dir = os.path.join(data_root, part_name, mode_name, 'ground_truth')
        
        # 이미지 리스트 확보 (.jpg)
        self.img_list = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))
        
        if len(self.img_list) == 0:
            # 경로가 다를 경우를 대비한 대체 경로 탐색
            self.img_dir = os.path.join(data_root, mode_name, 'images')
            self.gt_dir = os.path.join(data_root, mode_name, 'ground_truth')
            self.img_list = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))

        print(f"📊 [DataLoader] {part}-{phase} : {len(self.img_list)} images loaded.")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        bname = os.path.basename(img_path)
        
        # GT 파일 매핑 (GT_IMG_1.mat 등)
        gt_path = os.path.join(self.gt_dir, 'GT_' + bname.replace('.jpg', '.mat'))
        if not os.path.exists(gt_path):
            gt_path = gt_path.replace('.mat', '.txt')

        # 1. 원본 데이터 로드
        img_raw = cv2.imread(img_path)
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        h, w, _ = img_raw.shape
        points = self.load_gt(gt_path)
        
        # 2. NPoint 증강 (훈련 시에만)
        if self.phase == 'train' and self.use_npoint:
            if len(points) >= self.adaptive_npoint:
                points = apply_npoint(points, (h, w), alpha=self.alpha, k=6)

        # 3. Density Map 생성 (MCNN은 1/4 크기 정답 사용)
        ds_factor = 4
        # MCNN은 32의 배수 크기 입력 권장
        target_h, target_w = (h // 32) * 32, (w // 32) * 32
        img_resized = cv2.resize(img_raw, (target_w, target_h))
        
        gt_map = self.generate_density_map(target_h // ds_factor, target_w // ds_factor, points / ds_factor)
        
        # 4. 최종 변환
        img_pil = Image.fromarray(img_resized)
        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
            
        gt_tensor = torch.from_numpy(gt_map).unsqueeze(0)
        
        return img_tensor, gt_tensor

    def load_gt(self, gt_path):
        if gt_path.endswith('.mat'):
            try:
                mat = io.loadmat(gt_path)
                points = mat['image_info'][0, 0][0, 0][0]
            except:
                mat = io.loadmat(gt_path)
                points = mat['location'] if 'location' in mat else []
        else:
            points = []
            with open(gt_path, 'r', errors='ignore') as f:
                for line in f:
                    line = line.strip().replace(',', ' ').split()
                    if line: points.append([float(line[0]), float(line[1])])
        return np.array(points)

    def generate_density_map(self, h, w, points, sigma=4):
        d_map = np.zeros((h, w), dtype=np.float32)
        for p in points:
            x, y = int(p[0]), int(p[1])
            if 0 <= x < w and 0 <= y < h:
                d_map[y, x] = 1.0
        return gaussian_filter(d_map, sigma=sigma)