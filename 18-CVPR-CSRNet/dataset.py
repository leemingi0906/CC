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

# NPoint 모듈 임포트 (npoint_aug.py가 같은 폴더에 있어야 함)
try:
    from npoint_aug import apply_npoint
except ImportError:
    # 파일이 없을 경우를 대비한 더미 함수
    def apply_npoint(points, *args, **kwargs):
        return points.copy()

class CSRNet_Dataset(Dataset):
    """
    CSRNet 전용 패치 적응형 데이터 로더
    - 훈련 시 Random Crop 수행 후, 패치 내 인원수에 따라 NPoint 적용 여부 결정
    - CSRNet 구조에 맞춰 1/8 크기의 Density Map 생성
    """
    def __init__(self, data_root, part='B', phase='train', transform=None, 
                 use_npoint=False, alpha=0.0, adaptive_npoint=7, crop_size=400):
        self.data_root = data_root
        self.use_npoint = use_npoint
        self.alpha = alpha
        self.adaptive_npoint = adaptive_npoint # 패치 내 최소 인원수 (ex: 7)
        self.transform = transform
        self.phase = phase
        self.crop_size = crop_size # 패치 크기 (CSRNet은 8의 배수 권장)
        
        # 경로 설정 (ShanghaiTech 표준 구조)
        part_name = f'part_{part}_final'
        mode_name = f'{phase}_data'
        
        self.img_dir = os.path.join(data_root, part_name, mode_name, 'images')
        self.gt_dir = os.path.join(data_root, part_name, mode_name, 'ground_truth')
        
        # 심볼릭 링크 등 다양한 경로 구조 대응
        self.img_list = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))
        if len(self.img_list) == 0:
            self.img_dir = os.path.join(data_root, mode_name, 'images')
            self.gt_dir = os.path.join(data_root, mode_name, 'ground_truth')
            self.img_list = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))

        print(f"📊 [CSRNet] {part}-{phase} 로드 완료: {len(self.img_list)} images.")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        bname = os.path.basename(img_path)
        
        # GT 파일 매핑 (.mat 또는 .txt)
        gt_path = os.path.join(self.gt_dir, 'GT_' + bname.replace('.jpg', '.mat'))
        if not os.path.exists(gt_path):
            gt_path = gt_path.replace('.mat', '.txt')

        # 1. 원본 데이터 로드
        img_raw = cv2.imread(img_path)
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        h, w, _ = img_raw.shape
        points = self.load_gt(gt_path)

        # 2. [핵심] 훈련 시 패치 기반 적응형 처리
        if self.phase == 'train':
            # 이미지 크기가 패치보다 작을 경우 검은색 패딩 처리
            if h < self.crop_size or w < self.crop_size:
                pad_h = max(0, self.crop_size - h)
                pad_w = max(0, self.crop_size - w)
                img_raw = cv2.copyMakeBorder(img_raw, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
                h, w, _ = img_raw.shape

            # 랜덤 좌표 선택 (Random Crop)
            y1 = random.randint(0, h - self.crop_size)
            x1 = random.randint(0, w - self.crop_size)
            y2 = y1 + self.crop_size
            x2 = x1 + self.crop_size

            # 이미지 크롭
            img_patch = img_raw[y1:y2, x1:x2]
            
            # 패치 영역 안의 포인트만 필터링
            idx = (points[:, 0] >= x1) & (points[:, 0] < x2) & \
                  (points[:, 1] >= y1) & (points[:, 1] < y2)
            points_in_patch = points[idx].copy()
            
            # 좌표를 패치 상대 좌표로 변환
            points_in_patch[:, 0] -= x1
            points_in_patch[:, 1] -= y1

            # [NPoint 적응형 로직] 패치 내 인원수가 기준(T) 이상일 때만 수행
            if self.use_npoint and len(points_in_patch) >= self.adaptive_npoint:
                # 패치 크기 내에서 노이즈 주입
                points_in_patch = apply_npoint(points_in_patch, (self.crop_size, self.crop_size), 
                                               alpha=self.alpha, k=6)
            
            final_img = Image.fromarray(img_patch)
            final_points = points_in_patch
            cur_h, cur_w = self.crop_size, self.crop_size
        else:
            # 테스트 시에는 원본 전체 사용
            final_img = Image.fromarray(img_raw)
            final_points = points
            cur_h, cur_w = h, w

        # 3. Density Map 생성 (CSRNet Output Stride = 8)
        ds_factor = 8
        # Density Map 크기는 입력의 1/8
        target_h, target_w = cur_h // ds_factor, cur_w // ds_factor
        
        # 좌표값도 1/8로 스케일링하여 맵 생성
        gt_map = self.generate_density_map(target_h, target_w, final_points / ds_factor, sigma=4)
        
        # 4. 텐서 변환 및 정규화
        if self.transform:
            img_tensor = self.transform(final_img)
        else:
            # 기본 정규화 (ImageNet 기준)
            t = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = t(final_img)
            
        gt_tensor = torch.from_numpy(gt_map).unsqueeze(0)
        
        return img_tensor, gt_tensor

    def load_gt(self, gt_path):
        """ .mat 또는 .txt 파일에서 좌표 로드 """
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
        """ 가우시안 커널을 사용한 밀도 맵 생성 """
        d_map = np.zeros((h, w), dtype=np.float32)
        if len(points) > 0:
            for p in points:
                x, y = int(p[0]), int(p[1])
                # 다운샘플링된 좌표가 맵 범위 내에 있는지 확인
                if 0 <= x < w and 0 <= y < h:
                    d_map[y, x] = 1.0
        return gaussian_filter(d_map, sigma=sigma)