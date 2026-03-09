import os
import glob
import torch
import numpy as np
import scipy.io as io
from torch.utils.data import Dataset
from PIL import Image
import random

# NPoint 모듈 임포트
try:
    from npoint_aug import apply_npoint
except ImportError:
    # 모듈이 없을 경우를 대비한 더미 함수
    def apply_npoint(points, img_shape, **kwargs): return points

class CrowdDataset(Dataset):
    def __init__(self, data_root, dataset_name, phase='train', transform=None, downsample_ratio=4, aug_alpha=1.0):
        self.data_root = data_root
        self.dataset_name = dataset_name.upper()
        self.phase = phase
        self.transform = transform
        self.downsample_ratio = downsample_ratio
        self.aug_alpha = aug_alpha
        self.data_files = []

        # 지원하는 모든 확장자 (대소문자 포함)
        ext_list = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']

        target_path = None

        # =========================================================
        # 1. ShanghaiTech (A / B)
        # 사용자 지정 구조: data_root/original/shanghaitech/part_A_final/train_data/images
        # =========================================================
        if self.dataset_name in ['A', 'B', 'SHA', 'SHB']:
            if self.dataset_name in ['A', 'SHA']:
                part_names = ['part_A_final', 'part_A']
            else:
                part_names = ['part_B_final', 'part_B']
            
            sub_dir = 'train_data' if phase == 'train' else 'test_data'
            
            path_candidates = []
            for pname in part_names:
                # [최우선] 사용자 지정 경로: data/original/shanghaitech/...
                path_candidates.append(os.path.join(data_root, 'original', 'shanghaitech', pname, sub_dir, 'images'))
                path_candidates.append(os.path.join(data_root, 'original', 'ShanghaiTech', pname, sub_dir, 'images'))
                
                # 그 외 일반적인 경로 패턴들
                path_candidates.append(os.path.join(data_root, 'original', pname, sub_dir, 'images'))
                path_candidates.append(os.path.join(data_root, 'shanghaitech', pname, sub_dir, 'images'))
                path_candidates.append(os.path.join(data_root, pname, sub_dir, 'images'))
            
            # data_root 자체가 해당 폴더인 경우
            path_candidates.append(os.path.join(data_root, sub_dir, 'images'))
            path_candidates.append(data_root)

            # 유효한 경로 탐색
            for opt in path_candidates:
                if os.path.exists(opt):
                    found_any = False
                    for ext in ext_list:
                        if len(glob.glob(os.path.join(opt, ext))) > 0:
                            found_any = True
                            break
                    if found_any:
                        target_path = opt
                        break

        # =========================================================
        # 2. UCF-QNRF (QNRF)
        # =========================================================
        elif self.dataset_name == 'QNRF':
            sub_dir = 'Train' if phase == 'train' else 'Test'
            path_options = [
                os.path.join(data_root, 'original', 'ucf', 'UCF-QNRF', sub_dir),
                os.path.join(data_root, 'UCF-QNRF', sub_dir),
                os.path.join(data_root, sub_dir)
            ]
            
            for opt in path_options:
                if os.path.exists(opt):
                    target_path = opt
                    break
            
            if target_path is None: target_path = data_root

        # =========================================================
        # 3. UCF_CC_50 (CC50)
        # =========================================================
        elif self.dataset_name == 'CC50':
            path_options = [
                os.path.join(data_root, 'original', 'ucf', 'UCF_CC_50'),
                os.path.join(data_root, 'UCF_CC_50'),
                data_root
            ]
            
            for opt in path_options:
                if os.path.exists(opt):
                    target_path = opt
                    break

        # =========================================================
        # 4. JHU++ (JHU)
        # =========================================================
        elif self.dataset_name == 'JHU':
            # expected folder structure: [train, val, test] directories inside data_root
            sub_dir = phase
            path_options = [
                os.path.join(data_root, sub_dir),
                os.path.join(data_root, 'JHU_Processed', sub_dir)
            ]
            
            for opt in path_options:
                if os.path.exists(opt):
                    target_path = opt
                    break

        # 최종 파일 리스트 수집
        if target_path and os.path.exists(target_path):
            for ext in ext_list:
                self.data_files.extend(glob.glob(os.path.join(target_path, ext)))
        
        self.data_files.sort()

        # CC50 전용 셔플 분할 로직
        if self.dataset_name == 'CC50' and len(self.data_files) > 0:
            np.random.seed(42)
            np.random.shuffle(self.data_files)
            if phase == 'train':
                self.data_files = self.data_files[:45]
            else:
                self.data_files = self.data_files[45:]

        print(f"[{self.dataset_name} | {phase}] Load Success: {len(self.data_files)} images (Path: {target_path})")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        img_path = self.data_files[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Image Load Failed: {img_path} ({e})")
            return torch.zeros(3, 512, 512), torch.zeros(1, 128, 128)

        points = self.load_gt_points(img_path)
        w, h = img.size

        # Train: Random Crop
        if self.phase == 'train':
            crop_size = 512
            if w > crop_size and h > crop_size:
                dx = random.randint(0, w - crop_size)
                dy = random.randint(0, h - crop_size)
                img = img.crop((dx, dy, dx + crop_size, dy + crop_size))
                if len(points) > 0:
                    mask = (points[:, 0] >= dx) & (points[:, 0] < dx + crop_size) & \
                           (points[:, 1] >= dy) & (points[:, 1] < dy + crop_size)
                    points = points[mask]
                    points[:, 0] -= dx
                    points[:, 1] -= dy
            else:
                img = img.resize((crop_size, crop_size), Image.BILINEAR)
                if len(points) > 0:
                    points[:, 0] = points[:, 0] * (crop_size / w)
                    points[:, 1] = points[:, 1] * (crop_size / h)
        else:
            # Test: 1024 limit
            limit_size = 1024 
            if max(w, h) > limit_size:
                scale = limit_size / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.BILINEAR)
                if len(points) > 0:
                    points = points * scale
        
        # NPoint Augmentation (Always apply based on alpha intensity)
        if self.phase == 'train' and len(points) > 0 and self.aug_alpha > 0.0:
            curr_w, curr_h = img.size
            points = apply_npoint(points, (curr_h, curr_w), alpha=float(self.aug_alpha))
            points = np.array(points, dtype=np.float32) # Ensure floats are kept until mapped

        k = self.gen_density_map(img.size, points)
        
        if self.transform:
            img = self.transform(img)

        k = torch.from_numpy(k.copy()).float().unsqueeze(0)
        return img, k

    def load_gt_points(self, img_path):
        parent_dir = os.path.dirname(img_path)
        filename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(filename)[0]
        
        gt_path_candidates = []
        
        # 1. ShanghaiTech 스타일 (images 형제 폴더 ground_truth 탐색)
        base_folder = os.path.basename(parent_dir)
        if base_folder.lower() == 'images':
            gt_dir = os.path.join(os.path.dirname(parent_dir), 'ground_truth')
        else:
            gt_dir = parent_dir.replace('images', 'ground_truth')

        if os.path.exists(gt_dir):
            # 우선순위: GT_이름.mat -> GT_이름.txt -> 이름.mat -> 이름.txt
            gt_path_candidates.append(os.path.join(gt_dir, f"GT_{name_no_ext}.mat"))
            gt_path_candidates.append(os.path.join(gt_dir, f"GT_{name_no_ext}.txt"))
            gt_path_candidates.append(os.path.join(gt_dir, f"{name_no_ext}.mat"))
            gt_path_candidates.append(os.path.join(gt_dir, f"{name_no_ext}.txt"))
        
        # 2. 일반 스타일 (이미지와 같은 폴더)
        gt_path_candidates.append(os.path.join(parent_dir, f"{name_no_ext}_ann.mat"))
        gt_path_candidates.append(os.path.join(parent_dir, f"{name_no_ext}.mat"))
        gt_path_candidates.append(os.path.join(parent_dir, f"{name_no_ext}.txt"))
        
        # 3. .npy 배열 (JHU 등을 위한 좌표 파일 탐색 최우선 설정)
        gt_path_candidates.insert(0, os.path.join(parent_dir, f"{name_no_ext}.npy"))

        for gt_path in gt_path_candidates:
            if os.path.exists(gt_path):
                if gt_path.endswith('.npy'):
                    try:
                        pts = np.load(gt_path)
                        if len(pts.shape) == 1:
                            return pts.reshape(-1, 2) if len(pts) > 0 else np.array([])
                        elif len(pts.shape) >= 2:
                            return pts[:, :2]
                    except:
                        continue
                elif gt_path.endswith('.mat'):
                    try:
                        mat = io.loadmat(gt_path)
                        if 'image_info' in mat:
                            return mat['image_info'][0, 0]['location'][0, 0]
                        elif 'annPoints' in mat:
                            return mat['annPoints']
                        else:
                            keys = [k for k in mat.keys() if not k.startswith('_')]
                            return mat[keys[0]] if keys else np.array([])
                    except:
                        continue
                elif gt_path.endswith('.txt'):
                    try:
                        # txt 파일 로드 (구분자 자동 처리 시도)
                        try:
                            pts = np.loadtxt(gt_path, delimiter=',')
                        except:
                            pts = np.loadtxt(gt_path) # 공백 구분
                        
                        if pts.size == 0: return np.array([])
                        if pts.ndim == 1: pts = pts.reshape(1, -1)
                        # 일부 데이터셋은 x,y 외에 다른 정보가 있을 수 있으므로 앞 2개만 사용
                        return pts[:, :2] 
                    except:
                        continue

        return np.array([])

    def gen_density_map(self, img_size, points):
        w, h = img_size
        w_down, h_down = w // self.downsample_ratio, h // self.downsample_ratio
        density_map = np.zeros((h_down, w_down), dtype=np.float32)
        
        if len(points) == 0:
            return density_map

        points_down = points.copy()
        points_down[:, 0] /= self.downsample_ratio
        points_down[:, 1] /= self.downsample_ratio

        for x, y in points_down:
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < w_down and 0 <= iy < h_down:
                self.add_gaussian(density_map, ix, iy, sigma=4)
        return density_map

    def add_gaussian(self, density_map, x, y, sigma):
        k_size = int(sigma * 6)
        x_min, y_min = max(0, x - k_size // 2), max(0, y - k_size // 2)
        x_max, y_max = min(density_map.shape[1], x + k_size // 2 + 1), min(density_map.shape[0], y + k_size // 2 + 1)
        
        if x_max <= x_min or y_max <= y_min: return
        xx, yy = np.meshgrid(np.arange(x_min, x_max) - x, np.arange(y_min, y_max) - y)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel /= (np.sum(kernel) + 1e-9)
        density_map[y_min:y_max, x_min:x_max] += kernel