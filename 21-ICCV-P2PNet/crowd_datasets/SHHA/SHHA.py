 import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob
import scipy.io as io

# NPoint 모듈 임포트
try:
    from npoint_aug import apply_npoint
except ImportError:
    import sys
    sys.path.append("..")
    from npoint_aug import apply_npoint


class SHHA(Dataset):
    def __init__(
        self,
        data_root,
        transform=None,
        train=False,
        patch=False,
        flip=False,
        use_npoint=False
    ):
        self.root_path = data_root
        self.train_lists = "shanghai_tech_part_a_train.list"
        self.eval_list = "shanghai_tech_part_a_test.list"

        self.train = train
        self.patch = patch
        self.flip = flip
        self.transform = transform

        # NPoint 관련 설정
        self.use_npoint = use_npoint
        self.alpha = 0.0   # 🔥 train.py에서 args.alpha로 덮어씀

        if train:
            self.img_list_file = self.train_lists.split(',')
            sub_path = 'part_A_final/train_data'
        else:
            self.img_list_file = self.eval_list.split(',')
            sub_path = 'part_A_final/test_data'

        self.img_map = {}
        self.img_list = []

        for _, list_name in enumerate(self.img_list_file):
            list_path = os.path.join(self.root_path, list_name.strip())
            if not os.path.exists(list_path):
                print(f"⚠️ 경고: 리스트 파일을 찾을 수 없습니다: {list_path}")
                continue

            with open(list_path) as fin:
                for line in fin:
                    if len(line) < 2:
                        continue
                    line = line.strip().split()

                    raw_img_path = line[0].strip()
                    raw_gt_path = line[1].strip()

                    img_name = os.path.basename(raw_img_path)
                    gt_name = os.path.basename(raw_gt_path)

                    real_img_path = os.path.join(
                        self.root_path, sub_path, 'images', img_name
                    )
                    real_gt_path = os.path.join(
                        self.root_path, sub_path, 'ground_truth', gt_name
                    )

                    self.img_map[real_img_path] = real_gt_path

        self.img_list = sorted(list(self.img_map.keys()))
        self.nSamples = len(self.img_list)

        if self.nSamples == 0:
            print(
                f"❌ 오류: '{self.root_path}' 내에서 이미지를 찾지 못했습니다."
            )

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        img_path = self.img_list[index]
        gt_path = self.img_map[img_path]

        # 데이터 로드
        img, point = load_data((img_path, gt_path), self.train)

        # 🔥 NPoint 적용 (alpha 기반, 하드코딩 제거)
        if self.train and self.use_npoint and self.alpha > 0:
            w, h = img.size
            point = apply_npoint(
                point,
                (h, w),
                alpha=self.alpha,
                k=4
            )

        # 이미지 변환
        if self.transform is not None:
            img = self.transform(img)

        # P2PNet 기본 증강 (scale / crop / flip)
        if self.train:
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)

            if scale * min_size > 128:
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0),
                    scale_factor=scale,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                point *= scale

            if self.patch:
                img, point = random_crop(img, point)

            if self.flip and random.random() > 0.5:
                img = torch.flip(img, dims=[-1])
                if isinstance(point, list):
                    for p in point:
                        p[:, 0] = img.shape[-1] - p[:, 0]
                else:
                    point[:, 0] = img.shape[-1] - point[:, 0]

        if not self.train:
            point = [point]

        # Target 생성
        target = []
        for i in range(len(point)):
            d = {}
            p = torch.tensor(point[i], dtype=torch.float32)
            d['point'] = p
            d['labels'] = torch.ones([p.shape[0]], dtype=torch.int64)
            target.append(d)

        return img, target


def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"❌ 이미지를 읽을 수 없습니다: {img_path}")

    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    points = []
    with open(gt_path) as f_label:
        for line in f_label:
            line = line.strip().split()
            if not line:
                continue
            points.append([float(line[0]), float(line[1])])

    return img, np.array(points)


def random_crop(img, den, num_patch=4):
    half_h, half_w = 128, 128
    result_img = torch.zeros([num_patch, img.shape[0], half_h, half_w])
    result_den = []

    c, h, w = img.shape
    for i in range(num_patch):
        start_h = random.randint(0, max(0, h - half_h))
        start_w = random.randint(0, max(0, w - half_w))

        result_img[i] = img[:, start_h:start_h + half_h, start_w:start_w + half_w]

        idx = (
            (den[:, 0] >= start_w) &
            (den[:, 0] <= start_w + half_w) &
            (den[:, 1] >= start_h) &
            (den[:, 1] <= start_h + half_h)
        )

        record_den = den[idx].copy()
        record_den[:, 0] -= start_w
        record_den[:, 1] -= start_h
        result_den.append(record_den)

    return result_img, result_den
