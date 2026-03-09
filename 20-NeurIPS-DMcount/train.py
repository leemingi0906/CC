import sys
import os
import warnings
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from glob import glob

# DM-Count 관련 모듈
from models import vgg19
from losses.ot_loss import OT_Loss
from datasets.crowd import Crowd

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    print("⚠️ TensorBoard not found.")

warnings.filterwarnings('ignore')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id):
    """
    DataLoader의 각 워커가 고유한 난수 시드를 갖도록 설정합니다.
    NumPy 기반의 Augmentation(NPoint 등)을 사용할 때 필수적입니다.
    """
    # PyTorch가 각 워커에게 부여한 시드를 가져와서 NumPy 시드로 설정
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_args():
    parser = argparse.ArgumentParser(description='DM-Count Unified Training')
    parser.add_argument('--data_root', default='./', help='데이터셋 최상위 폴더')
    parser.add_argument('--dataset', default='qnrf', choices=['qnrf', 'sha', 'shb', 'cc50', 'jhu'], help='데이터셋 선택')
    parser.add_argument('--save_path', default='./checkpoints', help='모델 저장 경로')
    parser.add_argument('--log_dir', default='./runs', help='텐서보드 로그 경로')
    
    # 모델 및 학습
    parser.add_argument('--crop_size', default=256, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu_id', default='0', type=str)
    
    # UCF-CC-50 전용 인자
    parser.add_argument('--test_fold', default=0, type=int, choices=[0, 1, 2, 3, 4], 
                        help='UCF-CC-50 테스트용 Fold 번호 (0~4)')

    # 손실함수 및 증강
    parser.add_argument('--wot', type=float, default=0.1, help='OT Loss Weight')
    parser.add_argument('--wtv', type=float, default=0.01, help='TV Loss Weight')
    parser.add_argument('--reg', type=float, default=10.0, help='Entropy Regularization')
    parser.add_argument('--alpha', default=0.0, type=float, help='NPoint Noise Alpha')
    parser.add_argument('--adaptive_npoint', default=7, type=int)
    return parser.parse_args()

def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]
    st_sizes = torch.tensor(transposed_batch[2])
    return images, points, st_sizes

def val_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]
    img_paths = transposed_batch[2]
    return images, points, img_paths

def compute_tv_loss(x):
    h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return h_tv + w_tv

def train():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    # -----------------------------------------------------------
    # [1] 데이터셋 준비 (CC50 5-Fold 분할 포함)
    # -----------------------------------------------------------
    train_file_list = None
    val_file_list = None
    
    if args.dataset == 'cc50':
        # 1. 모든 이미지 검색
        all_images = sorted(glob(os.path.join(args.data_root, '*.jpg')))
        
        if len(all_images) == 0:
            print(f"❌ [Error] UCF-CC-50 이미지를 찾을 수 없습니다: {args.data_root}")
            return
            
        print(f"📊 UCF-CC-50 Total Images: {len(all_images)}")
        
        # 2. 5-Fold Split
        # 시드에 상관없이 항상 동일한 순서로 섞어서 분할 (재현성 보장)
        split_rng = random.Random(42) 
        indices = list(range(len(all_images)))
        split_rng.shuffle(indices)
        
        fold_size = len(all_images) // 5
        start_idx = args.test_fold * fold_size
        end_idx = start_idx + fold_size
        
        val_indices = indices[start_idx:end_idx]
        train_indices = [idx for idx in indices if idx not in val_indices]
        
        train_file_list = [all_images[i] for i in train_indices]
        val_file_list = [all_images[i] for i in val_indices]
        
        # Crowd 클래스에 경로는 Dummy로 넘기고 file_list를 사용
        train_path = args.data_root 
        val_path = args.data_root
        print(f"🔄 5-Fold CV [Fold {args.test_fold}]: Train({len(train_file_list)}) / Val({len(val_file_list)})")

    # 기존 데이터셋 처리
    elif args.dataset == 'qnrf':
        train_path = os.path.join(args.data_root, 'Train')
        val_path = os.path.join(args.data_root, 'Test')
        if not os.path.exists(train_path): train_path = os.path.join(args.data_root, 'train')
        if not os.path.exists(val_path): val_path = os.path.join(args.data_root, 'test')
    elif args.dataset == 'sha':
        train_path = os.path.join(args.data_root, 'part_A_final', 'train_data')
        val_path = os.path.join(args.data_root, 'part_A_final', 'test_data')
    elif args.dataset == 'shb':
        train_path = os.path.join(args.data_root, 'part_B_final', 'train_data')
        val_path = os.path.join(args.data_root, 'part_B_final', 'test_data')
    elif args.dataset == 'jhu':
        train_path = os.path.join(args.data_root, 'train')
        val_path = os.path.join(args.data_root, 'val')
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # -----------------------------------------------------------
    # [2] 실험 이름 설정
    # -----------------------------------------------------------
    alpha_str = str(args.alpha).replace('.', '_')
    if args.dataset == 'cc50':
        experiment_name = f"DM_{args.dataset}_fold{args.test_fold}_a{alpha_str}"
    else:
        experiment_name = f"DM_{args.dataset}_a{alpha_str}_s{args.seed}"
        
    ckpt_dir = os.path.join(args.save_path, experiment_name)
    log_dir = os.path.join(args.log_dir, experiment_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir) if TENSORBOARD_FOUND else None

    print(f"🚀 실험 시작: {experiment_name}")
    if args.dataset != 'cc50':
        print(f"📂 Train Path: {train_path}")

    # -----------------------------------------------------------
    # [3] 데이터 로더 초기화
    # -----------------------------------------------------------
    train_set = Crowd(
        train_path, 
        crop_size=args.crop_size, 
        method='train', 
        dataset_name=args.dataset, 
        alpha=args.alpha, 
        adaptive_npoint=args.adaptive_npoint,
        file_list=train_file_list  # [New] CC50일 때만 리스트 전달
    )
    
    if len(train_set) == 0:
        print("❌ 학습 데이터가 0개입니다. 경로를 확인하세요.")
        return

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, 
                              num_workers=4, collate_fn=train_collate,
                              worker_init_fn=worker_init_fn) # [New] worker_init_fn 추가
    
    val_set = Crowd(
        val_path, 
        method='val', 
        dataset_name=args.dataset,
        file_list=val_file_list    # [New] CC50일 때만 리스트 전달
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, 
                            num_workers=2, collate_fn=val_collate,
                            worker_init_fn=worker_init_fn) # [New] worker_init_fn 추가

    # -----------------------------------------------------------
    # [4] 모델 및 학습 루프
    # -----------------------------------------------------------
    model = vgg19().to(device)
    ot_criterion = OT_Loss(args.crop_size, 8, 0, device, 100, args.reg).to(device)
    count_criterion = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_mae = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_epoch_loss = 0.0
        
        for imgs, points, st_sizes in train_loader:
            imgs = imgs.to(device)
            points = [p.to(device) for p in points]
            gd_counts = torch.tensor([len(p) for p in points], dtype=torch.float32).to(device)
            
            mu, mu_normed = model(imgs)
            
            ot_loss, _, _ = ot_criterion(mu_normed, mu, points)
            ot_loss = ot_loss * args.wot
            
            pred_counts = mu.sum(dim=(1, 2, 3))
            count_loss = count_criterion(pred_counts, gd_counts)
            tv_loss = compute_tv_loss(mu) * args.wtv
            
            loss = ot_loss + count_loss + tv_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_epoch_loss += loss.item()
        
        if writer:
            writer.add_scalar('Train/Loss', total_epoch_loss / len(train_loader), epoch)

        # Validation (CC50은 데이터가 적으므로 매 에폭마다 확인 권장)
        eval_freq = 1 if args.dataset == 'cc50' else 5
        
        if (epoch + 1) % eval_freq == 0:
            model.eval()
            mae = 0.0
            with torch.no_grad():
                for imgs, points, paths in val_loader:
                    imgs = imgs.to(device)
                    mu, _ = model(imgs)
                    pred_cnt = mu.sum().item()
                    gt_cnt = len(points[0])
                    mae += abs(pred_cnt - gt_cnt)
            
            avg_mae = mae / len(val_set)
            print(f"[Ep {epoch+1}] Loss: {total_epoch_loss/len(train_loader):.4f} | MAE: {avg_mae:.2f}")
            
            if writer: writer.add_scalar('Val/MAE', avg_mae, epoch)

            if avg_mae < best_mae:
                best_mae = avg_mae
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_model.pth'))
                print(f" ⭐ Best Saved! (MAE: {best_mae:.2f})")

    if writer: writer.close()
    print(f"✅ Training Complete. Best MAE: {best_mae:.2f}")

if __name__ == '__main__':
    train()