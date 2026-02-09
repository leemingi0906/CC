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
import gc

# [환경 설정] 현재 경로를 시스템 경로에 추가 (Import 에러 방지)
curr_path = os.getcwd()
sys.path.append(curr_path)

# TensorBoard 설정
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# 프로젝트 모듈 임포트
# model.py와 dataset.py가 같은 폴더에 있어야 합니다.
try:
    from model import CSRNet
    from dataset import CSRNet_Dataset
except ImportError as e:
    print(f"❌ 임포트 오류: {e}")
    print("가이드: model.py와 dataset.py가 현재 폴더에 있는지 확인하세요.")
    sys.exit(1)

# 경고 무시
warnings.filterwarnings('ignore')

def set_seed(seed):
    """실험 재현성을 위한 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser(description='CSRNet Training with NPoint')
    
    # 1. 데이터 및 경로
    parser.add_argument('--data_root', default='/home/mingi/Downloads/SHT', help='데이터셋 루트 경로')
    parser.add_argument('--dataset', default='B', choices=['A', 'B'], help='ShanghaiTech Part')
    parser.add_argument('--save_path', default='./checkpoints', help='모델 저장 경로')
    parser.add_argument('--log_dir', default='./runs', help='텐서보드 로그 경로')
    
    # 2. NPoint 하이퍼파라미터
    parser.add_argument('--alpha', default=0.0, type=float, help='NPoint 노이즈 강도 (0.0이면 비활성)')
    parser.add_argument('--adaptive_npoint', default=7, type=int, help='적응형 임계값 (0이면 모든 이미지 적용)')
    
    # 3. 학습 설정
    parser.add_argument('--lr', default=1e-5, type=float, help='학습률 (CSRNet은 1e-5 ~ 1e-6 추천)')
    parser.add_argument('--batch_size', default=1, type=int, help='배치 사이즈 (이미지 크기가 다양하므로 1 권장)')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--eval_freq', default=5, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', type=str, help='재시작할 체크포인트 경로')
    parser.add_argument('--gpu_id', default='0', type=str, help='사용할 GPU ID (예: 0 또는 0,1,2,3)')

    return parser.parse_args()

def train():
    args = get_args()
    
    # GPU 설정
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    # 폴더 이름 자동 생성 (실험 관리용)
    # 예: CSRNet_B_a0_2_ad7_s42
    alpha_str = str(args.alpha).replace('.', '_')
    ad_str = f"ad{args.adaptive_npoint}" if args.adaptive_npoint > 0 else "all"
    experiment_name = f"CSRNet_{args.dataset}_a{alpha_str}_{ad_str}_s{args.seed}"
    
    ckpt_dir = os.path.join(args.save_path, experiment_name)
    log_dir = os.path.join(args.log_dir, experiment_name)
    
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir) if TENSORBOARD_FOUND else None
    print(f"🚀 실험 시작: {experiment_name}")
    print(f"📂 저장 경로: {ckpt_dir}")

    # 데이터 로더
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # NPoint 사용 여부 판단
    use_npoint = args.alpha > 0
    
    print(f"📊 데이터 로딩 중... (NPoint: {use_npoint})")
    train_set = CSRNet_Dataset(args.data_root, part=args.dataset, phase='train', 
                               transform=transform, use_npoint=use_npoint, 
                               alpha=args.alpha, adaptive_npoint=args.adaptive_npoint)
    val_set = CSRNet_Dataset(args.data_root, part=args.dataset, phase='test', 
                             transform=transform, use_npoint=False)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # 모델 초기화
    model = CSRNet().to(device)
    
    # 멀티 GPU (DataParallel)
    if torch.cuda.device_count() > 1:
        print(f"✅ {torch.cuda.device_count()} GPUs DataParallel Activated.")
        model = nn.DataParallel(model)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # 가중치 로드 (Resume)
    if args.resume and os.path.exists(args.resume):
        print(f"🔄 체크포인트 로드: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)

    # 손실 함수 및 옵티마이저
    # CSRNet은 MSELoss(sum reduction)를 사용합니다.
    criterion = nn.MSELoss(reduction='sum').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 스케줄러 (선택 사항, 여기서는 단순하게 유지)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    best_mae = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        for i, (img, gt) in enumerate(train_loader):
            img = img.to(device)
            gt = gt.to(device)
            
            output = model(img)
            loss = criterion(output, gt)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # scheduler.step() # 스케줄러 사용 시 활성화
        avg_loss = epoch_loss / len(train_loader)
        
        if writer: writer.add_scalar('Train/Loss', avg_loss, epoch)

        # 평가 (Validation)
        if epoch % args.eval_freq == 0:
            model.eval()
            mae = 0.0
            mse = 0.0
            
            with torch.no_grad():
                for img, gt in val_loader:
                    img = img.to(device)
                    gt = gt.to(device)
                    output = model(img)
                    
                    p_cnt = output.sum().item()
                    g_cnt = gt.sum().item()
                    
                    mae += abs(p_cnt - g_cnt)
                    mse += (p_cnt - g_cnt) ** 2
            
            avg_mae = mae / len(val_set)
            avg_mse = np.sqrt(mse / len(val_set)) # RMSE
            
            if writer:
                writer.add_scalar('Val/MAE', avg_mae, epoch)
                writer.add_scalar('Val/RMSE', avg_mse, epoch)
            
            print(f"[Ep {epoch}/{args.epochs}] Loss: {avg_loss:.4f} | MAE: {avg_mae:.2f} | RMSE: {avg_mse:.2f}")
            
            # Best Model 저장
            if avg_mae < best_mae:
                best_mae = avg_mae
                save_name = os.path.join(ckpt_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'state_dict': model_without_ddp.state_dict(),
                    'best_mae': best_mae,
                    'optimizer': optimizer.state_dict(),
                    'args': vars(args)
                }, save_name)
                print(f" ⭐ Best Saved! ({best_mae:.2f})")
        
        # 메모리 관리
        gc.collect()
        torch.cuda.empty_cache()

    if writer: writer.close()
    print(f"✅ 학습 완료. 최종 Best MAE: {best_mae:.2f}")

if __name__ == '__main__':
    train()