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

# [환경 설정]
curr_path = os.getcwd()
sys.path.append(curr_path)

# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# 프로젝트 모듈
try:
    from model import CSRNet
    from dataset import CSRNet_Dataset
except ImportError as e:
    print(f"❌ 임포트 오류: {e}")
    sys.exit(1)

warnings.filterwarnings('ignore')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # [CUBLAS BUG FIX] Force bypass of CuDNN to prevent CUBLAS_STATUS_EXECUTION_FAILED on fragmented/shared GPUs
    torch.backends.cudnn.enabled = False

def get_args():
    parser = argparse.ArgumentParser(description='CSRNet Unified Training')
    
    # 1. 데이터셋 설정
    # SHT와 QCF가 현재 스크립트와 같은 폴더(18-CVPR-CSRNet) 내에 있으므로 기본값을 './'로 설정
    parser.add_argument('--data_root', default='./', help='데이터셋 최상위 폴더 (SHT, QCF가 있는 곳)')
    parser.add_argument('--dataset', dest='dataset_name', default='SHT', help='사용할 데이터셋 (SHT, QNRF, CC50)')
    parser.add_argument('--part', default='B', type=str, help='SHT Part (A or B)')
    parser.add_argument('--test_fold', default=0, type=int, help='CC50 Cross Validation Fold')
    
    # 2. 저장 경로
    parser.add_argument('--save_path', default='./checkpoints', help='모델 저장 경로')
    parser.add_argument('--log_dir', default='./runs', help='텐서보드 로그 경로')
    
    # 3. NPoint & Augmentation
    parser.add_argument('--alpha', default=0.0, type=float, help='NPoint 노이즈 강도 (0.0=Off)')
    parser.add_argument('--adaptive_npoint', default=7, type=int)
    parser.add_argument('--crop_size', default=400, type=int, help='Train Crop Size (QNRF는 이미지가 크므로 주의)')
    
    # 4. 학습 파라미터
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--eval_freq', default=5, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--gpu_id', default='0', type=str)

    return parser.parse_args()

def train():
    args = get_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    # 대소문자 무시 (유저가 소문자로 입력해도 동작하도록)
    args.dataset_name = args.dataset_name.upper()
    args.part = args.part.upper()

    # 실험 이름 생성
    # 예: SHT_B_a0.0_s42 또는 QNRF_a0.5_s42
    if args.dataset_name in ['SHT', 'SHA', 'SHB']:
        if args.dataset_name == 'SHA' or args.part == 'A':
            args.dataset_name = 'SHT'
            args.part = 'A'
        elif args.dataset_name == 'SHB' or args.part == 'B':
            args.dataset_name = 'SHT'
            args.part = 'B'
        data_tag = f"{args.dataset_name}_{args.part}"
    elif args.dataset_name == 'JHU':
        data_tag = 'JHU'
    else:
        data_tag = args.dataset_name
        
    alpha_str = str(args.alpha).replace('.', '_')
    experiment_name = f"CSRNet_{data_tag}_a{alpha_str}_s{args.seed}"
    
    ckpt_dir = os.path.join(args.save_path, experiment_name)
    log_dir = os.path.join(args.log_dir, experiment_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir) if TENSORBOARD_FOUND else None
    
    print(f"🚀 실험 시작: {experiment_name}")
    print(f"📂 데이터: {args.dataset_name}, 루트: {args.data_root}")

    # 데이터 로더 설정
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    use_npoint = args.alpha > 0
    
    # Train Loader
    train_set = CSRNet_Dataset(
        data_root=args.data_root, 
        dataset_name=args.dataset_name,
        part=args.part, 
        phase='train', 
        transform=transform, 
        use_npoint=use_npoint, 
        alpha=args.alpha, 
        adaptive_npoint=args.adaptive_npoint,
        crop_size=args.crop_size,
        test_fold=args.test_fold
    )
    
    # Val Loader
    val_set = CSRNet_Dataset(
        data_root=args.data_root, 
        dataset_name=args.dataset_name,
        part=args.part, 
        phase='test', 
        transform=transform, 
        use_npoint=False,
        test_fold=args.test_fold
    )
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False) # Test는 원본 크기이므로 배치 1 고정

    # 모델 초기화
    model = CSRNet().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    criterion = nn.MSELoss(reduction='sum').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Resume
    if args.resume and os.path.exists(args.resume):
        print(f"🔄 Resume: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])

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
        
        avg_loss = epoch_loss / len(train_loader)
        if writer: writer.add_scalar('Train/Loss', avg_loss, epoch)

        # Validation
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
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
            avg_mse = np.sqrt(mse / len(val_set))
            
            print(f"[Ep {epoch}] Loss: {avg_loss:.4f} | MAE: {avg_mae:.2f} | MSE: {avg_mse:.2f}")
            
            if writer:
                writer.add_scalar('Val/MAE', avg_mae, epoch)
                writer.add_scalar('Val/MSE', avg_mse, epoch)

            if avg_mae < best_mae:
                best_mae = avg_mae
                save_name = os.path.join(ckpt_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'state_dict': model_without_ddp.state_dict(),
                    'best_mae': best_mae,
                }, save_name)
                print(f" ⭐ Best Saved! ({best_mae:.2f})")
        else:
            print(f"[Ep {epoch}] Loss: {avg_loss:.4f} (Skipping Eval)")

    if writer: writer.close()
    print(f"Training Complete. Best MAE: {best_mae:.2f}")

if __name__ == '__main__':
    train()