import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import argparse
import time
import sys
import gc
import random

# [환경 설정] 실행 위치 및 src 폴더를 시스템 경로에 등록
curr_path = os.getcwd()
src_path = os.path.join(curr_path, 'src')
for path in [curr_path, src_path]:
    if path not in sys.path:
        sys.path.insert(0, path)

# TensorBoard 모듈 임포트 (패키지 미설치 시 에러 방지)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# 프로젝트 내 커스텀 모듈 임포트
try:
    from src.data_loader import MCNN_SHT_Dataset
    from src.models import MCNN
    # 시각화 관련 utils는 더 이상 사용하지 않으므로 임포트 제외
except ImportError as e:
    print(f"❌ 임포트 오류: {e}")
    sys.exit(1)

def set_seed(seed):
    """
    모든 라이브러리의 랜덤 시드를 고정하여 실험 재현성을 보장합니다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ 랜덤 시드가 {seed}로 고정되었습니다.")

def seed_worker(worker_id):
    """
    DataLoader 워커들의 시드를 고정합니다.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_args():
    parser = argparse.ArgumentParser(description='MCNN Training (Fast Mode - No Visualization)')
    
    # 경로 설정
    parser.add_argument('--data_path', default='./data/original/shanghaitech', help='Dataset root path')
    parser.add_argument('--dataset', default='B', choices=['A', 'B'], help='ShanghaiTech Part')
    parser.add_argument('--save_path', default='./checkpoints', help='Model save path')
    parser.add_argument('--log_dir', default='./runs_mcnn', help='TensorBoard log path')
    
    # 하이퍼파라미터
    parser.add_argument('--alpha', default=0.2, type=float, help='NPoint Strength')
    parser.add_argument('--adaptive_npoint', default=0, type=int, help='Adaptive threshold')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--eval_freq', default=5, type=int)
    
    return parser.parse_args()

def train():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    use_npoint = args.alpha > 0
    suffix = f"{args.dataset}_a{str(args.alpha).replace('.', '_')}_s{args.seed}"
    if args.adaptive_npoint > 0: suffix += f"_ad{args.adaptive_npoint}"
    
    current_log_dir = os.path.join(args.log_dir, suffix)
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(current_log_dir, exist_ok=True)

    writer = SummaryWriter(current_log_dir) if TENSORBOARD_AVAILABLE else None

    # 데이터 전처리
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_set = MCNN_SHT_Dataset(args.data_path, part=args.dataset, phase='train', 
                                 transform=transform, use_npoint=use_npoint, 
                                 alpha=args.alpha, adaptive_npoint=args.adaptive_npoint)
    val_set = MCNN_SHT_Dataset(args.data_path, part=args.dataset, phase='test', 
                               transform=transform, use_npoint=False)
    
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, 
        num_workers=4, worker_init_fn=seed_worker, generator=g, pin_memory=True
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    model = MCNN().to(device)
    
    if args.resume and os.path.exists(args.resume):
        print(f"📂 가중치 로드: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)

    if torch.cuda.device_count() > 1:
        print(f"✅ {torch.cuda.device_count()} GPUs 활성화")
        model = nn.DataParallel(model)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    criterion = nn.MSELoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_mae = float('inf')
    print(f"🚀 학습 시작 (Fast Mode - Metrics Only)")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        max_pred = 0.0

        for img, gt in train_loader:
            img, gt = img.to(device), gt.to(device)
            pred = model(img)
            loss = criterion(pred, gt)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            max_pred = max(max_pred, pred.max().item())
        
        avg_loss = epoch_loss / len(train_loader)
        
        if writer:
            writer.add_scalar('Train/Loss', avg_loss, epoch)
            writer.add_scalar('Train/Max_Prediction', max_pred, epoch)

        if epoch % args.eval_freq == 0:
            model.eval()
            mae, mse_sum = 0.0, 0.0
            with torch.no_grad():
                for img, gt in val_loader:
                    img, gt = img.to(device), gt.to(device)
                    pred = model(img)
                    p_cnt, g_cnt = pred.sum().item(), gt.sum().item()
                    mae += abs(p_cnt - g_cnt)
                    mse_sum += (p_cnt - g_cnt)**2
                    # [수정] 시각화 저장 로직(save_results) 제거됨
            
            avg_mae = mae / len(val_set)
            avg_rmse = np.sqrt(mse_sum / len(val_set))
            
            if writer:
                writer.add_scalar('Val/MAE', avg_mae, epoch)
                writer.add_scalar('Val/RMSE', avg_rmse, epoch)

            print(f"Ep {epoch:4d} | Loss: {avg_loss:.6f} | MAE: {avg_mae:6.2f} | RMSE: {avg_rmse:6.2f}", end="")
            
            if avg_mae < best_mae:
                best_mae = avg_mae
                save_name = os.path.join(args.save_path, f'mcnn_{args.dataset}_best.pth')
                torch.save(model_without_ddp.state_dict(), save_name)
                print(" ⭐ Best Updated!")
            else:
                print("")

        gc.collect()
        torch.cuda.empty_cache()

    if writer: writer.close()
    print(f"✅ 학습 완료. 최종 Best MAE: {best_mae:.2f}")

if __name__ == '__main__':
    train()