import os
import sys
import argparse
import random
import numpy as np

# [중요] CUDA_VISIBLE_DEVICES 설정은 torch 임포트 전에 하는 것이 가장 안전합니다.
def pre_config():
    parser = argparse.ArgumentParser(description='MCNN Training')
    
    # 기본 인자
    parser.add_argument('--data_root', default='./data', help='데이터셋 루트 경로')
    parser.add_argument('--dataset', default='sha', help='데이터셋 이름 (sha, shb, qnrf, cc50, jhu)')
    parser.add_argument('--save_path', default='./checkpoints', help='모델 저장 경로')
    parser.add_argument('--log_dir', default='./runs', help='로그 경로')
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    
    # 실험용 추가 인자 (쉘 스크립트 대응)
    parser.add_argument('--alpha', default=1.0, type=float, help='NPoint Augmentation 강도')
    parser.add_argument('--gpu_id', default=0, type=str, help='사용할 GPU 번호 (예: 0 또는 1)')
    parser.add_argument('--test_fold', default=0, type=int, help='CC50용 폴더 번호 (하위 호환성용)')
    
    args = parser.parse_args()
    
    # GPU 설정 적용
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    return args

# 인자를 먼저 파싱하고 GPU를 설정한 뒤 torch를 불러옵니다.
args = pre_config()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# 프로젝트 경로 설정
curr_path = os.getcwd()
if os.path.join(curr_path, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(curr_path, 'src'))

try:
    from src.data_loader import CrowdDataset
    from src.models import MCNN
except ImportError:
    from data_loader import CrowdDataset
    from models import MCNN

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train():
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 실험 이름 설정
    exp_name = f"{args.dataset}_alpha{args.alpha}_s{args.seed}_gpu{args.gpu_id}"
    writer = SummaryWriter(os.path.join(args.log_dir, exp_name))
    os.makedirs(args.save_path, exist_ok=True)

    print(f"🚀 실험 시작: {exp_name}")
    print(f"📍 Device: {device} (VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')})")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset 로드 (data_loader.py의 수정된 경로 로직 사용)
    train_set = CrowdDataset(args.data_root, args.dataset, 'train', transform, aug_alpha=args.alpha)
    val_set = CrowdDataset(args.data_root, args.dataset, 'test', transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    model = MCNN().to(device)
    criterion = nn.MSELoss(reduction='sum').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_mae = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for img, gt in train_loader:
            img, gt = img.to(device), gt.to(device)
            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, gt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Loss/Train', avg_loss, epoch)

        if epoch % 5 == 0:
            model.eval()
            mae = 0.0
            with torch.no_grad():
                for img, gt in val_loader:
                    img, gt = img.to(device), gt.to(device)
                    pred = model(img)
                    mae += abs(pred.sum().item() - gt.sum().item())
            
            avg_mae = mae / len(val_set)
            writer.add_scalar('Metric/MAE', avg_mae, epoch)
            print(f"Ep {epoch} | Loss: {avg_loss:.4f} | MAE: {avg_mae:.2f}")

            if avg_mae < best_mae:
                best_mae = avg_mae
                torch.save(model.state_dict(), os.path.join(args.save_path, f"{exp_name}_best.pth"))
                print(f"  ⭐ Best Model Saved")

    writer.close()

if __name__ == '__main__':
    train()