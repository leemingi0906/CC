import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import vgg19
from losses.ot_loss import OT_Loss
import random
import numpy as np
import time

from datasets.crowd import Crowd

def get_args():
    parser = argparse.ArgumentParser(description='DM-Count Final Training')
    parser.add_argument('--data_root', default='./SHT', help='상하이텍 데이터셋 경로')
    parser.add_argument('--dataset', default='B', choices=['A', 'B'])
    
    # NPoint 설정
    parser.add_argument('--alpha', default=0.0, type=float)
    parser.add_argument('--adaptive_npoint', default=7, type=int)
    
    # 하이퍼파라미터
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--save_path', default='./checkpoints')
    parser.add_argument('--crop_size', default=256, type=int)
    
    # DM-Count 손실함수 가중치
    parser.add_argument('--wot', type=float, default=0.1, help='OT Loss 가중치')
    parser.add_argument('--wtv', type=float, default=0.01, help='TV Loss 가중치')
    parser.add_argument('--reg', type=float, default=10.0, help='Sinkhorn entropy regularization')
    
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu_id', default='0', type=str)
    
    return parser.parse_args()

def train_collate(batch):
    """훈련용 collate: (img, points, st_size) 구조를 처리"""
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1] # 가변 길이 리스트
    st_sizes = torch.tensor(transposed_batch[2]) # 수치 데이터이므로 텐서화 가능
    return images, points, st_sizes

def val_collate(batch):
    """검증용 collate: (img, points, img_path) 구조를 처리"""
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]
    img_paths = transposed_batch[2] # 문자열 리스트
    return images, points, img_paths

def compute_tv_loss(x):
    """Density Map의 매끄러움을 위한 Total Variation Loss"""
    h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return h_tv + w_tv

def train():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    suffix = f"DM_{args.dataset}_a{str(args.alpha).replace('.', '_')}"
    ckpt_dir = os.path.join(args.save_path, suffix)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # 데이터 로더 설정
    train_set = Crowd(os.path.join(args.data_root, f'part_{args.dataset}_final', 'train_data'), 
                      crop_size=args.crop_size, method='train', alpha=args.alpha, adaptive_npoint=args.adaptive_npoint)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, 
                              num_workers=4, collate_fn=train_collate)
    
    val_set = Crowd(os.path.join(args.data_root, f'part_{args.dataset}_final', 'test_data'), method='val')
    # [수정] 검증용 collate를 따로 사용하여 타입 에러 방지
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=val_collate)

    model = vgg19().to(device)
    
    # [참고] 만약 OT_Loss.forward가 st_sizes를 받도록 수정되었다면 아래 호출 시 인자를 추가해야 합니다.
    ot_criterion = OT_Loss(args.crop_size, 8, 0, device, 100, args.reg).to(device)
    count_criterion = nn.L1Loss(reduction='mean').to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_mae = float('inf')

    print(f"🚀 학습 시작: OT(w={args.wot}) + TV(w={args.wtv}) + Count(w=1.0)")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        for imgs, points, st_sizes in train_loader:
            imgs = imgs.to(device)
            points = [p.to(device) for p in points]
            gd_counts = torch.tensor([len(p) for p in points], dtype=torch.float32).to(device)
            # st_sizes = st_sizes.to(device) # 필요 시 사용
            
            mu, mu_normed = model(imgs)
            
            # 1. OT Loss (st_sizes가 OT_Loss 내부에서 좌표 스케일링에 쓰일 수 있음)
            # 현재 OT_Loss forward 시그니처에 따라 호출 (필요 시 st_sizes 추가)
            ot_loss, _, _ = ot_criterion(mu_normed, mu, points)
            ot_loss = ot_loss * args.wot
            
            # 2. Counting Loss
            pred_counts = mu.sum(dim=(1, 2, 3))
            count_loss = count_criterion(pred_counts, gd_counts)
            
            # 3. TV Loss
            tv_loss = compute_tv_loss(mu) * args.wtv
            
            total_loss = ot_loss + count_loss + tv_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()

        # 5에폭마다 평가
        if epoch % 5 == 0:
            model.eval()
            mae = 0.0
            with torch.no_grad():
                for imgs, points, paths in val_loader: # 세 번째 인자가 paths임이 명확해짐
                    imgs = imgs.to(device)
                    mu, _ = model(imgs)
                    
                    pred_cnt = mu.sum().item()
                    gt_cnt = len(points[0])
                    mae += abs(pred_cnt - gt_cnt)
            
            avg_mae = mae / len(val_set)
            print(f"[Ep {epoch}] Total Loss: {epoch_loss/len(train_loader):.6f} | MAE: {avg_mae:.2f}")

            if avg_mae < best_mae:
                best_mae = avg_mae
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_model.pth'))
                print(" ⭐ Best Saved!")

    print(f"✅ Final Best MAE: {best_mae:.2f}")

if __name__ == '__main__':
    train()