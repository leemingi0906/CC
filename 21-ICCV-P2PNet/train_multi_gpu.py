import argparse
import datetime
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from crowd_datasets import build_dataset
from engine import train_one_epoch, evaluate_crowd_no_overlap
from models import build_model
import os
from tensorboardX import SummaryWriter
import warnings
import numpy as np
import util.misc as utils
import gc

# [안정화 설정] cuBLAS 연산 에러 방지
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
warnings.filterwarnings('ignore')

# 재현성을 위한 시드 고정 함수
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# DataLoader 워커 시드 고정
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_args_parser():
    parser = argparse.ArgumentParser('P2PNet Training for RTX 3090 Stability', add_help=False)
    
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int) 
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=3500, type=int)
    parser.add_argument('--lr_drop', default=3500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)

    # NPoint 설정
    parser.add_argument('--use_npoint', action='store_true', help='NPoint 증강 활성화 (alpha가 0보다 크면 자동 활성화)')
    parser.add_argument('--alpha', default=0.0, type=float, help='NPoint 노이즈 강도')

    # 모델 아키텍처
    parser.add_argument('--backbone', default='vgg16_bn', type=str)
    parser.add_argument('--row', default=2, type=int)
    parser.add_argument('--line', default=2, type=int)
    parser.add_argument('--set_cost_class', default=1, type=float)
    parser.add_argument('--set_cost_point', default=0.05, type=float)
    parser.add_argument('--point_loss_coef', default=0.0002, type=float)
    parser.add_argument('--eos_coef', default=0.5, type=float)

    # 경로 설정
    parser.add_argument('--data_root', default='/home/kimsooyeon/Downloads/SHT', help='데이터셋 경로')
    parser.add_argument('--dataset_file', default='SHHA', help='데이터셋 이름 (SHHA 또는 SHHB)')
    parser.add_argument('--output_dir', default='', help='자동 생성')
    parser.add_argument('--checkpoints_dir', default='', help='자동 생성')
    parser.add_argument('--tensorboard_dir', default='', help='자동 생성')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='가중치 재시작 경로')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--eval_freq', default=5, type=int)

    return parser

def main(args):
    set_seed(args.seed)

    device = torch.device('cuda')
    model, criterion = build_model(args, training=True)
    model.to(device)
    criterion.to(device)

    if torch.cuda.device_count() > 1:
        print(f"✅ Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    if not os.path.exists(args.data_root):
        print(f"❌ 오류: 데이터 경로를 찾을 수 없습니다: {args.data_root}")
        return

    # [핵심 수정] suffix에 args.dataset_file을 추가하여 폴더 혼선을 방지합니다.
    aug_suffix = f"a{str(args.alpha).replace('.', '_')}" if args.alpha > 0 else "baseline"
    suffix = f"{args.dataset_file}_{aug_suffix}_seed{args.seed}"
    
    exp_path = f"./my_exp/exp-{suffix}"
    if not args.output_dir: args.output_dir = os.path.join(exp_path, f'logs_{suffix}')
    if not args.checkpoints_dir: args.checkpoints_dir = os.path.join(exp_path, f'ckpt_{suffix}')
    if not args.tensorboard_dir: args.tensorboard_dir = os.path.join(exp_path, f'runs_{suffix}')

    for d in [args.output_dir, args.checkpoints_dir]:
        if not os.path.exists(d): os.makedirs(d, exist_ok=True)

    optimizer = torch.optim.Adam([
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad], "lr": args.lr_backbone},
    ], lr=args.lr, weight_decay=args.weight_decay)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # 데이터 로드 (args 전달)
    loading_data = build_dataset(args=args)
    train_set, val_set = loading_data(args.data_root, args)
    
    # NPoint 최종 상태 주입
    train_set.alpha = args.alpha
    if args.alpha > 0:
        train_set.use_npoint = True
        npoint_status = f"활성화 (Alpha: {args.alpha})"
    else:
        train_set.use_npoint = False
        npoint_status = "비활성화 (Baseline)"

    data_loader_train = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers, 
        pin_memory=True, worker_init_fn=seed_worker
    )
    
    data_loader_val = DataLoader(
        val_set, 1, shuffle=False,
        collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers, 
        pin_memory=True
    )

    writer = SummaryWriter(args.tensorboard_dir)
    run_log_name = os.path.join(args.output_dir, 'run_log.txt')
    mae_list = []
    
    print(f"✨ 학습 시작 [데이터셋: {args.dataset_file} | NPoint: {npoint_status} | Seed: {args.seed}]")
    
    for epoch in range(args.epochs):
        try:
            gc.collect()
            torch.cuda.empty_cache()

            t1 = time.time()
            stat = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
            t2 = time.time()

            log_text = f'[Ep {epoch}] LR: {optimizer.param_groups[0]["lr"]:.7f} | Loss: {stat["loss"]:.4f} | {t2-t1:.1f}s'
            print(log_text)
            with open(run_log_name, "a") as f: f.write(log_text + "\n")
            
            writer.add_scalar('loss/total', stat['loss'], epoch)
            lr_scheduler.step()

            if epoch % args.eval_freq == 0 and epoch > 0:
                torch.cuda.synchronize()
                result = evaluate_crowd_no_overlap(model_without_ddp, data_loader_val, device)
                
                mae, mse = result[0], result[1]
                mae_list.append(mae)
                best_mae = np.min(mae_list)
                
                eval_log = f"--- [Eval] Epoch {epoch} | MAE: {mae:.2f} | MSE: {mse:.2f} | Best MAE: {best_mae:.2f}"
                print(eval_log)
                with open(run_log_name, "a") as f: f.write(eval_log + "\n")
                
                writer.add_scalar('metric/mae', mae, epoch)
                writer.add_scalar('metric/mse', mse, epoch)

                if mae <= best_mae:
                    torch.save({'model': model_without_ddp.state_dict(), 'epoch': epoch, 'mae': mae}, 
                               os.path.join(args.checkpoints_dir, 'best_mae.pth'))
                    print(f"🔥 최고 성능 갱신 완료.")

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"⚠️ OOM 발생. 에폭 {epoch}을 건너뜁니다.")
                gc.collect()
                torch.cuda.empty_cache()
                continue
            else: 
                raise e

    writer.close()

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)