import argparse
import datetime
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# from crowd_datasets import build_dataset
from crowd_datasets import *
from engine import train_one_epoch, evaluate_crowd_no_overlap
from models import build_model
import os
import sys # 경로 조작을 위해 필수
from tensorboardX import SummaryWriter
import warnings
import numpy as np
import util.misc as utils
import gc

# [안정화 설정] cuBLAS 연산 에러 방지
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
warnings.filterwarnings('ignore')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_args_parser():
    parser = argparse.ArgumentParser('P2PNet Training', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int) 
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=3500, type=int)
    parser.add_argument('--lr_drop', default=3500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)
    # parser.add_argument('--use_npoint', action='store_true', help='NPoint 활성화')
    parser.add_argument('--alpha', default=0.0, type=float, help='노이즈 강도')
    parser.add_argument('--backbone', default='vgg16_bn', type=str)
    parser.add_argument('--row', default=2, type=int)
    parser.add_argument('--line', default=2, type=int)
    parser.add_argument('--set_cost_class', default=1, type=float)
    parser.add_argument('--set_cost_point', default=0.05, type=float)
    parser.add_argument('--point_loss_coef', default=0.0002, type=float)
    parser.add_argument('--eos_coef', default=0.5, type=float)
    parser.add_argument('--data_root', default='/home/kimsooyeon/Downloads/SHT', help='데이터셋 경로')
    parser.add_argument('--dataset_file', default='SHHA', help='데이터셋 이름 (SHHA/SHHB)')
    parser.add_argument('--output_dir', default='', help='자동 생성')
    parser.add_argument('--checkpoints_dir', default='', help='자동 생성')
    parser.add_argument('--tensorboard_dir', default='', help='자동 생성')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='가중치 재시작 경로')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--eval_freq', default=5, type=int)
    parser.add_argument('--adaptive_npoint', default=0, type=int, help='적응형 NPoint 활성화')
    return parser

def main(args):
    set_seed(args.seed)

    # ---------------------------------------------------------
    # [Fallback 로직 1단계] 현재 실행 경로를 최우선으로 등록
    # ---------------------------------------------------------
    curr_path = os.getcwd()
    if curr_path not in sys.path:
        sys.path.insert(0, curr_path) # 리스트 맨 앞에 추가하여 우선순위 확보

    device = torch.device('cuda')
    model, criterion = build_model(args, training=True)
    model.to(device)
    criterion.to(device)

    if torch.cuda.device_count() > 1:
        print(f"✅ {torch.cuda.device_count()} GPUs detected. DataParallel 활성화.")
        model = torch.nn.DataParallel(model)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    if not os.path.exists(args.data_root):
        print(f"❌ 데이터 경로 오류: {args.data_root}")
        return

    # 경로 자동 생성
    aug_tag = f"a{str(args.alpha).replace('.', '_')}" if args.alpha > 0 else "baseline"
    suffix = f"{args.dataset_file}_{aug_tag}_seed{args.seed}"
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

    # ---------------------------------------------------------
    # [Fallback 로직 2단계] 데이터셋 로딩 경로 탐색 순서 최적화
    # ---------------------------------------------------------
    print(f"📊 데이터셋 로딩 시도: {args.dataset_file}...")
    loader_found = False
    try:
        # 우선순위 1: 사용자가 직접 정의한 crowd_datasets/loading_data.py를 먼저 시도
        # SHHB 등 확장된 데이터셋 처리가 포함되어 있음
        from crowd_datasets.loading_data import loading_data as data_loader_fn
        train_set, val_set = data_loader_fn(args.data_root, args)
        loader_found = True
        print(f"✅ 커스텀 로더(loading_data.py)를 통해 {args.dataset_file}를 로드했습니다.")
    except (ImportError, TypeError) as e:
        # 우선순위 2: 실패 시 P2PNet 기본 factory 함수인 build_dataset 사용
        print(f"⚠️ 커스텀 로더 실패 ({e}). 기본 build_dataset으로 시도합니다.")
        loading_data_factory = build_dataset(args=args)
        if loading_data_factory is not None:
            train_set, val_set = loading_data_factory(args.data_root)
            loader_found = True
            print(f"✅ 기본 build_dataset을 통해 로드했습니다.")

    if not loader_found:
        print("❌ 최종 로딩 실패: 폴더 구조를 확인하세요.")
        print("가이드: crowd_datasets/ 폴더 안에 loading_data.py가 있어야 합니다.")
        return
    
    # NPoint 최종 파라미터 주입
    train_set.alpha = args.alpha
    train_set.use_npoint = True if args.alpha > 0 else False
    train_set.adaptive_npoint = args.adaptive_npoint

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
    
    print(f"✨ 학습 시작 [데이터셋: {args.dataset_file} | Alpha: {args.alpha}]")
    
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
                mae_list.append(result[0])
                best_mae = np.min(mae_list)
                
                eval_log = f"--- [Eval] Epoch {epoch} | MAE: {result[0]:.2f} | MSE: {result[1]:.2f} | Best MAE: {best_mae:.2f}"
                print(eval_log)
                with open(run_log_name, "a") as f: f.write(eval_log + "\n")
                
                writer.add_scalar('metric/mae', result[0], epoch)

                if result[0] <= best_mae:
                    torch.save({'model': model_without_ddp.state_dict(), 'epoch': epoch, 'mae': result[0]}, 
                               os.path.join(args.checkpoints_dir, 'best_mae.pth'))

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"⚠️ OOM 발생. 에폭 {epoch}을 건너뜁니다.")
                gc.collect()
                torch.cuda.empty_cache()
                continue
            else: raise e

    writer.close()

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)