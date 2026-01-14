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

warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('P2PNet Multi-GPU Training with NPoint', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    # 4개 GPU 사용 시 배치 사이즈 16 권장
    parser.add_argument('--batch_size', default=16, type=int) 
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=3500, type=int)
    parser.add_argument('--lr_drop', default=3500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # [NPoint 설정]
    parser.add_argument('--use_npoint', action='store_true', help='NPoint 증강 활성화 여부')
    parser.add_argument('--alpha', default=0.5, type=float, help='NPoint 노이즈 강도 (alpha)')

    # 모델 파라미터
    parser.add_argument('--frozen_weights', type=str, default=None)
    parser.add_argument('--backbone', default='vgg16_bn', type=str)
    parser.add_argument('--set_cost_class', default=1, type=float)
    parser.add_argument('--set_cost_point', default=0.05, type=float)
    parser.add_argument('--point_loss_coef', default=0.0002, type=float)
    parser.add_argument('--eos_coef', default=0.5, type=float)
    parser.add_argument('--row', default=2, type=int)
    parser.add_argument('--line', default=2, type=int)

    # 데이터셋 설정
    parser.add_argument('--dataset_file', default='SHHA')
    parser.add_argument('--data_root', default='/home/mingi/Downloads/SHT', help='데이터셋 경로')
    parser.add_argument('--output_dir', default='./logs_npoint_a05', help='로그 저장 경로')
    parser.add_argument('--checkpoints_dir', default='./ckpt_npoint_a05', help='체크포인트 저장 경로')
    parser.add_argument('--tensorboard_dir', default='./runs_npoint_a05')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='체크포인트에서 재시작')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--eval', action='store_true')
    # [수정] 멀티프로세싱 오류 방지를 위해 기본 워커 수를 4로 하향 조정
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--eval_freq', default=5, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)

    return parser

def main(args):
    # 사용 가능한 모든 GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(torch.cuda.device_count())))
    device = torch.device('cuda')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    run_log_name = os.path.join(args.output_dir, 'run_log.txt')
    with open(run_log_name, "w") as f:
        f.write(f"시작 시간: {time.strftime('%c')}\n설정: {args}\n")

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 모델 빌드
    model, criterion = build_model(args, training=True)
    model.to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"🚀 {torch.cuda.device_count()}개의 GPU를 사용하여 DataParallel 학습을 시작합니다!")
        model = nn.DataParallel(model)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    criterion.to(device)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad], "lr": args.lr_backbone},
    ]
    optimizer = torch.optim.Adam(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # Resume 로직
    if args.resume:
        if os.path.exists(args.resume):
            print(f"📂 가중치 파일 로드 중: {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu')
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
            model_without_ddp.load_state_dict(new_state_dict)
            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1
        else:
            print(f"⚠️ 경고: '{args.resume}' 파일이 없습니다. 처음부터 시작합니다.")

    # 데이터 로딩
    loading_data = build_dataset(args=args)
    train_set, val_set = loading_data(args.data_root)
    
    if hasattr(train_set, 'use_npoint'):
        train_set.use_npoint = args.use_npoint
        if hasattr(train_set, 'alpha'):
            train_set.alpha = args.alpha
        status = "활성화" if args.use_npoint else "비활성화"
        print(f"⚠️ NPoint 증강이 {status}되었습니다. (alpha={args.alpha})")

    # [수정] pin_memory=False로 설정하여 공유 메모리 부하를 줄임
    data_loader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                   collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers,
                                   pin_memory=False)
    data_loader_val = DataLoader(val_set, 1, shuffle=False,
                                    collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers,
                                    pin_memory=False)

    writer = SummaryWriter(args.tensorboard_dir)
    print(f"학습을 시작합니다 (알파={args.alpha})...")
    mae_list = []
    
    for epoch in range(args.start_epoch, args.epochs):
        t1 = time.time()
        stat = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        t2 = time.time()

        epoch_log = f'[에폭 {epoch}][학습률 {optimizer.param_groups[0]["lr"]:.7f}][소요시간 {t2 - t1:.2f}s]'
        print(epoch_log)
        with open(run_log_name, "a") as f:
            f.write(epoch_log + f" loss: {stat['loss']:.4f}\n")
        
        writer.add_scalar('loss/total', stat['loss'], epoch)
        lr_scheduler.step()

        if not os.path.exists(args.checkpoints_dir):
            os.makedirs(args.checkpoints_dir)
        
        torch.save({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
        }, os.path.join(args.checkpoints_dir, 'latest.pth'))
        
        if epoch % args.eval_freq == 0 and epoch > 0:
            result = evaluate_crowd_no_overlap(model_without_ddp, data_loader_val, device)
            mae_list.append(result[0])
            eval_log = f"MAE: {result[0]:.2f}, MSE: {result[1]:.2f}, Best MAE: {np.min(mae_list):.2f}"
            print(f"--- 테스트 결과: {eval_log}")
            with open(run_log_name, "a") as f:
                f.write(f"TEST: {eval_log}\n")
            writer.add_scalar('metric/mae', result[0], epoch)

            if result[0] <= np.min(mae_list):
                torch.save({'model': model_without_ddp.state_dict(), 'epoch': epoch}, 
                           os.path.join(args.checkpoints_dir, 'best_mae.pth'))

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet Training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
