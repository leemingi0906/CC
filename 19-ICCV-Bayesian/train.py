import argparse
import os
import sys

# 프로젝트 경로 설정
sys.path.append(os.getcwd())

def parse_args():
    parser = argparse.ArgumentParser(description='Bayesian Crowd Counting Training (Clean Version)')
    
    # 쉘 스크립트 호환 인자
    parser.add_argument('--data_root', '--data-dir', required=True, help='원본 데이터 경로 (예: ../SHT)')
    parser.add_argument('--dataset', default='SHB', choices=['SHA', 'SHB', 'QNRF', 'CC50', 'JHU', 'sha', 'shb', 'qnrf', 'cc50', 'jhu'])
    parser.add_argument('--save_dir', '--save-dir', default='./ckpts/baseline', help='저장 경로')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--alpha', default=0.0, type=float, help='NPoint Noise Alpha')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--test_fold', default=0, type=int, help='CC50 5-Fold ID')
    
    # RegTrainer 필수 인자
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--val_epoch', type=int, default=5)
    parser.add_argument('--val_start', type=int, default=0)
    parser.add_argument('--max_model_num', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--is_gray', type=bool, default=False)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--downsample_ratio', type=int, default=8)
    parser.add_argument('--use_background', type=bool, default=True)
    parser.add_argument('--sigma', type=float, default=8.0)
    parser.add_argument('--background_ratio', type=float, default=1.0)
    parser.add_argument('--resume', default='')
    
    args = parser.parse_args()
    args.dataset = args.dataset.upper()
    return args

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    # CUDA_VISIBLE_DEVICES 설정 후 torch 및 내부 모듈들 임포트
    import torch
    import random
    import numpy as np
    try:
        from utils.regression_trainer import RegTrainer
    except ImportError:
        print("❌ 오류: 'utils/regression_trainer.py'를 찾을 수 없습니다.")
        sys.exit(1)
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    args.max_epoch = args.epochs
    args.save_dir = os.path.join(args.save_dir, f'Bayesian_{args.dataset}_a{str(args.alpha).replace(".","_")}_s{args.seed}')
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print(f"🚀 Bayesian Loss 학습 시작 (Path: {args.data_root})")

    # RegTrainer 실행 (내부에서 train_data, test_data를 자동으로 찾습니다)
    trainer = RegTrainer(args)
    
    try:
        trainer.setup()
        print("\n📊 데이터셋 로드 성공:")
        for key, ds in trainer.datasets.items():
            print(f"   👉 [{key.upper()}] 로드 완료: {len(ds)} images")
    except Exception as e:
        print(f"\n❌ 셋업 중 에러 발생: {e}")
        sys.exit(1)
            
    print("\n>>> 학습 루프 시작\n")
    trainer.train()