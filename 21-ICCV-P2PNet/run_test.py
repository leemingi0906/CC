import argparse
import os
import torch
import torchvision.transforms as standard_transforms
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from crowd_datasets import loading_data
from models import build_model
from torch.utils.data import DataLoader
import util.misc as utils

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # [1] Unified Dataset Architecture
    parser.add_argument('--dataset', default='shb', choices=['sha', 'shb', 'qnrf', 'cc50'], help='데이터셋 선택')
    parser.add_argument('--data_root', default='../SHT', help='데이터셋 루트 경로')
    parser.add_argument('--model_path', required=True, help='학습된 가중치(.pth)')
    parser.add_argument('--test_fold', type=int, default=0, help='CC50 5-Fold 번호 (0~4)')
    
    # [2] P2PNet Architecture (Must match training params)
    parser.add_argument('--backbone', default='vgg16_bn', type=str, help="name of the convolutional backbone to use")
    parser.add_argument('--row', default=2, type=int, help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int, help="line number of anchor points")
    
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 매퍼: unified args -> loading_data.py args
    args.dataset = args.dataset.lower()
    if args.dataset == 'sha': args.dataset_file = 'SHHA'
    elif args.dataset == 'shb': args.dataset_file = 'SHHB'
    elif args.dataset == 'qnrf': args.dataset_file = 'QNRF'
    elif args.dataset == 'cc50': args.dataset_file = 'CC50'
    
    args.cc50_test_fold = args.test_fold
    args.use_npoint = False
    args.alpha = 0.0

    print(f"📊 [Test] Loading {args.dataset.upper()} dataset...")
    
    # DataLoader 초기화
    _, val_set = loading_data(args.data_root, args)
    
    if len(val_set) == 0:
        print("❌ 오류: 해당 경로에서 평가용 이미지를 찾을 수 없습니다.")
        return
        
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2, collate_fn=utils.collate_fn_crowd)

    # 모델 구축 및 로드
    model = build_model(args)
    model.to(device)
    
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        # P2PNet 저장 포맷 대응 ('model' 딕셔너리 내부)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"✅ 가중치 로드 성공: {args.model_path}")
    else:
        print(f"❌ 오류: 모델 가중치를 찾을 수 없습니다: {args.model_path}")
        return

    model.eval()

    mae, mse_sum = 0.0, 0.0
    print(f"🔎 Starting Inference on {len(val_set)} images...")

    threshold = 0.5

    with torch.no_grad():
        for i, (samples, targets) in enumerate(val_loader):
            samples = samples.to(device)
            # targets is a tuple/list of points, or dicts? SHT.py returns (img, {'point': points})
            gt_cnt = len(targets[0]['point']) if isinstance(targets[0], dict) else len(targets[0])

            outputs = model(samples)
            # P2PNet Logic: Softmax 적용 후 threshold 이상의 anchor만 유효 객체로 판정
            outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
            pred_cnt = int((outputs_scores > threshold).sum())

            mae += abs(pred_cnt - gt_cnt)
            mse_sum += (pred_cnt - gt_cnt)**2

            # 터미널 출력 (모든 이미지)
            print(f"[{i+1}/{len(val_set)}] GT: {gt_cnt} | Pred: {pred_cnt} | Err: {pred_cnt - gt_cnt}")

    avg_mae = mae / len(val_set)
    avg_rmse = np.sqrt(mse_sum / len(val_set))

    # 결과 출력
    print("\n" + "="*50)
    print(f"🏆 P2PNet Test Results ({args.dataset.upper()})")
    print(f"📊 MAE: {avg_mae:.2f}")
    print(f"📊 MSE: {avg_rmse:.2f}")
    print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)