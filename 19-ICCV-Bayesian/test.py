import torch
import os
import numpy as np
import torch.nn.functional as F
from datasets.crowd import Crowd
from models.vgg import vgg19
import argparse
import sys
import glob
import random


def parse_args():
    parser = argparse.ArgumentParser(description='Bayesian Crowd Counting Unified Test')
    
    parser.add_argument('--dataset', type=str, required=True, choices=['sha', 'shb', 'qnrf', 'cc50', 'jhu'], help='데이터셋 선택')
    parser.add_argument('--data_root', type=str, required=True, help='데이터셋 루트 경로 (예: ../SHT)')
    parser.add_argument('--model_path', type=str, required=True, help='학습된 모델 경로 (.pth)')
    parser.add_argument('--test_fold', type=int, default=0, help='CC50 전용 5-Fold 번호 (0~4)')
    parser.add_argument('--device', default='0', help='사용할 GPU ID')
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--downsample_ratio', type=int, default=8)
    parser.add_argument('--is_gray', type=bool, default=False)

    args = parser.parse_args()
    args.dataset = args.dataset.lower()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # [1] 데이터셋 경로 매핑 로직 (train.py와 동일)
    test_path = ""
    test_file_list = None
    
    if args.dataset == 'sha':
        test_path = os.path.join(args.data_root, 'part_A_final', 'test_data')
    elif args.dataset == 'shb':
        test_path = os.path.join(args.data_root, 'part_B_final', 'test_data')
    elif args.dataset == 'qnrf':
        test_path = os.path.join(args.data_root, 'Test')
        if not os.path.exists(test_path):
            test_path = os.path.join(args.data_root, 'test')
    elif args.dataset == 'jhu':
        test_path = os.path.join(args.data_root, 'test')
    elif args.dataset == 'cc50':
        all_images = sorted(glob.glob(os.path.join(args.data_root, '*.jpg')))
        
        # CC50의 경우 5-Fold 분할을 그대로 따라야 함
        split_rng = random.Random(42) 
        indices = list(range(len(all_images)))
        split_rng.shuffle(indices)
        
        fold_size = len(all_images) // 5
        start_idx = args.test_fold * fold_size
        end_idx = start_idx + fold_size
        
        val_indices = indices[start_idx:end_idx]
        test_file_list = [all_images[i] for i in val_indices]
        test_path = args.data_root # Dummy path

    print(f"📂 [Bayesian Test] Target: {args.dataset.upper()} | Root Path: {os.path.abspath(test_path)}")

    # Override data_root so the dataset class uses the exact test directory we just resolved
    args.data_root = test_path
    
    # Bay_Loss의 Crowd 클래스는 최근 우리가 args 객체를 단일 파라미터로 받도록 개선했음
    if args.dataset == 'cc50':
        dataset = Crowd(args, method='test')
        dataset.im_list = test_file_list # 하드 코딩 리스트 덮어쓰기
    else:
        dataset = Crowd(args, method='test')
        
    if len(dataset) == 0:
        print("❌ 오류: 해당 경로에서 평가용 이미지를 찾을 수 없습니다. 경로가 올바른지 확인해주세요.")
        sys.exit(1)

    print(f"✅ Loaded {len(dataset)} testing images.")

    dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False, num_workers=4, pin_memory=False)

    # [3] 모델 및 웨이트 로드
    model = vgg19()
    model.to(device)
    
    if not os.path.exists(args.model_path):
        print(f"❌ 오류: 지정한 경로에 모델 가중치 파일이 존재하지 않습니다: {args.model_path}")
        sys.exit(1)
        
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except Exception as e:
        print(f"❌ 오류: 모델 로드 실패. ({e})")
        sys.exit(1)
        
    model.eval()

    # [4] 평가 루프
    epoch_minus = []
    
    print(f"🚀 테스트 시작 (Model: {os.path.basename(args.model_path)})")
    
    for inputs, count, name in dataloader:
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        
        with torch.no_grad():
            b, c, h, w = inputs.size()
            long_side = max(h, w)
            if long_side > 2000:
                scale = 2000.0 / long_side
                new_h, new_w = int(h * scale), int(w * scale)
                new_h = (new_h // 8) * 8
                new_w = (new_w // 8) * 8
                inputs = F.interpolate(inputs, size=(new_h, new_w), mode='bilinear', align_corners=False)
            
            outputs = model(inputs)
            # count는 GT 리스트, torch.sum(outputs)는 예측 밀도합
            pred_count = torch.sum(outputs).item()
            gt_count = count[0].item()
            
            temp_minu = gt_count - pred_count
            epoch_minus.append(temp_minu)
            
            # 진행상황 확인을 원하면 주석 해제
            # print(f"File: {name[0]} | GT: {gt_count:.1f} | Pred: {pred_count:.1f} | Diff: {temp_minu:.1f}")

    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    
    print("\n" + "="*50)
    print(f"🏆 최종 테스트 결과 (Dataset: {args.dataset.upper()})")
    print(f"📊 MAE: {mae:.2f}")
    print(f"📊 MSE: {mse:.2f}")
    print("="*50)
