import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import argparse
import cv2
import sys
import time
import gc

# [환경 설정] 실행 위치 및 src 폴더를 시스템 경로에 등록
curr_path = os.getcwd()
src_path = os.path.join(curr_path, 'src')
for path in [curr_path, src_path]:
    if path not in sys.path:
        sys.path.insert(0, path)

# 프로젝트 내 모듈 임포트
try:
    from src.data_loader import CrowdDataset
    from src.models import MCNN
    from src.utils import save_results
except ImportError as e:
    print(f"❌ 임포트 오류: {e}")
    sys.exit(1)

def test():
    parser = argparse.ArgumentParser(description='MCNN Unified Test Script')
    
    # 1. 경로 설정 통일화
    parser.add_argument('--dataset', required=True, choices=['sha', 'shb', 'qnrf', 'cc50', 'jhu'], help='데이터셋 선택')
    parser.add_argument('--data_root', required=True, help='데이터셋 루트 경로')
    parser.add_argument('--model_path', required=True, help='Path to .pth weight file')
    parser.add_argument('--test_fold', type=int, default=0, help='CC50 전용 5-Fold 번호 (0~4)')
    
    # 2. 출력 및 시각화 설정
    parser.add_argument('--output_dir', default='./output', help='Directory to save results')
    parser.add_argument('--save_vis', action='store_true', help='예측 결과 이미지 저장 여부')
    parser.add_argument('--gpu_id', default=0, type=int)
    
    args = parser.parse_args()
    args.dataset = args.dataset.lower()
    
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # 훈련 시와 동일한 전처리 적용
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 1. 데이터셋 로드
    print(f"📊 [Test] Loading {args.dataset.upper()}...")
    
    test_set = CrowdDataset(args.data_root, dataset_name=args.dataset, phase='test', transform=transform)
    test_set.test_fold = args.test_fold # Inject test_fold for CC50
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

    # 2. 모델 로드 및 가중치 매핑
    model = MCNN().to(device)
    
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # module. 접두사 제거 (DataParallel 대응)
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print(f"✅ 가중치 로드 성공: {args.model_path}")
    else:
        print(f"❌ 오류: 가중치 파일을 찾을 수 없습니다: {args.model_path}")
        return

    model.eval()

    mae, mse_sum = 0.0, 0.0
    print(f"🔎 Starting Inference on {len(test_set)} images...")

    start_time = time.time()
    with torch.no_grad():
        for i, (img, gt) in enumerate(test_loader):
            img, gt = img.to(device), gt.to(device)
            
            # 모델 예측
            pred = model(img)
            
            # 카운트 계산 (Density Map 픽셀 합)
            p_cnt = torch.sum(pred).item()
            g_cnt = torch.sum(gt).item()
            
            mae += abs(p_cnt - g_cnt)
            mse_sum += (p_cnt - g_cnt)**2

            # [시각화] --save_vis 옵션 시 10장마다 저장
            if args.save_vis and i % 10 == 0:
                save_results(img, gt, pred, args.output_dir, fname=f'test_sample_{i}.png')

    avg_mae = mae / len(test_set)
    avg_rmse = np.sqrt(mse_sum / len(test_set))
    total_time = time.time() - start_time

    # 3. 최종 결과 리포트 작성
    result_text = f"""
========================================
🏆 MCNN Test Results ({args.dataset.upper()})
========================================
- Weight File: {os.path.abspath(args.model_path)}
- Total Images: {len(test_set)}
- MAE (Accuracy): {avg_mae:.2f}
- RMSE (Robustness): {avg_rmse:.2f}
- Total Time: {total_time:.1f}s
- Avg Speed: {total_time/len(test_set):.4f}s per image
========================================
"""
    print(result_text)

    # 결과 리포트 파일 저장
    model_name = os.path.basename(args.model_path).split('.')[0]
    report_name = f'report_{args.dataset}_{model_name}.txt'
    with open(os.path.join(args.output_dir, report_name), 'w') as f:
        f.write(result_text)
    
    print(f"💾 결과가 {args.output_dir}/{report_name}에 저장되었습니다.")

if __name__ == '__main__':
    test()