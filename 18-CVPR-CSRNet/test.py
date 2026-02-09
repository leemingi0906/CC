import os
import torch
import numpy as np
import argparse
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import time

# 프로젝트 모듈 임포트
try:
    from model import CSRNet
    from dataset import CSRNet_Dataset
except ImportError:
    print("❌ model.py 또는 dataset.py를 찾을 수 없습니다.")
    import sys
    sys.exit(1)

def get_args():
    parser = argparse.ArgumentParser(description='CSRNet Testing')
    # 1. 경로 설정
    parser.add_argument('--data_root', default='../SHT', help='데이터셋 루트 경로')
    parser.add_argument('--dataset', default='A', choices=['A', 'B'], help='테스트할 데이터셋 파트')
    parser.add_argument('--weight_path', required=True, help='학습된 .pth 파일 경로')
    
    # 2. 기타 설정
    parser.add_argument('--output_dir', default='./output_test', help='결과 시각화 저장 폴더')
    parser.add_argument('--save_vis', action='store_true', help='히트맵 시각화 결과 저장 여부')
    parser.add_argument('--gpu_id', default='0', type=str)
    
    return parser.parse_args()

def test():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # 1. 모델 로드
    model = CSRNet().to(device)
    
    if os.path.exists(args.weight_path):
        print(f"📂 가중치 로드 중: {args.weight_path}")
        checkpoint = torch.load(args.weight_path, map_location=device)
        
        # 가중치 딕셔너리 정제 (Train 시 저장된 구조에 대응)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # DataParallel('module.') 접두사 제거
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print("✅ 모델 로드 완료.")
    else:
        print(f"❌ 오류: 파일을 찾을 수 없습니다: {args.weight_path}")
        return

    model.eval()

    # 2. 데이터 로더
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 테스트 시에는 NPoint를 끕니다.
    test_set = CSRNet_Dataset(args.data_root, part=args.dataset, phase='test', transform=transform, use_npoint=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

    mae, mse = 0.0, 0.0
    print(f"🔎 추론 시작 (총 {len(test_set)}장)...")

    start_time = time.time()
    with torch.no_grad():
        for i, (img, gt) in enumerate(test_loader):
            img, gt = img.to(device), gt.to(device)
            
            # 예측 (Density Map)
            pred = model(img)
            
            # 인원수 계산 (합산)
            p_cnt = pred.sum().item()
            g_cnt = gt.sum().item()
            
            mae += abs(p_cnt - g_cnt)
            mse += (p_cnt - g_cnt)**2

            # [시각화] save_vis 옵션 시 10장마다 결과 저장
            if args.save_vis and i % 10 == 0:
                save_path = os.path.join(args.output_dir, f"test_{args.dataset}_{i}_gt{g_cnt:.1f}_pred{p_cnt:.1f}.jpg")
                
                # 텐서를 numpy 이미지로 변환
                img_np = img[0].cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                h, w = img_bgr.shape[:2]

                # 히트맵 생성 (JET 컬러맵)
                pred_map = pred[0, 0].cpu().numpy()
                pred_map = (pred_map / (pred_map.max() + 1e-6)) * 255
                pred_map = cv2.applyColorMap(pred_map.astype(np.uint8), cv2.COLORMAP_JET)
                pred_map = cv2.resize(pred_map, (w, h))

                # 원본과 히트맵 합성
                vis_img = cv2.addWeighted(img_bgr, 0.6, pred_map, 0.4, 0)
                cv2.imwrite(save_path, vis_img)

    avg_mae = mae / len(test_set)
    avg_rmse = np.sqrt(mse / len(test_set))
    total_time = time.time() - start_time

    print("\n" + "="*40)
    print(f"🏆 CSRNet Test Result (Part {args.dataset})")
    print(f"   - MAE: {avg_mae:.2f}")
    print(f"   - RMSE: {avg_rmse:.2f}")
    print(f"   - Speed: {total_time/len(test_set):.4f}s per image")
    print("="*40)

if __name__ == '__main__':
    test()