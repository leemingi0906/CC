import argparse
import torch
import os
import numpy as np
import cv2
from torch.utils.data import DataLoader
from models import vgg19
import datasets.crowd as crowd

def parse_args():
    parser = argparse.ArgumentParser(description='DM-Count Test & Visualization (Unified Args)')
    parser.add_argument('--device', default='0', help='assign device')
    
    # [수정] train.py와 동일하게 --model-path 및 --data-root 사용
    parser.add_argument('--model-path', type=str, required=True, help='학습된 모델(.pth 또는 .tar) 경로')
    parser.add_argument('--data-root', type=str, default='./SHT', help='데이터셋 루트 경로 (예: ./SHT)')
    
    parser.add_argument('--dataset', type=str, default='B', choices=['A', 'B'], help='데이터셋 파트 (A 또는 B)')
    parser.add_argument('--crop-size', type=int, default=512, help='테스트 시 참조할 크롭 사이즈')
    parser.add_argument('--pred-density-map-path', type=str, default='', help='밀도 맵 시각화 결과 저장 경로')

    return parser.parse_args()

def val_collate(batch):
    """가변 길이 포인트와 경로 문자열을 처리하기 위한 collate"""
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]
    img_paths = transposed_batch[2]
    return images, points, img_paths

def test():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 데이터셋 경로 설정 (Unified Logic)
    # 입력받은 data_root를 기반으로 ShanghaiTech 구조를 탐색합니다.
    target_path = os.path.join(args.data_root, f'part_{args.dataset}_final', 'test_data')
    
    # 만약 위 경로가 없다면, data_root 자체가 test_data일 가능성을 고려하여 대체 경로 탐색
    if not os.path.exists(target_path):
        alt_path = os.path.join(args.data_root, 'test_data')
        if os.path.exists(alt_path):
            target_path = alt_path
        else:
            # 최종적으로 data_root가 직접 이미지를 포함하고 있는지 확인
            target_path = args.data_root
    
    print(f"📂 [DM-Count Test] 데이터 경로: {os.path.abspath(target_path)}")
    
    if not os.path.exists(target_path):
        print(f"❌ 에러: 경로를 찾을 수 없습니다: {target_path}")
        return

    # 데이터 로더 생성
    dataset = crowd.Crowd(target_path, crop_size=args.crop_size, method='val')
    
    if len(dataset) == 0:
        print(f"⚠️ 경고: 해당 경로에서 이미지를 찾지 못했습니다. 폴더 구조를 확인하세요.")
        return

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=val_collate)

    # 2. 결과 저장 폴더 생성
    if args.pred_density_map_path and not os.path.exists(args.pred_density_map_path):
        os.makedirs(args.pred_density_map_path)

    # 3. 모델 로드
    model = vgg19()
    model.to(device)
    
    if not os.path.exists(args.model_path):
        print(f"❌ 에러: 모델 가중치 파일을 찾을 수 없습니다: {args.model_path}")
        return

    # [수정] weights_only=True를 추가하여 보안 경고를 해결합니다.
    # 인공신경망 가중치(Tensors)만 로드하므로 더 안전하고 빠릅니다.
    try:
        state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    except TypeError:
        # 구버전 PyTorch를 사용하는 경우 weights_only 인자가 없을 수 있으므로 예외 처리
        state_dict = torch.load(args.model_path, map_location=device)

    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    
    model.load_state_dict(state_dict)
    model.eval()

    image_errs = []
    print(f"🚀 테스트 시작 (모델: {os.path.basename(args.model_path)})")

    for i, (inputs, points, paths) in enumerate(dataloader):
        inputs = inputs.to(device)
        img_name = os.path.basename(paths[0]).split('.')[0]
        
        with torch.no_grad():
            outputs, _ = model(inputs)
        
        pred_cnt = torch.sum(outputs).item()
        gt_cnt = len(points[0])
        
        img_err = pred_cnt - gt_cnt
        image_errs.append(img_err)

        if i % 50 == 0:
            print(f"[{i}/{len(dataset)}] {img_name} | GT: {gt_cnt:.1f} | Pred: {pred_cnt:.2f} | Err: {img_err:.2f}")

        # 시각화 저장
        if args.pred_density_map_path:
            vis_img = outputs[0, 0].cpu().numpy()
            vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
            vis_img = (vis_img * 255).astype(np.uint8)
            vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(args.pred_density_map_path, f"{img_name}_pred_{pred_cnt:.1f}.png"), vis_img)

    # 최종 결과 계산
    image_errs = np.array(image_errs)
    mae = np.mean(np.abs(image_errs))
    mse = np.sqrt(np.mean(np.square(image_errs)))

    print("\n" + "="*50)
    print(f"🏆 최종 결과 (Dataset Part {args.dataset})")
    print(f"📊 MAE: {mae:.2f}")
    print(f"📊 MSE: {mse:.2f}")
    print("="*50)

if __name__ == '__main__':
    test()