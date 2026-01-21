import cv2
import numpy as np
import os
import torch

def denormalize(tensor):
    """
    학습 시 적용된 ImageNet 정규화(Mean/Std)를 역산하여 
    시각화 가능한 이미지로 변환합니다.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = tensor.clone().cpu().numpy()
    if len(img.shape) == 4: # (B, C, H, W) -> (C, H, W)
        img = img[0]
        
    img = img.transpose(1, 2, 0) # (H, W, C)
    img = (img * std + mean) * 255
    return np.clip(img, 0, 255).astype(np.uint8)

def to_heatmap(density_map):
    """
    Density Map(Tensor/Array)을 컬러 히트맵(BGR)으로 변환합니다.
    """
    if torch.is_tensor(density_map):
        density_map = density_map.detach().cpu().numpy()
    
    if len(density_map.shape) == 4: # (B, C, H, W)
        density_map = density_map[0, 0]
    elif len(density_map.shape) == 3: # (C, H, W)
        density_map = density_map[0]

    # 시각화를 위해 0~255 정규화
    mx = np.max(density_map)
    if mx > 0:
        density_map = (density_map / mx) * 255
    
    density_map = density_map.astype(np.uint8)
    heatmap = cv2.applyColorMap(density_map, cv2.COLORMAP_JET)
    return heatmap

def save_results(input_img, gt_data, density_map, output_dir, fname='results.png'):
    """
    [입력 이미지 | GT 히트맵 | 예측 히트맵] 3단 구성을 하나의 파일로 저장합니다.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 이미지 복원
    img_bgr = cv2.cvtColor(denormalize(input_img), cv2.COLOR_RGB2BGR)
    h, w, _ = img_bgr.shape

    # 2. 히트맵 생성 및 크기 조절 (MCNN은 1/4 크기이므로 원본 크기로 복원)
    gt_hmap = to_heatmap(gt_data)
    gt_hmap = cv2.resize(gt_hmap, (w, h))
    
    pred_hmap = to_heatmap(density_map)
    pred_hmap = cv2.resize(pred_hmap, (w, h))

    # 3. 가로로 붙이기
    result_img = np.hstack((img_bgr, gt_hmap, pred_hmap))
    
    # 4. 정보 텍스트 추가 (선택 사항)
    gt_cnt = torch.sum(gt_data).item() if torch.is_tensor(gt_data) else np.sum(gt_data)
    pr_cnt = torch.sum(density_map).item() if torch.is_tensor(density_map) else np.sum(density_map)
    cv2.putText(result_img, f"GT: {gt_cnt:.1f}  Pred: {pr_cnt:.1f}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imwrite(os.path.join(output_dir, fname), result_img)

def save_density_map(density_map, output_dir, fname='density.png'):
    """단일 히트맵 저장"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    heatmap = to_heatmap(density_map)
    cv2.imwrite(os.path.join(output_dir, fname), heatmap)

def display_results(input_img, gt_data, density_map):
    """결과 화면 표시 (서버 환경에서는 작동하지 않을 수 있음)"""
    img_bgr = cv2.cvtColor(denormalize(input_img), cv2.COLOR_RGB2BGR)
    h, w, _ = img_bgr.shape
    
    gt_hmap = cv2.resize(to_heatmap(gt_data), (w, h))
    pred_hmap = cv2.resize(to_heatmap(density_map), (w, h))
    
    result_img = np.hstack((img_bgr, gt_hmap, pred_hmap))
    cv2.imshow('Result [Image | GT | Pred]', result_img)
    cv2.waitKey(0)