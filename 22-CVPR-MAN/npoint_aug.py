import numpy as np
from scipy.spatial import KDTree

def reflect_boundary(vals, low, high):
    """
    [개선 1] Repeated Reflection (반복 반사)
    값이 경계를 크게 벗어나도 수학적으로 '반사'시켜 범위 내로 안전하게 넣습니다.
    
    Note: 
    - high는 반사 구간의 길이(너비/높이)로 설정하는 것이 수학적으로 정확합니다.
    - 예: 너비가 100이면 [0, 100] 구간에서 반사 후, 최종적으로 [0, 99]로 클리핑합니다.
    """
    span = high - low
    if span <= 0:
        return np.clip(vals, low, high)
    
    # 정규화 후 모듈러 연산으로 '지그재그' 패턴 생성
    norm_vals = (vals - low) % (2 * span)
    reflected = np.where(norm_vals > span, 2 * span - norm_vals, norm_vals)
    
    return reflected + low

def apply_npoint(points, img_shape, alpha=1.0, k=6, max_shift=25):
    """
    Grid-Adaptive NPoint Augmentation (Final Optimized Version)
    
    특징:
    1. Aspect Ratio Adaptive: 이미지 비율에 맞춰 격자(Grid) 개수 자동 계산 (정사각형 격자 유도)
    2. Percentile Threshold: 데이터셋 독립적으로 밀집/한산 구역 자동 판단
    3. Smooth Mapping: 밀도에 따른 Alpha 선형 보간
    4. Repeated Reflect: 경계면 반사 처리 (Off-by-one 수정됨)
    5. Input Validation: 데이터 형태 방어 코드 추가
    """
    
    # [개선 2] 입력 데이터 방어 (Input Validation)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) <= k:
        return points.copy()

    h, w = img_shape[:2]

    # -----------------------------------------------------------
    # 이미지 비율에 따른 격자 크기 자동 계산
    # -----------------------------------------------------------
    base_split = 4  # 짧은 변 기준 등분 수
    aspect_ratio = w / h
    
    if aspect_ratio >= 1: # 가로가 긴 경우 (Landscape)
        rows = base_split
        cols = int(round(base_split * aspect_ratio))
    else: # 세로가 긴 경우 (Portrait)
        cols = base_split
        rows = int(round(base_split * (1 / aspect_ratio)))
    
    # 최소/최대 격자 수 제한
    rows = max(2, min(rows, 8))
    cols = max(2, min(cols, 8))
    
    gh, gw = h / rows, w / cols

    # 1. Global Density Analysis (전역 밀도 분석)
    try:
        tree = KDTree(points)
        dists, _ = tree.query(points, k=k+1)
        all_d_avg = np.mean(dists[:, 1:], axis=1)
    except Exception:
        # KDTree 구축 실패 시 원본 반환
        return points.copy()

    # 2. 데이터셋 독립적 임계값 설정 (Percentile)
    d_dense_ref = np.percentile(all_d_avg, 20)
    d_sparse_ref = np.percentile(all_d_avg, 80)
    
    if d_sparse_ref <= d_dense_ref:
        d_sparse_ref = d_dense_ref + 1e-6

    # 3. Alpha 강도 범위 설정
    target_alpha_min, target_alpha_max = 0.1, 0.5

    new_points = points.copy()
    
    # 4. Grid Loop
    for r in range(rows):
        for c in range(cols):
            # 격자 경계 설정
            y1, x1 = r * gh, c * gw
            y2, x2 = (r + 1) * gh, (c + 1) * gw
            if r == rows - 1: y2 = h
            if c == cols - 1: x2 = w
            
            # 현재 격자 내 포인트 인덱스 추출
            idx_in_grid = np.where(
                (points[:, 0] >= x1) & (points[:, 0] < x2) & 
                (points[:, 1] >= y1) & (points[:, 1] < y2)
            )[0]
            
            if len(idx_in_grid) == 0:
                continue 

            # [최종 수정] 복잡한 조건문 제거 및 로직 단순화
            # KNN 거리가 이미 전역(Global)으로 계산되었으므로,
            # 격자 내 점이 1개라도 그 점의 밀도 정보는 정확합니다.
            # 따라서 별도의 Median 처리 없이 해당 점들의 평균값을 바로 사용합니다.
            local_d_metric = np.mean(all_d_avg[idx_in_grid])
            
            # Smooth Linear Mapping
            t = (local_d_metric - d_dense_ref) / (d_sparse_ref - d_dense_ref)
            t = np.clip(t, 0, 1)
            
            adaptive_alpha = target_alpha_min + t * (target_alpha_max - target_alpha_min)
            final_alpha = alpha * adaptive_alpha
            
            # 5. NPoint 적용
            current_d_avgs = all_d_avg[idx_in_grid]
            effective_radii = np.clip(final_alpha * current_d_avgs, 0, max_shift)
            
            theta = np.random.uniform(0, 2 * np.pi, len(idx_in_grid))
            r_scale = np.sqrt(np.random.uniform(0, 1, len(idx_in_grid)))
            magnitude = effective_radii * r_scale
            
            new_points[idx_in_grid, 0] += magnitude * np.cos(theta)
            new_points[idx_in_grid, 1] += magnitude * np.sin(theta)

    # 6. Boundary Reflect (범위: [0, w], [0, h])
    new_points[:, 0] = reflect_boundary(new_points[:, 0], 0, w)
    new_points[:, 1] = reflect_boundary(new_points[:, 1], 0, h)
    
    # 최종 클리핑 [0, w-1]
    new_points[:, 0] = np.clip(new_points[:, 0], 0, w - 1)
    new_points[:, 1] = np.clip(new_points[:, 1], 0, h - 1)
    
    return new_points