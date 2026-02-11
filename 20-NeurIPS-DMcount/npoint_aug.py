import numpy as np
from scipy.spatial import KDTree

def apply_npoint(points, img_shape, alpha=0.5, k=6, max_shift=20):
    """
    NPoint: Density-Adaptive Point Noise Injection (개선판: Area-Uniform)
    경계면 쏠림 현상을 방지하기 위해 허용 범위 내에서 균일한 분포로 노이즈를 생성합니다.
    
    Args:
        points (np.array): (N, 2) 좌표 배열 [[x, y], ...]
        img_shape (tuple): (H, W) 이미지 크기
        alpha (float): 노이즈 강도 계수
        k (int): KNN 이웃 수
        max_shift (int): 이동 거리 상한선 (픽셀 단위)
        
    Returns:
        new_points (np.array): 균일하게 분포된 노이즈가 주입된 새로운 좌표 배열
    """
    
    # 포인트 개수가 k보다 적으면 원본 반환
    if len(points) <= k:
        return points

    # 1. KNN 거리 계산 (Density Estimation)
    tree = KDTree(points)
    dists, _ = tree.query(points, k=k+1)
    # 자기 자신을 제외한 이웃들 간의 평균 거리 계산
    d_avg = np.mean(dists[:, 1:], axis=1)

    # 2. 이동 가능 '최대 반경' 결정 (Clamping 우선 적용)
    # [핵심 수정] 각 점이 이동할 수 있는 이론적 반경을 계산한 뒤, max_shift로 먼저 제한합니다.
    # 이렇게 하면 샘플링 전에 각 점의 '활동 가능 영역'이 확정됩니다.
    effective_max_radii = np.clip(alpha * d_avg, 0, max_shift)

    # 3. 면적 균일 샘플링 (Area-Uniform Sampling)
    # 모든 방향(0 ~ 2pi)으로 무작위 각도 생성
    theta = np.random.uniform(0, 2 * np.pi, len(points))
    
    # [핵심 수정] 결정된 '활동 가능 영역' 내부 어디든 균일한 확률로 존재하게 합니다.
    # r^2이 균일 분포를 따라야 하므로 sqrt(random)을 반경에 곱해줍니다.
    # 이 방식은 경계선에 점이 몰리지 않고 원 내부 면적에 골고루 퍼지게 합니다.
    magnitude = effective_max_radii * np.sqrt(np.random.uniform(0, 1, len(points)))
    
    # 4. 좌표 업데이트
    shift_x = magnitude * np.cos(theta)
    shift_y = magnitude * np.sin(theta)
    
    new_points = points.copy().astype(np.float32)
    new_points[:, 0] += shift_x
    new_points[:, 1] += shift_y

    # 5. Boundary Check (이미지 경계 밖으로 나가는 점들을 안으로 밀어 넣음)
    h, w = img_shape[:2]
    new_points[:, 0] = np.clip(new_points[:, 0], 0, w - 1)
    new_points[:, 1] = np.clip(new_points[:, 1], 0, h - 1)
    
    return new_points