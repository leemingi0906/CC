import numpy as np
from scipy.spatial import KDTree

def apply_npoint(points, img_shape, alpha=0.5, k=4, max_shift=20):
    """
    NPoint: Density-Adaptive Point Noise Injection (밀도 적응형 포인트 노이즈 주입)
    원형(Circular) 이동 방식을 사용하여 모든 방향으로 균일한 공간적 변화를 학습시킵니다.
    
    Args:
        points (np.array): (N, 2) 좌표 배열 [[x, y], ...]
        img_shape (tuple): (H, W) 이미지 크기
        alpha (float): 노이즈 강도 계수 (이웃 거리의 몇 %를 이동할지 결정)
        k (int): KNN 이웃 수 (국소 밀도 계산용)
        max_shift (int): 이동 거리 상한선 (픽셀 단위, Safety Clamping)
        
    Returns:
        new_points (np.array): 노이즈가 주입된 새로운 좌표 배열
    """

    
    # 포인트 개수가 k보다 적으면(밀도 계산 불가) 원본 반환
    if len(points) <= k:
        return points

    # 1. KNN 거리 계산 (Density Estimation)
    # KDTree를 사용하여 각 점의 최근접 이웃 거리 계산
    tree = KDTree(points)
    dists, _ = tree.query(points, k=k+1)
    
    # 자기 자신을 제외한 k개 이웃의 평균 거리 
    d_avg = np.mean(dists[:, 1:], axis=1)

    # 2. 원형 노이즈 생성 
    theta = np.random.uniform(0, 2 * np.pi, len(points))
    
    # 원 내부에 균일한 밀도로 점을 배치하기 위해 sqrt(random) 사용
    r_scale = np.sqrt(np.random.uniform(0, 1, len(points)))
    
    # 3. 이동 변위 계산 (Shift Calculation)
    # 밀도가 높으면 d_avg가 작아져서 조금만 이동, 낮으면 많이 이동
    magnitude = alpha * d_avg * r_scale
    
    # [Safety] 과도한 이동 방지 (Clamping)
    magnitude = np.clip(magnitude, 0, max_shift)
    
    # 삼각함수를 이용해 x, y 이동량 계산
    shift_x = magnitude * np.cos(theta)
    shift_y = magnitude * np.sin(theta)
    
    # 4. 좌표 업데이트
    new_points = points.copy()
    new_points[:, 0] += shift_x
    new_points[:, 1] += shift_y

    # [Boundary Check] 이미지 밖으로 나가지 않게 처리
    h, w = img_shape[:2]
    
    # x 좌표 제한 (0 ~ Width)
    new_points[:, 0] = np.clip(new_points[:, 0], 0, w - 1)
    # y 좌표 제한 (0 ~ Height)
    new_points[:, 1] = np.clip(new_points[:, 1], 0, h - 1)
    
    return new_points
