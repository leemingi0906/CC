import torchvision.transforms as standard_transforms
# [수정] 폴더.파일명 구조에서 클래스를 정확히 임포트합니다.
# try:
#     from .SHHA.SHHA import SHHA
#     from .SHHB.SHHB import SHHB
# except ImportError:
#     # 패키지 구조가 아닌 직접 실행 시를 위한 대비
#     from SHHA.SHHA import SHHA
#     from SHHB.SHHB import SHHB

try:
    from SHT import SHHA, SHHB
except ImportError:
    from .SHT import SHHA, SHHB


def loading_data(data_root, args=None):
    # 전처리 설정
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
    ])
    
    # args에서 속성을 안전하게 가져옵니다.
    dataset_type = getattr(args, 'dataset_file', 'SHHA')
    use_npoint = getattr(args, 'use_npoint', False)
    alpha = getattr(args, 'alpha', 0.2)
    adaptive_npoint = getattr(args, 'adaptive_npoint', False)

    # 데이터셋 선택
    if dataset_type == 'SHHA':
        DatasetClass = SHHA
    elif dataset_type == 'SHHB':
        DatasetClass = SHHB
    else:
        raise ValueError(f"지원되지 않는 데이터셋 유형: {dataset_type}")

    # 학습 데이터셋 생성
    train_set = DatasetClass(
        data_root, 
        train=True, 
        transform=transform, 
        patch=True, 
        flip=True, 
        use_npoint=use_npoint, 
        adaptive_npoint=adaptive_npoint,
        alpha=alpha
    )
    
    # 검증 데이터셋 생성
    val_set = DatasetClass(
        data_root, 
        train=False, 
        transform=transform, 
        use_npoint=False
    )

    return train_set, val_set