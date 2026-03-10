import torchvision.transforms as standard_transforms

try:
    from SHT import SHHA, SHHB
    from QNRF import QNRF
    from CC50 import CC50
except ImportError:
    from .SHT import SHHA, SHHB
    from .QNRF import QNRF
    from .CC50 import CC50

DATASET_REGISTRY = {
    'SHHA': SHHA,
    'SHHB': SHHB,
    'QNRF': QNRF,
    'CC50': CC50
}

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
    cc50_test_fold = getattr(args, 'cc50_test_fold', 0)

    # 데이터셋 선택
    if dataset_type not in DATASET_REGISTRY:
        raise ValueError(f"지원되지 않는 데이터셋 유형: {dataset_type}")
        
    DatasetClass = DATASET_REGISTRY[dataset_type]

    kwargs = {}
    if dataset_type == 'CC50':
        kwargs['cc50_test_fold'] = cc50_test_fold

    # 학습 데이터셋 생성
    train_set = DatasetClass(
        data_root, 
        train=True, 
        transform=transform, 
        patch=True, 
        flip=True, 
        use_npoint=use_npoint, 
        adaptive_npoint=adaptive_npoint,
        alpha=alpha,
        **kwargs
    )
    
    # 검증 데이터셋 생성
    val_set = DatasetClass(
        data_root, 
        train=False, 
        transform=transform, 
        use_npoint=False,
        **kwargs
    )

    return train_set, val_set