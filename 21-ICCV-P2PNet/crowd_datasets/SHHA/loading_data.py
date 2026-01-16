import torchvision.transforms as standard_transforms
from .SHHA import SHHA

def loading_data(data_root, args): # args 인자를 추가합니다.
    # 전처리 설정
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
    ])
    
    # [수정] args에서 use_npoint와 alpha 값을 가져와 SHHA에 전달합니다.
    train_set = SHHA(
        data_root, 
        train=True, 
        transform=transform, 
        patch=True, 
        flip=True, 
        use_npoint=args.use_npoint, 
        alpha=args.alpha # 터미널 인자 적용
    )
    
    # 검증셋은 증강을 하지 않으므로 고정값 사용
    val_set = SHHA(
        data_root, 
        train=False, 
        transform=transform, 
        use_npoint=False
    )

    return train_set, val_set