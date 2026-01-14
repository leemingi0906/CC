import torchvision.transforms as standard_transforms
from .SHHA import SHHA

# DeNormalize used to get original images
class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def loading_data(data_root):
    # the pre-proccssing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
    ])
    
    # [수정 1] 학습 데이터셋(train_set) 생성 시 'use_npoint=True' 추가
    # P2PNet 학습 시 NPoint 증강을 활성화합니다.
    train_set = SHHA(data_root, train=True, transform=transform, patch=True, flip=True, use_npoint=True)
    
    # [수정 2] 검증 데이터셋(val_set) 생성 시 'use_npoint=False' 명시 (또는 생략 가능)
    # 검증/테스트 시에는 정답지를 변형하면 안 되므로 False로 설정합니다.
    val_set = SHHA(data_root, train=False, transform=transform, use_npoint=False)

    return train_set, val_set
