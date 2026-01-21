import torch
import torch.nn as nn

# src 폴더 내의 network.py에서 현대화된 Conv2d를 가져옵니다.
try:
    from src.network import Conv2d
except ImportError:
    from network import Conv2d

class MCNN(nn.Module):
    """
    Multi-column CNN (Zhang et al.)
    입력 이미지의 다양한 사람 크기를 3개의 브랜치로 대응합니다.
    """
    def __init__(self, bn=False):
        super(MCNN, self).__init__()
        
        # Branch 1: Large Kernels
        self.branch1 = nn.Sequential(
            Conv2d(3, 16, 9, same_padding=True, bn=bn),
            nn.MaxPool2d(2),
            Conv2d(16, 32, 7, same_padding=True, bn=bn),
            nn.MaxPool2d(2),
            Conv2d(32, 16, 7, same_padding=True, bn=bn),
            Conv2d(16, 8, 7, same_padding=True, bn=bn)
        )
        
        # Branch 2: Medium Kernels
        self.branch2 = nn.Sequential(
            Conv2d(3, 20, 7, same_padding=True, bn=bn),
            nn.MaxPool2d(2),
            Conv2d(20, 40, 5, same_padding=True, bn=bn),
            nn.MaxPool2d(2),
            Conv2d(40, 20, 5, same_padding=True, bn=bn),
            Conv2d(20, 10, 5, same_padding=True, bn=bn)
        )
        
        # Branch 3: Small Kernels
        self.branch3 = nn.Sequential(
            Conv2d(3, 24, 5, same_padding=True, bn=bn),
            nn.MaxPool2d(2),
            Conv2d(24, 48, 3, same_padding=True, bn=bn),
            nn.MaxPool2d(2),
            Conv2d(48, 24, 3, same_padding=True, bn=bn),
            Conv2d(24, 12, 3, same_padding=True, bn=bn)
        )
        
        # 최종 통합 층: relu=False로 설정하여 출력이 0 아래로 내려갈 수 있게 하여 
        # 학습 초기에 죽은 뉴런(Dead Neuron) 현상을 방지합니다.
        self.fuse = nn.Sequential(
            Conv2d(30, 1, 1, same_padding=True, bn=bn, relu=False)
        )
        
        # 가중치 초기화 실행
        self._initialize_weights()

    def forward(self, im_data):
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x1, x2, x3), 1)
        x = self.fuse(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming 초기화는 ReLU와 궁합이 좋아 학습 초기 속도를 높입니다.
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)