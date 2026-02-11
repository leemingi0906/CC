import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

__all__ = ['vgg19']

# PyTorch 공식 서버의 VGG19 사전 학습 가중치 URL
model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        
        # 특징 추출기 이후의 회귀 레이어 설정
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # 최종 밀도 예측 레이어 (1채널)
        self.density_layer = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1), 
            nn.ReLU()
        )

    def forward(self, x):
        # 1. VGG 특징 추출 (Backbone)
        x = self.features(x)
        
        # 2. 업샘플링: bilinear 보간법 사용 (Warning 방지를 위해 F.interpolate 사용)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        # 3. 회귀 및 밀도 맵 생성
        x = self.reg_layer(x)
        mu = self.density_layer(x)
        
        # 4. DM-Count의 핵심: 정규화된 밀도 맵(mu_normed) 계산
        # mu는 실제 카운트 값, mu_normed는 확률 분포 값으로 활용됨
        B, C, H, W = mu.size()
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)
        
        return mu, mu_normed

def make_layers(cfg, batch_norm=False):
    """
    구성 리스트(cfg)를 바탕으로 Conv 레이어들을 생성합니다.
    """
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# VGG-19 구조 설정 (E 구성)
cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

def vgg19():
    """
    VGG 19 레이어 모델 생성 및 ImageNet 가중치 로드
    """
    model = VGG(make_layers(cfg['E']))
    
    # ImageNet으로 사전 학습된 가중치 로드 (strict=False로 추가 레이어 허용)
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    
    return model