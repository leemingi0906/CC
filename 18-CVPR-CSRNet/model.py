import torch
import torch.nn as nn
from torchvision import models

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        
        # CSRNet 구성 요소 정의
        # Frontend: VGG16의 앞부분 (특징 추출)
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        # Backend: Dilated Convolution (밀도 맵 추정)
        self.backend_feat  = [512, 512, 512, 256, 128, 64]
        
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        
        # 최종 출력 레이어 (1채널 Density Map)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        # 가중치 초기화 및 로드
        if not load_weights:
            # 1. ImageNet으로 사전 학습된 VGG16 가중치 가져오기
            mod = models.vgg16(pretrained=True)
            
            # 2. 나머지 레이어 초기화 (Gaussian)
            self._initialize_weights()
            
            # 3. VGG16 가중치를 Frontend에 덮어쓰기
            # (state_dict의 키 순서가 동일하므로 순서대로 복사)
            frontend_state_dict = self.frontend.state_dict()
            vgg_state_dict = mod.features.state_dict()
            
            for k, v in vgg_state_dict.items():
                if k in frontend_state_dict:
                    # 차원이 맞는지 확인 후 복사
                    if frontend_state_dict[k].shape == v.shape:
                        frontend_state_dict[k].data[:] = v.data[:]
            
            # 로드된 가중치 적용
            self.frontend.load_state_dict(frontend_state_dict)
            
    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    """
    설정 리스트(cfg)를 받아 nn.Sequential 레이어를 생성하는 헬퍼 함수
    """
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
        
    layers = []
    for v in cfg:
        if v == 'M':
            # Max Pooling
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # Convolution (Dilated or Standard)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
            
    return nn.Sequential(*layers)