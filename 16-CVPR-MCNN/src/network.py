import torch
import torch.nn as nn
import numpy as np

class Conv2d(nn.Module):
    """
    Conv2d + (Optional BN) + ReLU를 하나로 묶은 헬퍼 클래스
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        
        # padding 계산 logic
        padding = int((kernel_size - 1) / 2) if same_padding else 0
            
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # 원본의 eps, momentum 설정을 유지하면서 현대화
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class FC(nn.Module):
    """Fully Connected Layer wrapper"""
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

def save_net(fname, net):
    """모델 가중치를 H5 파일로 저장 (원본 방식 유지)"""
    import h5py
    # DataParallel인 경우 내부 module만 저장
    state_dict = net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict()
    with h5py.File(fname, 'w') as h5f:
        for k, v in state_dict.items():
            h5f.create_dataset(k, data=v.cpu().numpy())

def load_net(fname, net):
    """H5 또는 PTH 파일로부터 가중치 로드"""
    if fname.endswith('.h5'):
        import h5py
        with h5py.File(fname, 'r') as h5f:
            state_dict = net.state_dict()
            for k, v in state_dict.items():
                if k in h5f:
                    param = torch.from_numpy(np.asarray(h5f[k]))
                    v.copy_(param)
    else:
        # 일반적인 PyTorch 저장 방식 (.pth) 대응
        state_dict = torch.load(fname, map_location='cpu')
        if 'model' in state_dict: state_dict = state_dict['model']
        # module. 접두사 제거 로직
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        net.load_state_dict(new_state_dict)

def np_to_variable(x, is_cuda=True, is_training=False, dtype=torch.FloatTensor):
    """
    [현대화] Variable과 volatile을 제거하고 표준 Tensor 방식으로 변경
    """
    v = torch.from_numpy(x).type(dtype)
    if is_cuda:
        v = v.cuda()
    # 훈련 모드가 아닐 때 (volatile=True의 현대적 대체)
    if not is_training:
        v.requires_grad = False
    return v

def set_trainable(model, requires_grad):
    """레이어들의 학습 여부 동적 설정"""
    for param in model.parameters():
        param.requires_grad = requires_grad

def weights_normal_init(model, dev=0.01):
    """가중치 초기화 유틸리티"""
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)