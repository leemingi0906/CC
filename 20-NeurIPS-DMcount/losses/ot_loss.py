import torch
from torch.nn import Module
from .bregman_pytorch import sinkhorn

class OT_Loss(Module):
    def __init__(self, c_size, stride, norm_cood, device, num_of_iter_in_ot=100, reg=10.0):
        super(OT_Loss, self).__init__()
        assert c_size % stride == 0

        self.c_size = c_size
        self.device = device
        self.norm_cood = norm_cood
        self.num_of_iter_in_ot = num_of_iter_in_ot
        self.reg = reg

        # Coordinate grid 생성
        self.cood = torch.arange(0, c_size, step=stride,
                                 dtype=torch.float32, device=device) + stride / 2
        self.density_size = self.cood.size(0)
        self.cood = self.cood.unsqueeze(0) # [1, #cood]
        
        if self.norm_cood:
            self.cood = self.cood / c_size * 2 - 1 
        self.output_size = self.cood.size(1)

    def forward(self, normed_density, unnormed_density, points):
        batch_size = normed_density.size(0)
        assert len(points) == batch_size
        
        # [핵심 수정] loss를 Tensor 리스트로 관리하여 그래프가 끊기지 않게 함
        losses = []
        ot_obj_values = []
        wd_total = 0.0

        for idx, im_points in enumerate(points):
            if len(im_points) > 0:
                # 1. Distance matrix 계산
                if self.norm_cood:
                    im_points = im_points / self.c_size * 2 - 1
                
                x = im_points[:, 0].unsqueeze(1)  
                y = im_points[:, 1].unsqueeze(1)
                
                x_dis = -2 * torch.matmul(x, self.cood) + x * x + self.cood * self.cood
                y_dis = -2 * torch.matmul(y, self.cood) + y * y + self.cood * self.cood
                
                dis = y_dis.unsqueeze(2) + x_dis.unsqueeze(1)
                dis = dis.view((dis.size(0), -1)) # [#gt, #cood * #cood]

                # 2. Sinkhorn OT 계산
                source_prob = normed_density[idx][0].view([-1])
                target_prob = (torch.ones([len(im_points)], device=self.device) / len(im_points))
                
                # Sinkhorn 내부 연산을 위해 detach된 확률값 사용 (안정성)
                with torch.no_grad():
                    P, log = sinkhorn(target_prob, source_prob.detach(), dis.detach(), 
                                      self.reg, maxIter=self.num_of_iter_in_ot, log=True)
                    beta = log['beta'] # [#cood * #cood]
                
                # 3. OT Objective value (로그용)
                ot_obj_values.append(torch.sum(normed_density[idx] * beta.view([1, self.output_size, self.output_size])))
                
                # 4. [핵심] DM-Count의 'Surrogate Loss' 기법 적용
                # 이 기법은 OT의 해석적 그래디언트를 모델 출력에 직접 곱해주는 방식입니다.
                source_density = unnormed_density[idx][0].view([-1])
                source_count = source_density.sum() + 1e-8
                
                # 그래디언트 근사값 계산
                # beta는 detach 상태이므로 unnormed_density에만 grad가 흐르게 됨
                im_grad_1 = (1.0 / source_count) * beta
                im_grad_2 = torch.sum(source_density * beta) / (source_count * source_count)
                im_grad = (im_grad_1 - im_grad_2).detach()
                im_grad = im_grad.view([1, self.output_size, self.output_size])
                
                # 모델 출력과 가중 그래디언트의 곱을 손실로 정의 (backward 시 im_grad가 흐름)
                losses.append(torch.sum(unnormed_density[idx] * im_grad))
                wd_total += torch.sum(dis * P).item()
            else:
                # 점이 없는 경우 더미 0 추가 (그래프 연결 유지)
                losses.append(unnormed_density[idx].sum() * 0.0)

        if len(losses) == 0:
            return torch.zeros([1], device=self.device, requires_grad=True), 0.0, torch.zeros([1], device=self.device)

        return torch.stack(losses).mean(), wd_total / batch_size, torch.stack(ot_obj_values).mean() if ot_obj_values else torch.zeros([1], device=self.device)