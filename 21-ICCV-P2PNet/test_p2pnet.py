import argparse
import os
import torch
from torch.utils.data import DataLoader
import util.misc as utils
from models import build_model
from crowd_datasets import build_dataset
from engine import evaluate_crowd_no_overlap
import warnings

# 경고 무시
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for testing P2PNet', add_help=False)
    
    # 기본 설정 (학습 시와 동일하게 맞춰야 함)
    parser.add_argument('--backbone', default='vgg16_bn', type=str, help="Name of the convolutional backbone to use")
    parser.add_argument('--row', default=2, type=int, help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int, help="line number of anchor points")
    
    # 데이터셋 및 경로 설정
    parser.add_argument('--dataset_file', default='SHHA', help='dataset name (SHHA or SHHB)')
    parser.add_argument('--data_root', default='/home/mingi/Downloads/SHT', help='path where the dataset is')
    
    # 모델 가중치 경로 (필수 입력)
    parser.add_argument('--weight_path', default='./weights/best_mae.pth', type=str, help='path to the trained model checkpoint')
    
    # 기타 설정
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for testing')
    parser.add_argument('--num_workers', default=8, type=int)
    
    # 모델 빌드에 필요한 더미 인자들 (테스트엔 안 쓰이지만 build_model 호출 시 필요)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=3500, type=int)
    parser.add_argument('--lr_drop', default=3500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)
    parser.add_argument('--frozen_weights', type=str, default=None)
    parser.add_argument('--set_cost_class', default=1, type=float)
    parser.add_argument('--set_cost_point', default=0.05, type=float)
    parser.add_argument('--point_loss_coef', default=0.0002, type=float)
    parser.add_argument('--eos_coef', default=0.5, type=float)

    return parser

def main(args):
    # GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    device = torch.device('cuda')

    # 모델 빌드
    print(f"Loading model from {args.weight_path}...")
    
    # [수정] build_model 반환값 처리 (단일 객체 반환 시 언패킹 오류 방지)
    res = build_model(args, training=False)
    if isinstance(res, tuple):
        model, _ = res
    else:
        model = res

    model.to(device)

    # 가중치 로드
    if os.path.exists(args.weight_path):
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        
        # DataParallel로 저장된 경우 'module.' 접두사 제거
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint # state_dict만 저장된 경우 대비
            
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict)
        print("✅ Model weights loaded successfully.")
    else:
        print(f"❌ Error: Checkpoint not found at {args.weight_path}")
        return

    # 데이터셋 로드 (Validation Set = Test Set)
    loading_data = build_dataset(args=args)
    _, val_set = loading_data(args.data_root) # val_set이 곧 Test Set임
    
    # 데이터 로더 (Test는 batch_size=1 권장)
    sampler_val = torch.utils.data.SequentialSampler(val_set)
    data_loader_val = DataLoader(val_set, 1, sampler=sampler_val,
                                    drop_last=False, collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)

    print(f"Start Testing on {len(val_set)} images...")
    
    # 평가 수행
    model.eval() # 평가 모드 확실히 설정
    mae, mse = evaluate_crowd_no_overlap(model, data_loader_val, device)
    
    print("\n" + "="*40)
    print(f"🏆 Final Test Result for {args.dataset_file}")
    print(f"   MAE: {mae:.2f}")
    print(f"   MSE: {mse:.2f}")
    print("="*40 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet testing script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
