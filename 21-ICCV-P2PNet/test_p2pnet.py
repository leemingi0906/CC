import argparse
import os
import torch
import sys
from torch.utils.data import DataLoader
import util.misc as utils
from models import build_model
from crowd_datasets import build_dataset
from engine import evaluate_crowd_no_overlap
import warnings
import time

# 불필요한 경고 무시
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('P2PNet Test Set Evaluation Script', add_help=False)
    
    # 1. 모델 구조 설정 (학습 시와 동일하게 설정)
    parser.add_argument('--backbone', default='vgg16_bn', type=str, help="Backbone 모델 명")
    parser.add_argument('--row', default=2, type=int, help="앵커 포인트 행 수")
    parser.add_argument('--line', default=2, type=int, help="앵커 포인트 열 수")
    
    # 2. 데이터셋 및 경로 설정
    parser.add_argument('--dataset_file', default='SHHA', help='데이터셋 종류 (SHHA 또는 SHHB)')
    parser.add_argument('--data_root', default='/home/kimsooyeon/Downloads/SHT', help='데이터셋 루트 경로')
    
    # 3. 가중치 파일 경로
    parser.add_argument('--weight_path', default='', type=str, required=True, help='가중치 파일(.pth) 경로')
    
    # 4. 결과 저장 및 하드웨어 설정
    parser.add_argument('--output_dir', default='./logs_test_result', help='결과 저장 폴더')
    parser.add_argument('--gpu_id', default=0, type=int, help='GPU ID')
    parser.add_argument('--num_workers', default=4, type=int)
    
    # build_model 호환용 더미 인자 (테스트 시 값 무관)
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
    # [환경 설정] 현재 경로를 파이썬 경로에 등록하여 모듈 탐색 우선순위 확보
    sys.path.append(os.getcwd())
    
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # 1. 모델 빌드
    print(f"🚀 [Step 1] 모델 구조 생성 중: {args.backbone}")
    res = build_model(args, training=False)
    model = res[0] if isinstance(res, tuple) else res
    model.to(device)

    # 2. 가중치 로드
    if os.path.exists(args.weight_path):
        print(f"📂 [Step 2] 가중치 로드: {args.weight_path}")
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # DataParallel('module.') 제거
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print("✅ 가중치 로드 완료.")
    else:
        print(f"❌ 오류: 가중치 파일을 찾을 수 없습니다: {args.weight_path}")
        return

    # 3. 공식 테스트 데이터셋 로딩
    print(f"📊 [Step 3] 공식 테스트셋(Test Set) 구성 시도: {args.dataset_file}")
    loading_data_fn = None
    
    try:
        from crowd_datasets.loading_data import loading_data as loading_data_fn
        print(f"🔍 커스텀 로더(loading_data.py)를 사용합니다.")
    except ImportError:
        print("⚠️ 커스텀 로더가 없어 기본 Factory 빌더를 사용합니다.")
        try:
            loading_data_fn = build_dataset(args=args)
        except:
            print("❌ 치명적 오류: 데이터셋 빌더를 찾을 수 없습니다.")
            return

    # 실제 데이터셋 객체 생성
    try:
        # loading_data_fn은 (train_set, test_set) 튜플을 반환함
        # 우리는 두 번째 인자인 공식 테스트 데이터셋만 사용함
        import inspect
        sig = inspect.signature(loading_data_fn)
        if len(sig.parameters) >= 2:
            _, test_set = loading_data_fn(args.data_root, args)
        else:
            _, test_set = loading_data_fn(args.data_root)
            
        print(f"✅ {args.dataset_file} 공식 테스트 데이터 로딩 완료 (이미지 {len(test_set)}장)")
    except Exception as e:
        print(f"❌ 데이터셋 생성 중 오류 발생: {e}")
        return
    
    # 테스트용 데이터 로더 (배치 사이즈 1 고정)
    data_loader_test = DataLoader(
        test_set, 1, shuffle=False, num_workers=args.num_workers,
        collate_fn=utils.collate_fn_crowd
    )

    # 4. 성능 평가 (Evaluation on Test Set)
    print(f"🔎 [Step 4] 최종 테스트(Official Test) 시작...")
    model.eval()
    start_time = time.time()
    
    with torch.no_grad():
        # engine.py의 공식 평가 로직 사용 (여기서 MAE, MSE 산출)
        mae, mse = evaluate_crowd_no_overlap(model, data_loader_test, device)
    
    end_time = time.time()
    
    # 5. 결과 리포트 출력 및 저장
    result_text = f"""
========================================
🏆 {args.dataset_file} 공식 테스트(Test Set) 결과
========================================
- 가중치 파일: {args.weight_path}
- 테스트 이미지 수: {len(test_set)}
- MAE (정확도): {mae:.2f}
- MSE (강건성): {mse:.2f}
- 총 소요 시간: {end_time - start_time:.2f}s
========================================
"""
    print(result_text)
    
    # 결과 파일 저장
    res_path = os.path.join(args.output_dir, f"official_test_res_{args.dataset_file}.txt")
    with open(res_path, "w") as f:
        f.write(result_text)
    print(f"💾 공식 테스트 결과가 저장되었습니다: {res_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet official evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)