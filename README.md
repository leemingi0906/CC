# CC

16-CVPR-MCNN

- 훈련 (alpha>0이면, NPoint 자동 활성화)

  ```shell
  python train.py\
  --dataset A\
  --alpha 0.0\
  --adaptive_npoint 0\
  --epochs 1000\
  --batch_size 1\
  --seed 0

  #알파값 설정으로 훈련

  ```

- 테스트
  ```shell
  python test.py\
  --dataset A\
  --weight_path ./checkpoints/mcnn_A_a0_0_ad0_s0_best.pth\
  --data_path ./data/original/shanghaitech
  ```

18-CVPR-CSRNet

- 훈련 (alpha>0이면, NPoint 자동 활성화)

  ```shell
  python train.py \
  --dataset_name SHT \
  --part A
  --alpha 0.0 \
  --data_root ./SHT \
  --gpu_id 0 \
  --epochs 1000
  ```
  ```shell
    python train.py \
  --dataset_name QNRF \
  --alpha 1\
  --data_root ./SHT \
  --gpu_id 0 \
  --epochs 1000

  #알파값 설정으로 훈련

- 테스트

  ```shell
  python test_p2pnet.py \
  --weight_path ./ckpt_nponint_a05/best_mae.pth \
  --data_root ../SHT \
  --dataset_file SHHA \
  --gpu_id 0
  ```

  20-NeurIPS-DMcount

- 훈련 (alpha>0이면, NPoint 자동 활성화)#python preprocess_dataset_qnrf.py(전처리 필요)

  ```shell
  python train.py \
  --dataset shb \
  --alpha 0.0 \
  --data_root ./SHT \
  --gpu_id 0 \
  --epochs 1000
  ```
  
  
  ```shell
  python train.py\
  --dataset cc50\
  --data_root ./UCF/UCF_CC_50\
  --test_fold 0\
  --alpha 1.0\
  --epochs 1000
  ```

  ```shell
  python train.py --dataset qnrf \
  --data_root ./UCF-QNRF_Processed \
  --alpha 1.0 \
  --gpu_id 0
  ```

  #알파값 설정으로 훈련

- 테스트
  ```shell
  python test.py \
  --dataset B \
  --data-root ./SHT \
  --model-path ./checkpoints/DM_B_a0_0/best_model.pth\
  ```

21-ICCV-P2PNet

- 초기 설치

  ```shell
  conda create -n p2pnet python=3.8 -y
  conda activate p2pnet
  pip install -r 21-ICCV-P2PNet/requirements.txt

  cd 21-ICCV-P2PNet
  mkdir -p pretrained
  wget https://download.pytorch.org/models/vgg16_bn-6c64b313.pth \
       -O pretrained/vgg16_bn-6c64b313.pth
  ```

- 훈련 (alpha>0이면, NPoint 자동 활성화)

  ```shell
  python train_multi_gpu.py \
  --alpha 0.2 \
  --data_root ../SHT \
  --dataset_file SHHA \
  --adaptive_npoint 10 \
  --batch_size 16 \
  --epochs 3500
  ```

  #알파값 설정으로 훈련

- 테스트
  ```shell
  python test_p2pnet.py \
  --weight_path ./ckpt_nponint_a05/best_mae.pth \
  --data_root ../SHT \
  --dataset_file SHHA \
  --gpu_id 0
  ```
