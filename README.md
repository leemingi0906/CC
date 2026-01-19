# CC

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

- 훈련

  ```shell
  python train_multi_gpu.py \
  --use_npoint \
  --alpha 0.2 \
  --data_root ../SHT \
  --dataset_file SHHA \
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
