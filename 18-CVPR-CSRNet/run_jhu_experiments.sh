#!/bin/bash

# =================================================================
# CSRNet + NPoint Augmentation
# Automated Experiment Script for JHU++ Dataset Evaluation
# =================================================================

# JHU++ Dataset
echo "Running JHU++ Experiments..."
/home/kimsooyeon/miniconda3/envs/p2pnet/bin/python train.py --dataset JHU --data_root ../JHU_Processed --alpha 0.0 --epochs 1000 --batch_size 1 --seed 0 --gpu_id 2
/home/kimsooyeon/miniconda3/envs/p2pnet/bin/python train.py --dataset JHU --data_root ../JHU_Processed --alpha 0.25 --epochs 1000 --batch_size 1 --seed 0 --gpu_id 2
/home/kimsooyeon/miniconda3/envs/p2pnet/bin/python train.py --dataset JHU --data_root ../JHU_Processed --alpha 0.5 --epochs 1000 --batch_size 1 --seed 0 --gpu_id 2
/home/kimsooyeon/miniconda3/envs/p2pnet/bin/python train.py --dataset JHU --data_root ../JHU_Processed --alpha 1.0 --epochs 1000 --batch_size 1 --seed 0 --gpu_id 2

echo "All CSRNet JHU experiments finished successfully!"
