#!/usr/bin/env bash

set -x
export CUDA_VISIBLE_DEVICES=2,3
p=cpu2
node=ttnusa9

srun --partition ${p} \
     --nodelist ${node} \
     --job-name=pretrain \
     --gres=gpu:2 \
     --ntasks=1 \
     --ntasks-per-node=1 \
     --cpus-per-task=16 \
     --kill-on-bad-exit=1 \
     python DDP_simsiam_ccrop.py configs/small/cifar10/simsiam_alpha0.1_th0.1.py

srun --partition ${p} \
     --nodelist ${node} \
     --job-name=linear \
     --gres=gpu:2 \
     --ntasks=1 \
     --ntasks-per-node=1 \
     --cpus-per-task=16 \
     --kill-on-bad-exit=1 \
     python DDP_linear.py configs/linear/cifar10_res18.py \
         --load ./checkpoints/small/cifar10/simsiam_alpha0.1_th0.1/last.pth
