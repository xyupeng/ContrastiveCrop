#!/usr/bin/env bash

set -x
export CUDA_VISIBLE_DEVICES=2,3
p=cpu2
node=ttnusa9

srun --partition ${p} \
     --nodelist ${node} \
     --job-name=cifar10_moco \
     --gres=gpu:2 \
     --ntasks=1 \
     --ntasks-per-node=1 \
     --cpus-per-task=16 \
     --kill-on-bad-exit=1 \
     python DDP_moco_ccrop.py configs/small/cifar10/moco_alpha0.1_th0.1.py
