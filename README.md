## Crafting Better Contrastive Views for Siamese Representation Learning

This is the official PyTorch implementation of the [ContrastiveCrop paper](https://arxiv.org/abs/2202.03278):
```
@article{peng2021crafting,
  title={CRAFTING BETTER CONTRASTIVE VIEWS FOR SIAMESE REPRESENTATION LEARNING},
  author={Peng, Xiangyu and Wang, Kai and Zhu, Zheng and You, Yang},
  year={2021}
}
```
## Preparation
1. Create a python enviroment with `pytorch >= 1.8.1`.
2. `pip install -r requirements.txt`
3. Modify dataset `root` in the config file.

## Pre-train
```
# MoCo, CIFAR-10
python DDP_moco_ccrop.py configs/small/cifar10/moco_alpha0.1_th0.1.py

# SimSiam, CIFAR-100
python DDP_simsiam_ccrop.py configs/small/cifar100/simsiam_alpha0.1_th0.1.py
```
## Linear Evaluation
```
# CIFAR-10
python DDP_linear.py configs/linear/cifar10_res18.py --load ./checkpoints/small/cifar10/moco_alpha0.1_th0.1/last.pth

# CIFAR-100
python DDP_linear.py configs/linear/cifar100_res18.py --load ./checkpoints/small/cifar100/simsiam_alpha0.1_th0.1/last.pth
```

More models and datasets coming soon.
