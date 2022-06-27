from .small import cifar_train_ccrop, cifar_train_rcrop, cifar_linear, cifar_test,\
    stl10_train_rcrop, stl10_train_ccrop, stl10_linear, stl10_test
from .imagenet import imagenet_pretrain_rcrop, imagenet_pretrain_ccrop, imagenet_linear_train, \
    imagenet_val, imagenet_eval_boxes

from .build import build_transform
