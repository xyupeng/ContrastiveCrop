import torch
import os
from PIL import Image
from torchvision import datasets
import numpy as np


class STL10_boxes(datasets.STL10):
    def __init__(self, split, root, transform_rcrop, transform_ccrop, init_box=(0., 0., 1., 1.), **kwargs):
        assert split in ['train', 'test', 'unlabeled', 'train+unlabeled']
        super().__init__(split=split, root=root, **kwargs)
        self.transform_rcrop = transform_rcrop
        self.transform_ccrop = transform_ccrop
        self.boxes = torch.tensor(init_box).repeat(self.__len__(), 1)
        self.use_box = True

    def __getitem__(self, index):
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.use_box:
            box = self.boxes[index].float().tolist()
            img = self.transform_ccrop([img, box])
        else:
            img = self.transform_rcrop(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
