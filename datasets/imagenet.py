import torch
import os
from PIL import Image
from torchvision import transforms, datasets


class ImageFolderCCrop(datasets.ImageFolder):
    def __init__(self, root, transform_rcrop, transform_ccrop, init_box=[0., 0., 1., 1.], **kwargs):
        super().__init__(root=root, **kwargs)
        # transform
        self.transform_rcrop = transform_rcrop
        self.transform_ccrop = transform_ccrop

        # self.boxes = torch.tensor([0., 0., 1., 1.]).repeat(self.__len__(), 1)
        self.boxes = torch.tensor(init_box).repeat(self.__len__(), 1)
        self.use_box = True

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.use_box:
            box = self.boxes[index].float().tolist()  # box=[h_min, w_min, h_max, w_max]
            sample = self.transform_ccrop([sample, box])
        else:
            sample = self.transform_rcrop(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
