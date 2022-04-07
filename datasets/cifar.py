import torch
from PIL import Image
from torchvision import datasets


class CIFAR10_boxes(datasets.CIFAR10):
    def __init__(self, train, root, transform_rcrop, transform_ccrop, init_box=(0., 0., 1., 1.), **kwargs):
        super().__init__(train=train, root=root, **kwargs)
        self.transform_rcrop = transform_rcrop
        self.transform_ccrop = transform_ccrop
        self.boxes = torch.tensor(init_box).repeat(self.__len__(), 1)
        self.use_box = True

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]  # img:(H, W, C)=(32, 32, 3)
        img = Image.fromarray(img)

        if self.use_box:
            box = self.boxes[index].float().tolist()
            img = self.transform_ccrop([img, box])
        else:
            img = self.transform_rcrop(img)

        return img, target


class CIFAR100_boxes(datasets.CIFAR100):
    def __init__(self, train, root, transform_rcrop, transform_ccrop, init_box=(0., 0., 1., 1.), **kwargs):
        super().__init__(train=train, root=root, **kwargs)
        self.transform_rcrop = transform_rcrop
        self.transform_ccrop = transform_ccrop
        self.boxes = torch.tensor(init_box).repeat(self.__len__(), 1)
        self.use_box = True

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]  # img:(H, W, C)=(32, 32, 3)
        img = Image.fromarray(img)

        if self.use_box:
            box = self.boxes[index].float().tolist()
            img = self.transform_ccrop([img, box])
        else:
            img = self.transform_rcrop(img)

        return img, target

