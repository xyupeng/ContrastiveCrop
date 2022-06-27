import torch
from torchvision import datasets


class Tiny200_boxes(datasets.ImageFolder):
    def __init__(self, root, transform_rcrop, transform_ccrop, init_box=(0., 0., 1., 1.), **kwargs):
        super().__init__(root=root, **kwargs)
        self.transform_rcrop = transform_rcrop
        self.transform_ccrop = transform_ccrop
        self.boxes = torch.tensor(init_box).repeat(self.__len__(), 1)
        self.use_box = True

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)

        if self.use_box:
            box = self.boxes[index].float().tolist()
            img = self.transform_ccrop([img, box])
        else:
            img = self.transform_rcrop(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
