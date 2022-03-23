import torch
import os
from PIL import Image
from torchvision import transforms, datasets


class ImageFolderSubsetCCrop(datasets.ImageFolder):
    def __init__(self, root, class_path, transform_rcrop, transform_ccrop, init_box=[0., 0., 1., 1.], **kwargs):
        super().__init__(root=root, **kwargs)
        # load subset samples
        self.class_path = class_path
        new_samples = self.get_class_samples()
        self.imgs = self.samples = new_samples

        # transform
        self.transform_rcrop = transform_rcrop
        self.transform_ccrop = transform_ccrop
        # self.boxes = torch.tensor([0., 0., 1., 1.]).repeat(self.__len__(), 1)
        self.boxes = torch.tensor(init_box).repeat(self.__len__(), 1)
        self.use_box = True

    def get_class_samples(self):
        classes = open(self.class_path).readlines()
        classes = [m.strip() for m in classes]
        classes = set(classes)
        class_to_sample = [[os.path.basename(os.path.dirname(m[0])), m] for m in self.imgs]
        selected_samples = [m[1] for m in class_to_sample if m[0] in classes]

        sorted_classes = sorted(list(classes))
        target_mapping = {self.class_to_idx[k]: j for j, k in enumerate(sorted_classes)}

        valid_pairs = [[m[0], target_mapping[m[1]]] for m in selected_samples]
        return valid_pairs

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


class ImageFolderSubset(datasets.ImageFolder):
    """Folder datasets which returns the index of the image (for memory_bank)
    """
    def __init__(self, class_path, root, transform, **kwargs):
        super(ImageFolderSubset, self).__init__(root, transform, **kwargs)
        self.class_path = class_path
        new_samples = self.get_class_samples()
        self.imgs = self.samples = new_samples  # len=126689

    def get_class_samples(self):
        classes = open(self.class_path).readlines()
        classes = [m.strip() for m in classes]
        classes = set(classes)
        class_to_sample = [[os.path.basename(os.path.dirname(m[0])), m] for m in self.imgs]
        selected_samples = [m[1] for m in class_to_sample if m[0] in classes]

        sorted_classes = sorted(list(classes))
        target_mapping = {self.class_to_idx[k]: j for j, k in enumerate(sorted_classes)}

        valid_pairs = [[m[0], target_mapping[m[1]]] for m in selected_samples]
        return valid_pairs

