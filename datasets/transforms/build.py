# import datasets.transform as transform
from datasets import transforms


def build_transform(cfg):
    args = cfg.copy()
    func_name = args.pop('type')
    return transforms.__dict__[func_name](**args)
