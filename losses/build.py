import losses
import torch


def build_loss(cfg):
    args = cfg.copy()
    name = args.pop('type')
    if hasattr(torch.nn, name):
        criterion = getattr(torch.nn, name)(**args)
    else:
        criterion = losses.__dict__[name](**args)
    return criterion
