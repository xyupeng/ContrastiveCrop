"""Build optimizers and schedulers"""
import torch
import os
import logging


def build_optimizer(cfg, params):
    # cfg: ConfigDict
    # TODO: print(type(cfg))
    args = cfg.copy()
    name = args.pop('type')
    if hasattr(torch.optim, name):
        optimizer = getattr(torch.optim, name)(params, **args)
    else:
        raise ValueError(f'torch.optim has no optimizer \'{name}\'.')
    return optimizer


def build_scheduler(cfg, **kwargs):
    # cfg: ConfigDict
    args = cfg.copy()
    name = args.pop('type')
    if hasattr(torch.optim.lr_scheduler, name):
        scheduler = getattr(torch.optim.lr_scheduler, name)(**kwargs, **args)
    else:
        raise ValueError(f'torch.optim.lr_scheduler has no scheduler\'{name}\'.')
    return scheduler


def build_logger(work_dir, cfgname):
    log_file = cfgname + '.log'
    log_path = os.path.join(work_dir, log_file)

    logger = logging.getLogger(cfgname)
    logger.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    handler1 = logging.FileHandler(log_path)
    handler1.setFormatter(formatter)

    handler2 = logging.StreamHandler()
    handler2.setFormatter(formatter)

    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.propagate = False

    return logger
