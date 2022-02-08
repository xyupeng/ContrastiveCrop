import models


def build_model(cfg):
    args = cfg.copy()
    name = args.pop('type')
    return models.__dict__[name](**args)
