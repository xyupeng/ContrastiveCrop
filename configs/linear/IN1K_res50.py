# python DDP_moco_ccrop_fix_center.py path/to/this/config

# model
model = dict(type='ResNet', depth=50, num_classes=1000, maxpool=True)
loss = dict(type='CrossEntropyLoss')

# data
root_train = './data/ImageNet/train'
root_val = './data/ImageNet/val'
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
batch_size = 256
num_workers = 4
data = dict(
    train=dict(
        ds_dict=dict(
            type='ImageFolder',
            root=root_train,
        ),
        trans_dict=dict(
            type='imagenet_linear_train',
            mean=mean, std=std
        ),
    ),
    test=dict(  # not used
        ds_dict=dict(
            type='ImageFolder',
            root=root_val,
        ),
        trans_dict=dict(
            type='imagenet_val',
            mean=mean, std=std
        ),
    ),
)

# training optimizer & scheduler
epochs = 100
lr = 30.
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=0)
lr_cfg = dict(  # passed to adjust_learning_rate(cfg=lr_cfg)
    type='MultiStep',
    steps=epochs,
    lr=lr,
    decay_rate=0.1,
    decay_steps=[60, 80],
    warmup_steps=0,
    # warmup_from=0.01
)


# log & save
log_interval = 100
work_dir = None  # rewritten by args
resume = None
load = None
port = 10001
