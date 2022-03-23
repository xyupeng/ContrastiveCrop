# python DDP_moco_ccrop.py path/to/this/config

# model
dim = 128
model = dict(type='ResNet', depth=50, num_classes=dim, maxpool=True)
moco = dict(dim=dim, K=65536, m=0.999, T=0.07, mlp=False)
loss = dict(type='CrossEntropyLoss')

# data
root_train = '/data/imagenet/train'
class_path = 'datasets/imagenet200.class'
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
batch_size = 256
num_workers = 16
data = dict(
    train=dict(
        ds_dict=dict(
            type='ImageFolderSubsetCCrop',
            root=root_train,
            class_path=class_path,
        ),
        rcrop_dict=dict(
            type='imagenet_pretrain_rcrop',
            mean=mean, std=std
        ),
        ccrop_dict=dict(
            type='imagenet_pretrain_ccrop',
            alpha=0.6,
            mean=mean, std=std
        ),
    ),
    eval_train=dict(
        ds_dict=dict(
            type='ImageFolderSubset',
            root=root_train,
            class_path=class_path,
        ),
        trans_dict=dict(
            type='imagenet_eval_boxes',
            mean=mean, std=std
        ),
    ),
)

# boxes
# loc_interval = 20
# box_thresh = 0.1

# training optimizer & scheduler
epochs = 100
warmup_epochs = epochs
lr = 0.03
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=1e-4)
lr_cfg = dict(  # passed to adjust_learning_rate(cfg=lr_cfg)
    type='MultiStep',
    steps=epochs,
    lr=lr,
    decay_rate=0.,
    decay_steps=[60, 80],
    warmup_steps=0,
    # warmup_from=0.01
)


# log & save
log_interval = 100
save_interval = 50
work_dir = None  # rewritten by args
resume = None
load = None
port = 10001
