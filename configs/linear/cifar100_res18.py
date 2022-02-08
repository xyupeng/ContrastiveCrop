# model
model = dict(type='ResNet', depth=18, num_classes=100, maxpool=False)
loss = dict(type='CrossEntropyLoss')

# dataset
root = '/path/to/your/dataset'
mean = (0.5071, 0.4867, 0.4408)
std = (0.2675, 0.2565, 0.2761)
batch_size = 512
num_workers = 4
data = dict(
    train=dict(
        ds_dict=dict(
            type='CIFAR100',
            root=root,
            train=True,
        ),
        trans_dict=dict(
            type='cifar_linear',
            mean=mean, std=std
        ),
    ),
    test=dict(
        ds_dict=dict(
            type='CIFAR100',
            root=root,
            train=False,
        ),
        trans_dict=dict(
            type='cifar_test',
            mean=mean, std=std
        ),
    ),
)

# training optimizer & scheduler
epochs = 100
lr = 10.0
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=0)
lr_cfg = dict(  # passed to adjust_learning_rate()
    type='MultiStep',
    steps=epochs,
    lr=lr,
    decay_rate=0.1,
    decay_steps=[60, 80],
)


# log, load & save
log_interval = 20
work_dir = None
resume = None
load = None
port = 10001
