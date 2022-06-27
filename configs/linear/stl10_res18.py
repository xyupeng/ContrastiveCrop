# model
model = dict(type='ResNet', depth=18, num_classes=10, maxpool=False)
loss = dict(type='CrossEntropyLoss')

# dataset
root = './data'
mean = (0.4406, 0.4273, 0.3858)
std = (0.2312, 0.2265, 0.2237)
batch_size = 512
num_workers = 4
data = dict(
    train=dict(
        ds_dict=dict(
            type='STL10',
            split='train',
            root=root,
        ),
        trans_dict=dict(
            type='stl10_linear',
            mean=mean, std=std
        ),
    ),
    test=dict(
        ds_dict=dict(
            type='STL10',
            split='test',
            root=root,
        ),
        trans_dict=dict(
            type='stl10_test',
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
