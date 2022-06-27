# model
model = dict(type='ResNet', depth=18, num_classes=200, maxpool=False)
loss = dict(type='CrossEntropyLoss')

# dataset
root_train = './data/tiny-imagenet-200/train'
root_test = './data/tiny-imagenet-200/val'
mean = (0.4802, 0.4481, 0.3975)
std = (0.2302, 0.2265, 0.2262)
batch_size = 512
num_workers = 4
data = dict(
    train=dict(
        ds_dict=dict(
            type='ImageFolder',
            root=root_train,
        ),
        trans_dict=dict(
            type='tiny200_linear',
            mean=mean, std=std
        ),
    ),
    test=dict(
        ds_dict=dict(
            type='ImageFolder',
            root=root_test,
        ),
        trans_dict=dict(
            type='tiny200_test',
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
