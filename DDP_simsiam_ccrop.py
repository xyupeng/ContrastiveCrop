import os
import argparse
import time
import math

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from builder import build_optimizer, build_logger
from models import SimSiam, build_model
from losses import build_loss
from datasets import build_dataset, build_dataset_ccrop

from utils.util import AverageMeter, format_time, set_seed, adjust_lr_simsiam
from utils.config import Config, ConfigDict, DictAction


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--cfgname', help='specify log_file; for debug use')
    parser.add_argument('--resume', type=str, help='path to resume checkpoint (default: None)')
    parser.add_argument('--load', type=str, help='Load init weights for fine-tune (default: None)')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='update the config; e.g., --cfg-options use_ema=True k1=a,b k2="[a,b]"'
                             'Note that the quotation marks are necessary and that no white space is allowed.')
    args = parser.parse_args()
    return args


def get_cfg(args):
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        dirname = os.path.dirname(args.config).replace('configs', 'checkpoints', 1)
        filename = os.path.splitext(os.path.basename(args.config))[0]
        cfg.work_dir = os.path.join(dirname, filename)
    os.makedirs(cfg.work_dir, exist_ok=True)

    # cfgname
    if args.cfgname is not None:
        cfg.cfgname = args.cfgname
    else:
        cfg.cfgname = os.path.splitext(os.path.basename(args.config))[0]
    assert cfg.cfgname is not None

    # seed
    if args.seed != 0:
        cfg.seed = args.seed
    elif not hasattr(cfg, 'seed'):
        cfg.seed = 42
    set_seed(cfg.seed)

    # resume or load init weights
    if args.resume:
        cfg.resume = args.resume
    if args.load:
        cfg.load = args.load
    assert not (cfg.resume and cfg.load)

    return cfg


def load_weights(ckpt_path, train_set, model, optimizer, resume=True):
    # load checkpoint
    print("==> Loading checkpoint '{}'".format(ckpt_path))
    assert os.path.isfile(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location='cuda')

    if resume:
        # load model & optimizer
        train_set.boxes = checkpoint['boxes'].cpu()
        model.load_state_dict(checkpoint['simsiam_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    else:
        raise ValueError

    start_epoch = checkpoint['epoch'] + 1
    print("Loaded. (epoch {})".format(checkpoint['epoch']))
    return start_epoch


def update_box(eval_train_loader, model, len_ds, logger, t=0.05):
    if logger:
        logger.info(f'==> Start updating boxes...')
    model.eval()
    boxes = []
    t1 = time.time()
    for cur_iter, (images, _) in enumerate(eval_train_loader):  # drop_last=False
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            feat_map = model(images, return_feat=True)  # (N, C, H, W)
        N, Cf, Hf, Wf = feat_map.shape
        eval_train_map = feat_map.sum(1).view(N, -1)  # (N, Hf*Wf)
        eval_train_map = eval_train_map - eval_train_map.min(1, keepdim=True)[0]
        eval_train_map = eval_train_map / eval_train_map.max(1, keepdim=True)[0]
        eval_train_map = eval_train_map.view(N, 1, Hf, Wf)
        eval_train_map = F.interpolate(eval_train_map, size=images.shape[-2:], mode='bilinear')  # (N, 1, Hi, Wi)
        Hi, Wi = images.shape[-2:]

        for hmap in eval_train_map:
            hmap = hmap.squeeze(0)  # (Hi, Wi)

            h_filter = (hmap.max(1)[0] > t).int()
            w_filter = (hmap.max(0)[0] > t).int()

            h_min, h_max = torch.nonzero(h_filter).view(-1)[[0, -1]] / Hi  # [h_min, h_max]; 0 <= h <= 1
            w_min, w_max = torch.nonzero(w_filter).view(-1)[[0, -1]] / Wi  # [w_min, w_max]; 0 <= w <= 1
            boxes.append(torch.tensor([h_min, w_min, h_max, w_max]))

    boxes = torch.stack(boxes, dim=0).cuda()  # (num_iters, 4)
    gather_boxes = [torch.zeros_like(boxes) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_boxes, boxes)
    all_boxes = torch.stack(gather_boxes, dim=1).view(-1, 4)
    all_boxes = all_boxes[:len_ds]
    if logger is not None:  # cfg.rank == 0
        t2 = time.time()
        epoch_time = format_time(t2 - t1)
        logger.info(f'Update box: {epoch_time}')
    return all_boxes


def train(train_loader, model, criterion, optimizer, epoch, cfg, logger, writer):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_iter = len(train_loader)
    end = time.time()
    time1 = time.time()
    for idx, (images, _) in enumerate(train_loader):
        images[0] = images[0].cuda(non_blocking=True)
        images[1] = images[1].cuda(non_blocking=True)

        # measure data time
        data_time.update(time.time() - end)

        # compute output
        p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
        loss = -0.5 * (criterion(p1, z2).mean() + criterion(p2, z1).mean())
        losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print info
        if (idx + 1) % cfg.log_interval == 0 and logger is not None:  # cfg.rank == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f'Epoch [{epoch}][{idx+1}/{num_iter}] - '
                        f'data_time: {data_time.avg:.3f},     '
                        f'batch_time: {batch_time.avg:.3f},     '
                        f'lr: {lr:.5f},     '
                        f'loss: {loss:.3f}({losses.avg:.3f})')

    if logger is not None:  # cfg.rank == 0
        time2 = time.time()
        epoch_time = format_time(time2 - time1)
        logger.info(f'Epoch [{epoch}] - epoch_time: {epoch_time}, '
                    f'train_loss: {losses.avg:.3f}')
    if writer is not None:
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Pretrain/lr', lr, epoch)
        writer.add_scalar('Pretrain/loss', losses.avg, epoch)


def main():
    # args & cfg
    args = parse_args()
    cfg = get_cfg(args)

    world_size = torch.cuda.device_count()
    print('GPUs on this node:', world_size)
    cfg.world_size = world_size

    # write cfg
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg.work_dir, f'{timestamp}.cfg')
    with open(log_file, 'a') as f:
        f.write(cfg.pretty_text)

    # spawn
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, cfg))


def main_worker(rank, world_size, cfg):
    print('==> Start rank:', rank)

    local_rank = rank % 8
    cfg.local_rank = local_rank
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{cfg.port}',
                            world_size=world_size, rank=rank)

    # build logger, writer
    logger, writer = None, None
    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(cfg.work_dir, 'tensorboard'))
        logger = build_logger(cfg.work_dir, 'pretrain')

    # build data loader
    bsz_gpu = int(cfg.batch_size / cfg.world_size)
    print('batch_size per gpu:', bsz_gpu)

    train_set = build_dataset_ccrop(cfg.data.train)
    len_ds = len(train_set)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=bsz_gpu,
        num_workers=cfg.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    eval_train_set = build_dataset(cfg.data.eval_train)
    eval_train_sampler = torch.utils.data.distributed.DistributedSampler(eval_train_set, shuffle=False)
    eval_train_loader = torch.utils.data.DataLoader(
        eval_train_set,
        batch_size=bsz_gpu,
        num_workers=cfg.num_workers,
        pin_memory=True,
        sampler=eval_train_sampler,
        drop_last=False
    )

    # build model, criterion; optimizer
    encoder = build_model(cfg.model)
    model = SimSiam(encoder, **cfg.simsiam)  # cfg.simsiam.dim, pred_dim
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.local_rank])
    criterion = build_loss(cfg.loss).cuda()
    if cfg.fix_pred_lr:
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()
    optimizer = build_optimizer(cfg.optimizer, optim_params)

    start_epoch = 1
    if cfg.resume:
        start_epoch = load_weights(cfg.resume, train_set, model, optimizer, resume=True)
    cudnn.benchmark = True

    # Start training
    print("==> Start training...")
    for epoch in range(start_epoch, cfg.epochs + 1):
        train_sampler.set_epoch(epoch)
        adjust_lr_simsiam(cfg.lr_cfg, optimizer, epoch)

        # start ContrastiveCrop
        train_set.use_box = epoch >= cfg.warmup_epochs + 1

        # train; all processes
        train(train_loader, model, criterion, optimizer, epoch, cfg, logger, writer)

        # update boxes; all processes
        if epoch >= cfg.warmup_epochs and epoch != cfg.epochs and epoch % cfg.loc_interval == 0:
            # all_boxes: tensor (len_ds, 4); (h_min, w_min, h_max, w_max)
            all_boxes = update_box(eval_train_loader, model.module.encoder, len_ds, logger,
                                   t=cfg.box_thresh)  # on_cuda=True
            assert len(all_boxes) == len_ds
            train_set.boxes = all_boxes.cpu()

        # save ckpt; master process
        if rank == 0 and epoch % cfg.save_interval == 0:
            model_path = os.path.join(cfg.work_dir, f'epoch_{epoch}.pth')
            state_dict = {
                'optimizer_state': optimizer.state_dict(),
                'simsiam_state': model.state_dict(),
                'boxes': train_set.boxes,
                'epoch': epoch
            }
            torch.save(state_dict, model_path)

    # save the last model; master process
    if rank == 0:
        model_path = os.path.join(cfg.work_dir, 'last.pth')
        state_dict = {
            'optimizer_state': optimizer.state_dict(),
            'simsiam_state': model.state_dict(),
            'boxes': train_set.boxes,
            'epoch': cfg.epochs
        }
        torch.save(state_dict, model_path)


if __name__ == '__main__':
    main()
