import os
import argparse
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from datasets import build_dataset
from models import build_model
from losses import build_loss
from builder import build_optimizer, build_logger

from utils.util import AverageMeter, TrackMeter, format_time, adjust_learning_rate, accuracy, set_seed
from utils.config import Config, ConfigDict, DictAction


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config file path')  # linear config
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--cfgname', default='linear', help='specify log_file; for debug use')
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
        if args.load:
            cfg.work_dir = os.path.dirname(args.load)
        else:
            cfg.work_dir = os.path.dirname(args.resume)
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


def load_weights(ckpt_path, model, optimizer, resume=False):
    # load checkpoint
    print("==> Loading checkpoint '{}'".format(ckpt_path))
    assert os.path.isfile(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cuda')

    if resume:
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
    else:
        if 'simclr_state' in ckpt.keys():  # simclr
            state_dict = ckpt['simclr_state']
            new_state_dict = {}
            for k, v in state_dict.items():
                newk = k
                if 'fc.' in newk:
                    continue
                new_state_dict[newk] = v
            del state_dict
        elif 'simsiam_state' in ckpt.keys():  # simsiam
            state_dict = ckpt['simsiam_state']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.encoder.') and not k.startswith('module.encoder.fc'):
                    newk = k.replace('encoder.', '')
                    new_state_dict[newk] = v
            del state_dict
        else:  # moco & byol
            for k in ['moco_state', 'byol_state']:
                if k in ckpt.keys():
                    state_dict = ckpt[k]
                    break
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.encoder_q.') and not k.startswith('module.encoder_q.fc'):
                    newk = k.replace('encoder_q.', '')
                    new_state_dict[newk] = v
            del state_dict

        msg = model.load_state_dict(new_state_dict, strict=False)
        assert set(msg.missing_keys) == {'module.fc.weight', 'module.fc.bias'}, set(msg.missing_keys)

    start_epoch = ckpt['epoch'] + 1
    print("Model weights loaded. (epoch {})".format(ckpt['epoch']))
    return start_epoch


def train(train_loader, model, criterion, optimizer, epoch, cfg, logger, writer):
    """one epoch training"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    num_iter = len(train_loader)

    end = time.time()
    time1 = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # compute loss
        logits = model(images)
        loss = criterion(logits, labels)
        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))

        # update metric
        losses.update(loss.item(), bsz)
        top1.update(acc1.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % cfg.log_interval == 0 and logger is not None:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f'Epoch [{epoch}][{idx+1}/{num_iter}] - '
                        f'batch_time: {batch_time.avg:.3f},     '
                        f'lr: {lr:.5f},     '
                        f'loss: {losses.avg:.3f},     '
                        f'Acc@1: {top1.avg:.3f}')

    time2 = time.time()
    epoch_time = format_time(time2 - time1)
    if logger is not None:
        logger.info(f'Epoch [{epoch}] - epoch_time: {epoch_time}, '
                    f'train_loss: {losses.avg:.3f}, '
                    f'train_Acc@1: {top1.avg:.3f}')
    if writer is not None:
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Linear/lr', lr, epoch)
        writer.add_scalar('Linear/train/loss', losses.avg, epoch)
        writer.add_scalar('Linear/train/acc', top1.avg, epoch)
    return losses.avg, top1.avg


def test(test_loader, model, criterion, epoch, logger, writer):
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    time1 = time.time()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            logits = model(images)
            loss = criterion(logits, labels)
            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))

            # update metric
            losses.update(loss.item(), bsz)
            top1.update(acc1.item(), bsz)

    time2 = time.time()
    epoch_time = format_time(time2 - time1)
    if logger is not None:
        logger.info(f'Epoch [{epoch}] - epoch_time: {epoch_time}, '
                    f'test_loss: {losses.avg:.3f}, '
                    f'test_Acc@1: {top1.avg:.3f}')
    if writer is not None:
        writer.add_scalar('Linear/test/loss', losses.avg, epoch)
        writer.add_scalar('Linear/test/acc', top1.avg, epoch)
    return losses.avg, top1.avg


def main():
    # args & cfg
    args = parse_args()
    cfg = get_cfg(args)

    world_size = torch.cuda.device_count()
    print('GPUs on this node:', world_size)
    cfg.world_size = world_size
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, cfg))


def main_worker(rank, world_size, cfg):
    # dist init
    print('==> Start rank:', rank)

    local_rank = rank % world_size
    cfg.local_rank = local_rank
    torch.cuda.set_device(local_rank)
    # cfg.rank = rank

    dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{cfg.port}',
                            world_size=world_size, rank=rank)

    # logger
    logger = None
    writer = None
    if rank == 0:
        logger = build_logger(cfg.work_dir, 'linear')
        writer = SummaryWriter(log_dir=os.path.join(cfg.work_dir, 'tensorboard'))

    # build data loader
    bsz_gpu = int(cfg.batch_size / cfg.world_size)
    print('batch_size per gpu:', bsz_gpu)
    train_set = build_dataset(cfg.data.train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=bsz_gpu, num_workers=cfg.num_workers,
        pin_memory=True, sampler=train_sampler, drop_last=True)

    test_set = build_dataset(cfg.data.test)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=bsz_gpu, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=False)

    # build model and criterion
    model = build_model(cfg.model)
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
    model = model.cuda()

    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.local_rank])
    criterion = build_loss(cfg.loss).cuda()
    optimizer = build_optimizer(cfg.optimizer, parameters)
    start_epoch = 1
    if cfg.resume:
        start_epoch = load_weights(cfg.resume, model, optimizer, resume=True)
    elif cfg.load:
        load_weights(cfg.load, model, optimizer, resume=False)
    cudnn.benchmark = True

    # train loop
    test_meter = TrackMeter()
    print("==> Start training...")
    for epoch in range(start_epoch, cfg.epochs + start_epoch):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(cfg.lr_cfg, optimizer, epoch)

        # train
        train(train_loader, model, criterion, optimizer, epoch, cfg, logger, writer)

        # test & save best
        test_loss, test_acc = test(test_loader, model, criterion, epoch, logger, writer)
        if test_acc > test_meter.max_val and rank == 0:
            model_path = os.path.join(cfg.work_dir, f'best_{cfg.cfgname}.pth')
            state_dict = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'acc': test_acc,
                'epoch': epoch
            }
            torch.save(state_dict, model_path)
        test_meter.update(test_acc, idx=epoch)
        if rank == 0:
            logger.info(f'Best acc: {test_meter.max_val:.2f} (epoch={test_meter.max_idx}).')


if __name__ == '__main__':
    main()
