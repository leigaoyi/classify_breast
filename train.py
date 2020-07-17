# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:21:17 2020

@author: gaoyilei
"""

from __future__ import print_function
import datetime
import os
import time
import sys
from shutil import copyfile

import torch
import torch.utils.data
from torch import nn
import models
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

try:
    from apex import amp
except ImportError:
    amp = None

import utils
from datasets import InputDataset
from loss import CrossEntropyLossV2

# Use CUDA
#os.environ['CUDA_VISIBLE_DEVICES'] = "7,5,3,6, 2, 1, 0"
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, apex=False, writer=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)
    cur_iter = len(data_loader) * epoch
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        if apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        acc = utils.accuracy(output, target)
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc'].update(acc.item(), n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

        # add summary
        if writer is not None:
            writer.add_scalar("Loss/train", metric_logger.loss.avg, cur_iter)
            writer.add_scalar("Accuracy/train", metric_logger.acc.avg, cur_iter)
        cur_iter += 1


def evaluate(model, criterion, data_loader, device, epoch, print_freq=100, writer=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc = utils.accuracy(output, target)
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc'].update(acc.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    if writer is not None:
        writer.add_scalar("Loss/test", metric_logger.loss.global_avg, epoch)
        writer.add_scalar("Accuracy/test", metric_logger.acc.global_avg, epoch)

    print(' * Acc {acc.global_avg:.3f}'
          .format(acc=metric_logger.acc))
    return metric_logger.acc.global_avg


def load_data(train_csv, val_csv, distributed):
    # Data loading code
    print("Loading data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        #transforms.Resize((96,96), interpolation=2),
        transforms.RandomCrop((96,96), padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #transforms.ColorJitter(brightness=2, contrast=2, saturation=2, hue=0.1),
        transforms.ToTensor(),
        normalize,
        ])

    test_transform = transforms.Compose([
        transforms.Resize((96,96), interpolation=2),
        transforms.ToTensor(),
        normalize,
        ])

    print("Loading training data")
    st = time.time()
    dataset = InputDataset(train_csv, True, train_transform)
    print("Took", time.time() - st)

    print("Loading validation data")
    dataset_test = InputDataset(val_csv, False, test_transform)

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args):
    if args.apex:
        if sys.version_info < (3, 0):
            raise RuntimeError("Apex currently only supports Python 3. Aborting.")
        if amp is None:
            raise RuntimeError("Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                               "to enable mixed-precision training.")

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    #device = torch.device(device)

    torch.backends.cudnn.benchmark = True

    dataset, dataset_test, train_sampler, test_sampler = load_data(args.train_csv, args.val_csv, args.distributed)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    print("Creating model")
    model = models.__dict__[args.model](pretrained=args.pretrained,
            num_classes=args.num_classes)
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    #criterion = CrossEntropyLossV2(label_smoothing=0.1) #nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()

    #optimizer = torch.optim.SGD(
    #    model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.apex_opt_level
                                          )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluate(model, criterion, data_loader_test, device=device)
        return

    # create summary writer
    if args.output_dir and utils.is_main_process():
        writer = SummaryWriter(os.path.join(args.output_dir, "logs"))
    else:
        writer = None

    print("Start training : {0}".format(args.ckpt_name))
    best_result = 0
    best_epoch = 0
    is_best = True
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args.print_freq, args.apex, writer)
        lr_scheduler.step()
        result = evaluate(model, criterion, data_loader_test, device, epoch, args.print_freq, writer)
        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            #utils.save_on_master(
            #    checkpoint,
            #    os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            #utils.save_on_master(
            #    checkpoint,
            #    os.path.join(args.output_dir, 'checkpoint.pth'))
            # save model
            is_best = result > best_result
            if is_best:
                best_result = result
                best_epoch = epoch
            print("Now, Best Acc is {:.3f} at epoch {}".format(best_result,
                    best_epoch))
            
            assert os.path.exists(args.output_dir)
            
            utils.save_checkpoint(checkpoint, is_best,
                    os.path.join(args.output_dir, 'ckpt_{0}.pth'.format(args.ckpt_name)))


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    ## ckpt-name, 1: the bad dataset, 2: the 'train' dif, 3: the teacher provides the dataset
    parser.add_argument('--ckpt-name', default='densenet169', help='name of the ckpt file')
    parser.add_argument('--train-csv', default='./sample_train.csv', help='train csv file')
    parser.add_argument('--val-csv', default='./sample_test.csv', help='validation csv file')
    parser.add_argument('--model', default='densenet169', help='model')
    parser.add_argument('--num_classes', default=2, help='total class')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./ckpt/', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )


    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

