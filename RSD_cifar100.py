from __future__ import print_function

import os
import sys
import argparse
import socket
import time
import numpy as np
import random

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from models import model_dict
from distiller_zoo import DistillKL

from dataset.cifar100 import get_cifar100_dataloaders

from helper.util import adjust_learning_rate, accuracy, AverageMeter, save_dict_to_json
from helper.loops import validate
from models.convnet_utils import switch_deploy_flag, switch_conv_bn_impl


def parse_option():
    hostname = socket.gethostname()
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=240, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')

    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # Optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # Dataset & Model
    parser.add_argument('--model', type=str, default='repvgg16',
                        choices=['repvgg16', 'repvgg19', 'repResNet32', 'repResNet110', 'repWRN20_8', 'repDenseNet40_12'])
    parser.add_argument('-t', '--blocktype', metavar='BLK', default='AMBB',
                        choices=['base', 'DBB', 'ACB', 'TDB', 'AMBB'])
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    # Hyper-parameters
    parser.add_argument('--kd_T', type=float, default=3, help='temperature for KD distillation')
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help='weight for classification')
    parser.add_argument('-b', '--beta', type=float, default=1.0, help='weight for kd')
    parser.add_argument('-r', '--gamma', type=float, default=0.2, help='drop path rate')

    parser.add_argument('--seed', type=int, default=41, help='random seed')
    parser.add_argument('--trial', type=str, default='1', help='the experiment id')

    opt = parser.parse_args()
    
    # set different learning rate from these 4 models
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/model'
        opt.tb_path = '/path/to/my/tensorboard'
    else:
        opt.model_path = './save/student_model'
        opt.tb_path = './save/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                            opt.weight_decay, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_random_seed(seed=None):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def main():
    opt = parse_option()
    print(opt)
    set_random_seed(opt.seed)

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    switch_deploy_flag(False)
    switch_conv_bn_impl(opt.blocktype)

    # model
    model = model_dict[opt.model](num_classes=n_cls)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    r = opt.gamma
    for m in model.modules():
        if hasattr(m, 'switch_drop'):
            m.switch_drop(r)

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)

    if torch.cuda.is_available():
        model = model.cuda()
        criterion_cls = criterion_cls.cuda()
        criterion_div = criterion_div.cuda()
        cudnn.benchmark = True

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion_cls, criterion_div, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion_cls, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        if epoch == opt.epochs:
            test_merics = {'opt': {'kd_T': opt.kd_T, 'cls_weight': opt.alpha, 'kd_weight': opt.beta, 'drop_rate': opt.gamma},
                           'test_loss': test_loss,
                           'test_acc': float(test_acc),
                           'epoch': epoch}
            save_dict_to_json(test_merics, os.path.join(opt.save_folder, "test_last_metrics.json"))
            print('last accuracy:', test_acc)

            # save the last model
            state = {
               'opt': opt,
               'model': model.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
            torch.save(state, save_file)


def train(epoch, train_loader, model, criterion_cls, criterion_div, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        feat_s, output_s = model(input, is_feat=True)
        loss_cls_s = criterion_cls(output_s, target)

        for m in model.modules():
            if hasattr(m, 'switch_drop'):
                m.switch_drop(0.0)

        output_t, y_a, y_b, lam = model(input, feat_s=feat_s, y=target)
        bz = input.shape[0]
        loss_cls_t = criterion_cls(output_t[0:bz, :], target)
        loss_mix_tt = criterion_cls(output_t[bz:2 * bz, :], y_a) * lam + criterion_cls(output_t[bz:2 * bz, :], y_b) * (1. - lam)
        loss_mix_ts = criterion_cls(output_t[2 * bz:, :], y_a) * lam + criterion_cls(output_t[2 * bz:, :], y_b) * (1. - lam)
        loss_div = criterion_div(output_s, output_t[0:bz, :])
        loss_div_mix = criterion_div(output_t[2 * bz:, :], output_t[bz:2 * bz, :])

        loss = opt.alpha * (loss_cls_s + loss_cls_t + loss_mix_ts + loss_mix_tt) + opt.beta * (loss_div + loss_div_mix)

        r = opt.gamma
        for m in model.modules():
            if hasattr(m, 'switch_drop'):
                m.switch_drop(r)

        acc1, acc5 = accuracy(output_t[0:bz, :], target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


if __name__ == '__main__':
    main()
