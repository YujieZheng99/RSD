import torch
import torch.nn as nn
import argparse
from helper.loops import validate
from models import model_dict
from dataset.cifar100 import get_cifar100_dataloaders
from models.convnet_utils import switch_deploy_flag, switch_conv_bn_impl
from measure_model import cal_param_size


def parse_option():
    parser = argparse.ArgumentParser('argument for validation')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--model', type=str, default='repResNet32',
                        choices=['repvgg16', 'repvgg19', 'repResNet32', 'repResNet110', 'repWRN20_8', 'repDenseNet40_12'])
    parser.add_argument('--model_path', type=str, default='save/student_model/repResNet32_deploy.pth', help='model_path')
    parser.add_argument('-t', '--blocktype', metavar='BLK', default='AMBB', choices=['base', 'AMBB'])
    parser.add_argument('--deploy_flag',  type=bool, default=True)

    opt = parser.parse_args()
    return opt


def main():
    opt = parse_option()
    _, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                     num_workers=opt.num_workers, is_instance=True)

    switch_deploy_flag(opt.deploy_flag)
    switch_conv_bn_impl(opt.blocktype)
    model = model_dict[opt.model](num_classes=100)
    print('total parameters:', cal_param_size(model))
    checkpoint = torch.load(opt.model_path)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.cuda()

    criterion_cls = nn.CrossEntropyLoss()
    test_acc, tect_acc_top5, _ = validate(val_loader, model, criterion_cls, opt)
    print('accuracy: ', test_acc)


if __name__ == "__main__":
    main()
