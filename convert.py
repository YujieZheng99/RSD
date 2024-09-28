import argparse
import os
import torch
from models.convnet_utils import switch_conv_bn_impl, switch_deploy_flag
from models import model_dict

parser = argparse.ArgumentParser(description='AMBB Conversion')
parser.add_argument('--load', metavar='LOAD', help='path to the weights file',
                    default='save/student_model/repWRN20_8_cifar100_lr_0.05_decay_0.0005_trial_1/repWRN20_8_last.pth')
parser.add_argument('--save', metavar='SAVE', help='path to the weights file',
                    default='save/student_model/repWRN20_8_cifar100_lr_0.05_decay_0.0005_trial_1/repWRN20_8_deploy.pth')
parser.add_argument('-a', '--arch', metavar='ARCH', default='repWRN20_8',
                    choices=['repvgg16', 'repvgg19', 'repResNet32', 'repResNet110', 'repWRN20_8', 'repDenseNet40_12'])
parser.add_argument('-t', '--blocktype', metavar='BLK', default='AMBB', choices=['AMBB'])


def convert():
    args = parser.parse_args()

    switch_conv_bn_impl(args.blocktype)
    switch_deploy_flag(False)
    train_model = model_dict[args.arch](num_classes=100)

    if 'hdf5' in args.load:
        from models.util import model_load_hdf5
        model_load_hdf5(train_model, args.load)
    elif os.path.isfile(args.load):
        print("=> loading checkpoint '{}'".format(args.load))
        checkpoint = torch.load(args.load)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        elif 'model' in checkpoint:
            checkpoint = checkpoint['model']
        ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
        train_model.load_state_dict(ckpt)
    else:
        print("=> no checkpoint found at '{}'".format(args.load))

    for m in train_model.modules():
        if hasattr(m, 'switch_to_deploy'):
            m.switch_to_deploy()

    torch.save(train_model.state_dict(), args.save)


if __name__ == '__main__':
    convert()
