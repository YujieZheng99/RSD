"""
calculate the parameters of original model, trained model and deployed model
"""
import argparse
from models import model_dict
from models.convnet_utils import switch_deploy_flag, switch_conv_bn_impl


def parse_option():
    parser = argparse.ArgumentParser('argument for measure model')

    parser.add_argument('--model_original', type=str, default='MobileNetV2',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'ResNet50',
                                 'resnet56', 'resnet110', 'resnet8x4', 'resnet32x4', 'vgg8', 'vgg11',
                                 'vgg13', 'vgg16', 'vgg19', 'MobileNetV2'])
    parser.add_argument('--model_rep', type=str, default='repmobilenetv2',
                        choices=['represnet20', 'represnet32', 'repvgg8', 'repmobilenetv2'])
    parser.add_argument('-t', '--blocktype', metavar='BLK', default='ACTDB', choices=['DBB', 'ACB', 'base', 'ACDBB', "ACTDB"])

    opt = parser.parse_args()

    return opt


def cal_param_size(model):
    return sum([i.numel() for i in model.parameters()])


def main():
    opt = parse_option()
    model_original = model_dict[opt.model_original]()
    switch_deploy_flag(False)
    switch_conv_bn_impl(opt.blocktype)
    model_train = model_dict[opt.model_rep]()
    switch_deploy_flag(True)
    model_deploy = model_dict[opt.model_rep]()
    print('{}:'.format(opt.model_original), cal_param_size(model_original))
    print('{}_train:'.format(opt.model_rep), cal_param_size(model_train))
    print('{}_deploy:'.format(opt.model_rep), cal_param_size(model_deploy))


if __name__ == '__main__':
    main()
