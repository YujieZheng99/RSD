import torch.nn as nn
import torch
from .dbb_transforms import *


def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                   padding_mode='zeros'):
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding, dilation=dilation, groups=groups,
                           bias=False, padding_mode=padding_mode)  # note: bias=False
    bn_layer = nn.BatchNorm2d(num_features=out_channels, affine=True)
    se = nn.Sequential()
    se.add_module('conv', conv_layer)
    se.add_module('bn', bn_layer)
    return se


class AMBB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 deploy=False, nonlinear=None):
        super(AMBB, self).__init__()

        self.deploy = deploy

        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        assert padding == kernel_size // 2

        if deploy:
            self.tdb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            self.duplicate1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups)

            self.duplicate2 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups)

            self.duplicate3 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups)

            if padding - kernel_size // 2 >= 0:
                #   Common use case. E.g., k=3, p=1 or k=5, p=2
                self.crop = 0
                #   Compared to the KxK layer, the padding of the 1xK layer and Kx1 layer should be adjust to align the sliding windows (Fig 2 in the paper)
                hor_padding = [padding - kernel_size // 2, padding]
                ver_padding = [padding, padding - kernel_size // 2]
            else:
                #   A negative "padding" (padding - kernel_size//2 < 0, which is not a common use case) is cropping.
                #   Since nn.Conv2d does not support negative padding, we implement it manually
                self.crop = kernel_size // 2 - padding
                hor_padding = [0, padding]
                ver_padding = [padding, 0]

            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                                      stride=stride,
                                      padding=ver_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode="zeros")

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                                      stride=stride,
                                      padding=hor_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode="zeros")

            self.ver_bn = nn.BatchNorm2d(num_features=out_channels, affine=True)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels, affine=True)

    def _fuse_bn_tensor(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        asym_h = asym_kernel.size(2)
        asym_w = asym_kernel.size(3)
        square_h = square_kernel.size(2)
        square_w = square_kernel.size(3)
        square_kernel[:, :, square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
                square_w // 2 - asym_w // 2: square_w // 2 - asym_w // 2 + asym_w] += asym_kernel

    def get_equivalent_kernel_bias(self):
        dup1_k, dup1_b = self._fuse_bn_tensor(self.duplicate1.conv, self.duplicate1.bn)
        dup2_k, dup2_b = self._fuse_bn_tensor(self.duplicate2.conv, self.duplicate2.bn)
        dup3_k, dup3_b = self._fuse_bn_tensor(self.duplicate3.conv, self.duplicate3.bn)
        if hasattr(self, "ver_conv") and hasattr(self, "ver_bn"):
            ver_k, ver_b = transI_fusebn(self.ver_conv.weight, self.ver_bn)

        if hasattr(self, "hor_conv") and hasattr(self, "hor_bn"):
            hor_k, hor_b = transI_fusebn(self.hor_conv.weight, self.hor_bn)
        k_origin = dup1_k + dup2_k + dup3_k
        self._add_to_square_kernel(k_origin, hor_k)
        self._add_to_square_kernel(k_origin, ver_k)
        return k_origin, dup1_b + dup2_b + dup3_b + ver_b + hor_b

    def switch_to_deploy(self):
        deploy_k, deploy_b = self.get_equivalent_kernel_bias()
        self.deploy = True
        self.tdb_reparam = nn.Conv2d(in_channels=self.duplicate1.conv.in_channels,
                                     out_channels=self.duplicate1.conv.out_channels,
                                     kernel_size=self.duplicate1.conv.kernel_size,
                                     stride=self.duplicate1.conv.stride,
                                     padding=self.duplicate1.conv.padding,
                                     dilation=self.duplicate1.conv.dilation,
                                     groups=self.duplicate1.conv.groups)
        self.__delattr__('duplicate1')
        self.__delattr__('duplicate2')
        self.__delattr__('duplicate3')
        self.__delattr__('ver_conv')
        self.__delattr__('ver_bn')
        self.__delattr__('hor_conv')
        self.__delattr__('hor_bn')
        self.tdb_reparam.weight.data = deploy_k
        self.tdb_reparam.bias.data = deploy_b

    def forward(self, inputs):
        if hasattr(self, 'tdb_reparam'):
            return self.nonlinear(self.tdb_reparam(inputs))

        return self.nonlinear(self.duplicate1(inputs) + self.duplicate2(inputs) + self.duplicate3(inputs) + \
                               self.ver_bn(self.ver_conv(inputs)) + self.hor_bn(self.hor_conv(inputs)))
