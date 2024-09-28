"""
ResNet for CIFAR-10/100 Dataset.

Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
2. https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua
3. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385

"""

import torch
import torch.nn as nn
from .convnet_utils import conv_bn
import random
import numpy as np

__all__ = ["ResNet", "repResNet50", "repResNet32", "repResNet110", "repWRN20_8"]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv_bn(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=stride, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_bn(in_channels=planes, out_channels=planes, kernel_size=3, padding=1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv_bn(in_channels=inplanes, out_channels=width, kernel_size=1, stride=1)
        self.conv2 = conv_bn(in_channels=width, out_channels=width, kernel_size=3, stride=stride, padding=dilation, groups=groups)
        self.conv3 = conv_bn(in_channels=width, out_channels=planes * self.expansion, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        KD=False,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = conv_bn(in_channels=3, out_channels=self.inplanes, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d()
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.KD = KD
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def manifold_mixup(self, feat1, feat2, y, alpha=2.0, idx=None, lam=None):
        if lam is None:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = lam
        batch_size = feat1.size()[0]
        if idx is None:
            index = torch.randperm(batch_size).cuda()
        else:
            index = idx
        mixed_feat = lam * feat1 + (1 - lam) * feat2[index, :]
        y_a, y_b = y, y[index]
        return mixed_feat, y_a, y_b, lam, index

    def forward(self, x, is_feat=False, feat_s=None, y=None):
        index = random.randint(0, 3)
        if self.training is False:
            index = -1
        if feat_s is None:
            index = -1
        if y is None:
            index = -1
        x = self.conv1(x)
        x = self.relu(x)  # B x 16 x 32 x 32
        f0 = x
        if index == 0:
            mixed_x1, y_a, y_b, lam, idx = self.manifold_mixup(x, x, y)
            mixed_x2, y_a, y_b, lam, _ = self.manifold_mixup(x, feat_s[0], y, idx=idx, lam=lam)
            x = torch.cat([x, mixed_x1, mixed_x2])
        x = self.layer1(x)  # B x 16 x 32 x 32
        f1 = x
        if index == 1:
            mixed_x1, y_a, y_b, lam, idx = self.manifold_mixup(x, x, y)
            mixed_x2, y_a, y_b, lam, _ = self.manifold_mixup(x, feat_s[1], y, idx=idx, lam=lam)
            x = torch.cat([x, mixed_x1, mixed_x2])
        x = self.layer2(x)  # B x 32 x 16 x 16
        f2 = x
        if index == 2:
            mixed_x1, y_a, y_b, lam, idx = self.manifold_mixup(x, x, y)
            mixed_x2, y_a, y_b, lam, _ = self.manifold_mixup(x, feat_s[2], y, idx=idx, lam=lam)
            x = torch.cat([x, mixed_x1, mixed_x2])
        x = self.layer3(x)  # B x 64 x 8 x 8
        f3 = x
        if index == 3:
            mixed_x1, y_a, y_b, lam, idx = self.manifold_mixup(x, x, y)
            mixed_x2, y_a, y_b, lam, _ = self.manifold_mixup(x, feat_s[3], y, idx=idx, lam=lam)
            x = torch.cat([x, mixed_x1, mixed_x2])
        x = self.avgpool(x)  # B x 64 x 1 x 1
        x_f = x.view(x.size(0), -1)  # B x 64
        f4 = x_f
        x = self.fc(x_f)  # B x num_classes
        if is_feat:
            return [f0, f1, f2, f3, f4], x
        elif feat_s is not None:
            return x, y_a, y_b, lam
        else:
            return x


def repResNet20(pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-32 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """

    model = ResNet(BasicBlock, [3, 3, 3], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))["state_dict"])
    return model


def repResNet32(pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-32 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """

    model = ResNet(BasicBlock, [5, 5, 5], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))["state_dict"])
    return model


def repResNet50(**kwargs):
    return ResNet(BasicBlock, [8, 8, 8], **kwargs)


def repResNet110(pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-110 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """

    model = ResNet(Bottleneck, [12, 12, 12], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))["state_dict"])
    return model


def repWRN20_8(pretrained=False, path=None, **kwargs):

    """Constructs a Wide ResNet-28-10 model.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """

    model = ResNet(Bottleneck, [2, 2, 2], width_per_group=64 * 8, **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))["state_dict"])
    return model
