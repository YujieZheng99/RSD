"""
DenseNet for CIFAR-10/100 Dataset.

Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
2. https://github.com/liuzhuang13/DenseNet
3. https://github.com/gpleiss/efficient_densenet_pytorch
4. Gao Huang, zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
Densely Connetcted Convolutional Networks. https://arxiv.org/abs/1608.06993

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
import random
import numpy as np

__all__ = [
    "DenseNet",
    "repDenseNetd40k12",
]


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                   padding_mode='zeros'):
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding, dilation=dilation, groups=groups,
                           bias=True, padding_mode=padding_mode)  # note: bias=False
    se = nn.Sequential()
    se.add_module('conv', conv_layer)
    return se


class AMBB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 deploy=False, nonlinear=None):
        super(AMBB, self).__init__()

        self.deploy = deploy
        self.drop = DropPath(drop_prob=0.0)

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
            self.duplicate1 = conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups)

            self.duplicate2 = conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups)

            self.duplicate3 = conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
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
                                      padding=ver_padding, dilation=dilation, groups=groups, bias=True,
                                      padding_mode="zeros")

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                                      stride=stride,
                                      padding=hor_padding, dilation=dilation, groups=groups, bias=True,
                                      padding_mode="zeros")

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        asym_h = asym_kernel.size(2)
        asym_w = asym_kernel.size(3)
        square_h = square_kernel.size(2)
        square_w = square_kernel.size(3)
        square_kernel[:, :, square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
                square_w // 2 - asym_w // 2: square_w // 2 - asym_w // 2 + asym_w] += asym_kernel

    def get_equivalent_kernel_bias(self):
        dup1_k, dup1_b = self.duplicate1.conv.weight, self.duplicate1.conv.bias.data
        dup2_k, dup2_b = self.duplicate2.conv.weight, self.duplicate2.conv.bias.data
        dup3_k, dup3_b = self.duplicate3.conv.weight, self.duplicate3.conv.bias.data
        if hasattr(self, "ver_conv"):
            ver_k, ver_b = self.ver_conv.weight, self.ver_conv.bias.data

        if hasattr(self, "hor_conv"):
            hor_k, hor_b = self.hor_conv.weight, self.hor_conv.bias.data
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
        self.__delattr__('hor_conv')
        self.tdb_reparam.weight.data = deploy_k
        self.tdb_reparam.bias.data = deploy_b

    def switch_drop(self, drop_prob):
        self.drop = DropPath(drop_prob=drop_prob)

    def forward(self, inputs):
        if hasattr(self, 'tdb_reparam'):
            return self.nonlinear(self.tdb_reparam(inputs))

        return self.nonlinear(self.drop(self.duplicate1(inputs)) + self.drop(self.duplicate2(inputs)) + self.drop(self.duplicate3(inputs)) + \
                              self.drop(self.ver_conv(inputs)) + self.drop(self.hor_conv(inputs)))


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(
        self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False
    ):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features)),
        self.add_module("relu1", nn.ReLU(inplace=True)),
        self.add_module(
            "conv1",
            nn.Conv2d(
                num_input_features,
                bn_size * growth_rate,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        ),
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module("relu2", nn.ReLU(inplace=True)),
        self.add_module(
            "conv2",
            AMBB(in_channels=bn_size * growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, padding=1),
        ),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(
            prev_feature.requires_grad for prev_feature in prev_features
        ):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training
            )
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "conv",
            nn.Conv2d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    def __init__(
        self,
        num_layers,
        num_input_features,
        bn_size,
        growth_rate,
        drop_rate,
        efficient=False,
    ):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """

    def __init__(
        self,
        growth_rate=12,
        block_config=[16, 16, 16],
        compression=0.5,
        num_init_features=24,
        bn_size=4,
        drop_rate=0,
        num_classes=10,
        small_inputs=True,
        efficient=False,
        KD=False,
    ):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, "compression of densenet should be between 0 and 1"
        self.avgpool_size = 8 if small_inputs else 7
        self.KD = KD
        # First convolution
        if small_inputs:
            self.features = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv0",
                            AMBB(in_channels=3, out_channels=num_init_features, kernel_size=3, stride=1, padding=1),
                        ),
                    ]
                )
            )
        else:
            self.features = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv0",
                            nn.Conv2d(
                                3,
                                num_init_features,
                                kernel_size=7,
                                stride=2,
                                padding=3,
                                bias=False,
                            ),
                        ),
                    ]
                )
            )
            self.features.add_module("norm0", nn.BatchNorm2d(num_init_features))
            self.features.add_module("relu0", nn.ReLU(inplace=True))
            self.features.add_module(
                "pool0",
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),
            )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=int(num_features * compression),
                )
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module("norm_final", nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

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
        out = self.features.conv0(x)
        f0 = out
        if index == 0:
            mixed_x1, y_a, y_b, lam, idx = self.manifold_mixup(out, out, y)
            mixed_x2, y_a, y_b, lam, _ = self.manifold_mixup(out, feat_s[0], y, idx=idx, lam=lam)
            out = torch.cat([out, mixed_x1, mixed_x2])
        out = self.features.transition1(self.features.denseblock1(out))
        f1 = out
        if index == 1:
            mixed_x1, y_a, y_b, lam, idx = self.manifold_mixup(out, out, y)
            mixed_x2, y_a, y_b, lam, _ = self.manifold_mixup(out, feat_s[1], y, idx=idx, lam=lam)
            out = torch.cat([out, mixed_x1, mixed_x2])
        out = self.features.transition2(self.features.denseblock2(out))
        f2 = out
        if index == 2:
            mixed_x1, y_a, y_b, lam, idx = self.manifold_mixup(out, out, y)
            mixed_x2, y_a, y_b, lam, _ = self.manifold_mixup(out, feat_s[2], y, idx=idx, lam=lam)
            out = torch.cat([out, mixed_x1, mixed_x2])
        out = self.features.denseblock3(out)
        f3 = out
        if index == 3:
            mixed_x1, y_a, y_b, lam, idx = self.manifold_mixup(out, out, y)
            mixed_x2, y_a, y_b, lam, _ = self.manifold_mixup(out, feat_s[3], y, idx=idx, lam=lam)
            out = torch.cat([out, mixed_x1, mixed_x2])
        out = self.features.norm_final(out)
        out = F.relu(out, inplace=True)
        x_f = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(
            out.size(0), -1
        )  # B x 132
        out = self.classifier(x_f)
        if is_feat:
            return [f0, f1, f2, f3], out
        elif feat_s is not None:
            return out, y_a, y_b, lam
        else:
            return out




def repDenseNetd40k12(pretrained=False, path=None, **kwargs):
    """
    Constructs a densenetD40K12 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """

    model = DenseNet(growth_rate=12, block_config=[6, 6, 6], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))["state_dict"])
    return model


if __name__ == '__main__':
    N = 1
    C = 2
    H = 62
    W = 62
    O = 8
    groups = 4

    x = torch.randn(N, C, H, W)
    print('input shape is ', x.size())

    test_kernel_padding = [(3, 1)]
    for k, p in test_kernel_padding:
        tdb = AMBB(C, O, kernel_size=k, padding=p, stride=1, deploy=False)
        tdb.eval()
        for module in tdb.modules():
            if isinstance(module, nn.BatchNorm2d):
                nn.init.uniform_(module.running_mean, 0, 0.1)
                nn.init.uniform_(module.running_var, 0, 0.2)
                nn.init.uniform_(module.weight, 0, 0.3)
                nn.init.uniform_(module.bias, 0, 0.4)
        out = tdb(x)
        tdb.switch_to_deploy()
        deployout = tdb(x)
        print('difference between the outputs of the training-time and converted ACTDB is')
        print(((deployout - out) ** 2).sum())






