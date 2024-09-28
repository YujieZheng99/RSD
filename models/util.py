from __future__ import print_function

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import math
import torch.nn.functional as F


class Paraphraser(nn.Module):
    """Paraphrasing Complex Network: Network Compression via Factor Transfer"""
    def __init__(self, t_shape, k=0.5, use_bn=False):
        super(Paraphraser, self).__init__()
        in_channel = t_shape[1]
        out_channel = int(t_shape[1] * k)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(out_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, f_s, is_factor=False):
        factor = self.encoder(f_s)
        if is_factor:
            return factor
        rec = self.decoder(factor)
        return factor, rec


class Translator(nn.Module):
    def __init__(self, s_shape, t_shape, k=0.5, use_bn=True):
        super(Translator, self).__init__()
        in_channel = s_shape[1]
        out_channel = int(t_shape[1] * k)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, f_s):
        return self.encoder(f_s)


class Connector(nn.Module):
    """Connect for Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons"""
    def __init__(self, s_shapes, t_shapes):
        super(Connector, self).__init__()
        self.s_shapes = s_shapes
        self.t_shapes = t_shapes

        self.connectors = nn.ModuleList(self._make_conenctors(s_shapes, t_shapes))

    @staticmethod
    def _make_conenctors(s_shapes, t_shapes):
        assert len(s_shapes) == len(t_shapes), 'unequal length of feat list'
        connectors = []
        for s, t in zip(s_shapes, t_shapes):
            if s[1] == t[1] and s[2] == t[2]:
                connectors.append(nn.Sequential())
            else:
                connectors.append(ConvReg(s, t, use_relu=False))
        return connectors

    def forward(self, g_s):
        out = []
        for i in range(len(g_s)):
            out.append(self.connectors[i](g_s[i]))

        return out


class ConnectorV2(nn.Module):
    """A Comprehensive Overhaul of Feature Distillation (ICCV 2019)"""
    def __init__(self, s_shapes, t_shapes):
        super(ConnectorV2, self).__init__()
        self.s_shapes = s_shapes
        self.t_shapes = t_shapes

        self.connectors = nn.ModuleList(self._make_conenctors(s_shapes, t_shapes))

    def _make_conenctors(self, s_shapes, t_shapes):
        assert len(s_shapes) == len(t_shapes), 'unequal length of feat list'
        t_channels = [t[1] for t in t_shapes]
        s_channels = [s[1] for s in s_shapes]
        connectors = nn.ModuleList([self._build_feature_connector(t, s)
                                    for t, s in zip(t_channels, s_channels)])
        return connectors

    @staticmethod
    def _build_feature_connector(t_channel, s_channel):
        C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
             nn.BatchNorm2d(t_channel)]
        for m in C:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return nn.Sequential(*C)

    def forward(self, g_s):
        out = []
        for i in range(len(g_s)):
            out.append(self.connectors[i](g_s[i]))

        return out


class ConvReg(nn.Module):
    """Convolutional regression for FitNet"""
    def __init__(self, s_shape, t_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1+s_H-t_H, 1+s_W-t_W))
        else:
            raise NotImplemented('student size {}, teacher size {}'.format(s_H, t_H))
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)


class Regress(nn.Module):
    """Simple Linear Regression for hints"""
    def __init__(self, dim_in=1024, dim_out=1024):
        super(Regress, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.relu(x)
        return x


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class LinearEmbed(nn.Module):
    """Linear Embedding"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(LinearEmbed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x


class MLPEmbed(nn.Module):
    """non-linear embed by MLP"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(MLPEmbed, self).__init__()
        self.linear1 = nn.Linear(dim_in, 2 * dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(2 * dim_out, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.l2norm(self.linear2(x))
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Flatten(nn.Module):
    """flatten module"""
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)


class PoolEmbed(nn.Module):
    """pool and embed"""
    def __init__(self, layer=0, dim_out=128, pool_type='avg'):
        super().__init__()
        if layer == 0:
            pool_size = 8
            nChannels = 16
        elif layer == 1:
            pool_size = 8
            nChannels = 16
        elif layer == 2:
            pool_size = 6
            nChannels = 32
        elif layer == 3:
            pool_size = 4
            nChannels = 64
        elif layer == 4:
            pool_size = 1
            nChannels = 64
        else:
            raise NotImplementedError('layer not supported: {}'.format(layer))

        self.embed = nn.Sequential()
        if layer <= 3:
            if pool_type == 'max':
                self.embed.add_module('MaxPool', nn.AdaptiveMaxPool2d((pool_size, pool_size)))
            elif pool_type == 'avg':
                self.embed.add_module('AvgPool', nn.AdaptiveAvgPool2d((pool_size, pool_size)))

        self.embed.add_module('Flatten', Flatten())
        self.embed.add_module('Linear', nn.Linear(nChannels*pool_size*pool_size, dim_out))
        self.embed.add_module('Normalize', Normalize(2))

    def forward(self, x):
        return self.embed(x)


class PointWiseConv(nn.Module):
    def __init__(self):
        super(PointWiseConv, self).__init__()
        self.conv = nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.norm = Normalize(2)

    def forward(self, x):
        x = self.conv(x)
        return self.norm(self.relu(x))




normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

class PCALighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img
        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()
        return img.add(rgb.view(3, 1, 1).expand_as(img))


imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}

def strong_train_preprocess(img_size):
    trans = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
        transforms.ToTensor(),
        PCALighting(0.1, imagenet_pca['eigval'], imagenet_pca['eigvec']),
        normalize,
    ])
    print('---------------------- strong dataaug!')
    return trans

def standard_train_preprocess(img_size):
    trans = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    print('---------------------- weak dataaug!')
    return trans

def val_preprocess(img_size):
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])
    return trans

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def read_hdf5(file_path):
    import h5py
    import numpy as np
    result = {}
    with h5py.File(file_path, 'r') as f:
        for k in f.keys():
            value = np.asarray(f[k])
            result[str(k).replace('+', '/')] = value
    print('read {} arrays from {}'.format(len(result), file_path))
    f.close()
    return result

def model_load_hdf5(model:torch.nn.Module, hdf5_path, ignore_keys='stage0.'):
    weights_dict = read_hdf5(hdf5_path)
    for name, param in model.named_parameters():
        print('load param: ', name, param.size())
        if name in weights_dict:
            np_value = weights_dict[name]
        else:
            np_value = weights_dict[name.replace(ignore_keys, '')]
        value = torch.from_numpy(np_value).float()
        assert tuple(value.size()) == tuple(param.size())
        param.data = value
    for name, param in model.named_buffers():
        print('load buffer: ', name, param.size())
        if name in weights_dict:
            np_value = weights_dict[name]
        else:
            np_value = weights_dict[name.replace(ignore_keys, '')]
        value = torch.from_numpy(np_value).float()
        assert tuple(value.size()) == tuple(param.size())
        param.data = value

if __name__ == '__main__':
    import torch

    g_s = [
        torch.randn(2, 16, 16, 16),
        torch.randn(2, 32, 8, 8),
        torch.randn(2, 64, 4, 4),
    ]
    g_t = [
        torch.randn(2, 32, 16, 16),
        torch.randn(2, 64, 8, 8),
        torch.randn(2, 128, 4, 4),
    ]
    s_shapes = [s.shape for s in g_s]
    t_shapes = [t.shape for t in g_t]

    net = ConnectorV2(s_shapes, t_shapes)
    out = net(g_s)
    for f in out:
        print(f.shape)
