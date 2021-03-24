import math
import torch
import numpy as np
import torch.nn as nn
import torchvision.models.resnet
from models.utils import *
from torchvision.models.resnet import resnet50


class Conv(nn.Module):
    # Basic Conv block: conv + bn + ac_func
    def __init__(self, ch_in, ch_out, k, s=1, p=0, d=1, ac_func='Relu'):
        super(Conv, self).__init__()
        self.conv = nn.Conv1d(ch_in, ch_out, k, s, padding=p, dilation=d, bias=False)
        self.bn = nn.BatchNorm1d(ch_out)
        if ac_func == 'PRelu':
            self.ac_func = nn.PReLU()
        elif ac_func == 'Mish':
            self.ac_func = Mish()
        elif ac_func == 'LeakyRelu':
            self.ac_func = nn.LeakyReLU()
        elif ac_func == 'Relu':
            self.ac_func = nn.ReLU()
        else:
            self.ac_func = nn.Identity()

    def forward(self, x):
        return self.ac_func(self.bn(self.conv(x)))


class Conv_1(nn.Module):
    # Basic Conv block: conv + bn
    def __init__(self, ch_in, ch_out, k, s=1, p=0, d=1):
        super(Conv_1, self).__init__()
        self.conv = nn.Conv1d(ch_in, ch_out, k, s, padding=p, dilation=d, bias=False)
        self.bn = nn.BatchNorm1d(ch_out)

    def forward(self, x):
        return self.bn(self.conv(x))


class Conv1d(nn.Module):
    # Basic Conv block: conv + bn + ac_func
    def __init__(self, ch_in, ch_out, k, s=1, p=0, d=1, ac_func='Relu'):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(ch_in, ch_out, k, s, padding=p, dilation=d, bias=False)
        self.bn = nn.BatchNorm1d(ch_out)
        if ac_func == 'PRelu':
            self.ac_func = nn.PReLU()
        elif ac_func == 'Mish':
            self.ac_func = Mish()
        elif ac_func == 'LeakyRelu':
            self.ac_func = nn.LeakyReLU()
        elif ac_func == 'Relu':
            self.ac_func = nn.ReLU()
        else:
            self.ac_func = nn.Identity()

    def forward(self, x):
        return self.ac_func(self.bn(self.conv(x)))


class Conv_1d_1(nn.Module):
    # Basic Conv block: conv + bn
    def __init__(self, ch_in, ch_out, k, s=1, p=0, d=1):
        super(Conv_1d_1, self).__init__()
        self.conv = nn.Conv1d(ch_in, ch_out, k, s, padding=p, dilation=d, bias=False)
        self.bn = nn.BatchNorm1d(ch_out)

    def forward(self, x):
        return self.bn(self.conv(x))


class FrequencyAttention(nn.Module):
    def __init__(self, chs_in, chs_out, size):
        super(FrequencyAttention, self).__init__()
        expansion = 16
        chs_mid = chs_in // expansion
        f = size
        target_size = (f, 1)
        self.avg = nn.AdaptiveAvgPool2d(target_size)
        self.se = nn.Sequential(Conv1d(chs_in, chs_mid, 1),
                                Conv1d(chs_mid, chs_mid, 3, p=1),
                                Conv_1d_1(chs_mid, chs_out, 1),
                                nn.Sigmoid())

    def forward(self, x):
        batch_size, chs, f, t = x.shape
        se = self.avg(x).view(batch_size, chs, f)
        se = self.se(se).view(batch_size, chs, f, 1)
        return x * se.expand_as(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, ac_func='Relu'):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv(inplanes, planes, k=1, s=stride)
        self.conv2 = Conv(planes, planes, k=3, p=dilation, s=1, d=dilation)
        self.conv3 = Conv_1(planes, planes * self.expansion, k=1)

        self.downsample = downsample
        self.stride = stride
        if ac_func == 'PRelu':
            self.ac_func = nn.PReLU()
        elif ac_func == 'Mish':
            self.ac_func = Mish()
        elif ac_func == 'LeakyRelu':
            self.ac_func = nn.LeakyReLU()
        elif ac_func == 'Relu':
            self.ac_func = nn.ReLU()
        else:
            self.ac_func = nn.Identity()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.ac_func(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, ac_func='PRelu'):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv(inplanes, planes, k=3, s=stride, p=1)
        self.conv2 = Conv_1(planes, planes, k=3, s=1, p=1)

        self.downsample = downsample
        self.stride = stride
        if ac_func == 'PRelu':
            self.ac_func = nn.PReLU()
        elif ac_func == 'Mish':
            self.ac_func = Mish()
        elif ac_func == 'LeakyRelu':
            self.ac_func = nn.LeakyReLU()
        elif ac_func == 'Relu':
            self.ac_func = nn.ReLU()
        else:
            self.ac_func = nn.Identity()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.ac_func(out)

        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = Conv(1, 32, 7, 2, 3)  # downsample x2
        self.conv2 = Conv(32, 64, 3, 2, 1)  # downsample x4
        self.conv3 = Conv(64, 128, 3, 2, 1)  # downsample x8
        self.conv4 = Conv(128, 256, 3, 2, 1)  # downsample x16
        self.shortcut = Conv(32, 256, 15, 8, 7)

    def forward(self, x):
        x = self.conv1(x)
        shortcut = self.shortcut(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x + shortcut
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, n_spk=4, num_emed=256):
        self.inplanes = 256
        super(ResNet, self).__init__()
        self.n_spk = n_spk
        self.n_emed = num_emed
        self.encoder = Encoder()
        block.expansion = 1
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        block.expansion = 4
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.conv1 = nn.Conv1d(512 * block.expansion, 512 * block.expansion, 1, 1, 0)
        # self.conv2 = nn.Conv1d(512 * block.expansion, 512 * block.expansion, 1, 1, 0)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc= nn.Linear(512 * block.expansion, num_emed*n_spk)
        # self.fc2 = nn.Linear(512 * block.expansion, num_emed)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(planes * block.expansion),
        )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            d = 2 ** i
            layers.append(block(self.inplanes, planes, dilation=d))

        return nn.Sequential(*layers)

    def forward_freeze(self, x):
        x = self.encoder(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).squeeze(-1)
        x = self.fc(x)
        return x

    def forward_once(self, x):
        x = self.encoder(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).squeeze(-1)
        x = self.fc(x)
        features = []
        start = 0
        for i in range(self.n_spk):
            f = x[:, start:start+self.n_emed]
            start += self.n_emed
            features.append(f)
        return features

    def forward(self, mix_0, mix_1):
        features = self.forward_once(mix_0)
        features_ = self.forward_once(mix_1)
        return features, features_


def resnet50_1d(n_spk, num_emed=256):
    model = ResNet(Bottleneck, [3, 4, 6, 3], n_spk=n_spk, num_emed=num_emed)
    return model


def resnet50_2d():
    model = resnet50()
    model.fc = nn.Sequential(nn.Linear(2048, 1024),
                             nn.PReLU(),
                             nn.Linear(1024, 256),
                             nn.PReLU(),
                                nn.Linear(256, 2)
                                )


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        backbone = resnet50()
        conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(conv1,backbone.bn1,backbone.maxpool,backbone.layer1,
                                      backbone.layer2,backbone.layer3,backbone.layer4)
        self.conv1 = Conv(2048, 2048, 3, 1, 1)
        self.conv2 = Conv(2048, 2048, 3, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(2048, 512)

    def forward_once(self, x):
        features = self.backbone(x)
        f1 = self.conv1(features)
        f2 = self.conv2(features)
        f1 = self.avgpool(f1)
        f2 = self.avgpool(f2)
        f1 = torch.flatten(f1, 1)
        f2 = torch.flatten(f2, 1)
        f1 = self.fc1(f1)
        f2 = self.fc2(f2)
        return [f1, f2]

    def forward(self, mix_0, mix_1):
        f1, f2 = self.forward_once(mix_0)
        f1_, f2_ = self.forward_once(mix_1)
        return [f1, f1_], [f2, f2_]


class ModelFreeze(nn.Module):
    def __init__(self, model_path, n_spk=4, num_emed=256):
        super(ModelFreeze, self).__init__()
        print("Loading backbone from the path {}".format(model_path))
        self.n_spk = n_spk
        self.num_emed = num_emed
        self.backbone = torch.load(model_path)['model']
        self.vis_layer_0 = nn.Linear(n_spk*num_emed, 256)
        self.vis_layer_1 = nn.Linear(256, 32)
        self.vis_layer_2 = nn.Linear(32, 2*num_emed)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backnone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward_once(self, x):
        x = self.backbone.forward_freeze(x)
        x = self.vis_layer_0(x)
        x = self.vis_layer_1(x)
        x = self.vis_layer_2(x)
        features = []
        start = 0
        for i in range(self.n_spk):
            f = x[:, start:start + 2]
            start += 2
            features.append(f)
        return features

    def forward(self, mix_0, mix_1):
        features = self.forward_once(mix_0)
        features_ = self.forward_once(mix_1)
        return features, features_


if __name__ == '__main__':
    model = Model()
    input = torch.zeros([1, 1, 512, 256])
    input_ = torch.zeros([1, 1, 512, 256])
    output = model(input, input_)
    print(model)
    print('test')
