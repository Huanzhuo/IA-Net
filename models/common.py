import torch
import torch.nn as nn
from models.utils import *


class Conv(nn.Module):
    # Basic Conv block: conv + bn + ac_func
    def __init__(self, ch_in, ch_out, k, s=1, p=0, d=1, ac_func='PRelu'):
        super(Conv, self).__init__()
        self.conv = nn.Conv1d(ch_in, ch_out, k, s, padding=p, dilation=d, bias=False)
        self.bn = nn.GroupNorm(32, ch_out, eps=1e-08)
        if ac_func == 'PRelu':
            self.ac_func = nn.PReLU()
        elif ac_func == 'Mish':
            self.ac_func = Mish()
        else:
            self.ac_func = nn.Identity()

    def forward(self, x):
        return self.ac_func(self.bn(self.conv(x)))


class Conv1(nn.Module):
    # Basic Conv block: conv + ac_func + bn
    def __init__(self, ch_in, ch_out, k, s=1, p=0, d=1, ac_func='PRelu'):
        super(Conv1, self).__init__()
        self.conv = nn.Conv1d(ch_in, ch_out, k, s, padding=p, dilation=d, bias=False)
        self.bn = nn.GroupNorm(32, ch_out, eps=1e-08)
        if ac_func == 'PRelu':
            self.ac_func = nn.PReLU()
        elif ac_func == 'Mish':
            self.ac_func = Mish()
        else:
            self.ac_func = nn.Identity()

    def forward(self, x):
        return self.bn(self.ac_func(self.conv(x)))


class TCNBlock(nn.Module):
    def __init__(self, ch_in, ch_mid, ch_out, k, p, d, skip=True, ac_func='PRelu', freq_att=False):
        super(TCNBlock, self).__init__()
        self.skip = skip
        self.freq_att = freq_att
        conv1 = Conv1(ch_in, ch_mid, 1)
        depthwise_conv = nn.Conv1d(ch_mid, ch_mid, k, stride=1, padding=p, dilation=d, groups=ch_mid, bias=False)
        norm = nn.GroupNorm(32, ch_mid)
        if ac_func == 'PRelu':
            ac_func = nn.PReLU()
        elif ac_func == 'Mish':
            ac_func = Mish()
        else:
            ac_func = nn.ReLU()
        # [M, H, K] -> [M, B, K]
        self.pointwise_conv = nn.Conv1d(ch_mid, ch_out, 1, bias=False)
        # Put together
        self.block = nn.Sequential(conv1, depthwise_conv, ac_func, norm)
        if skip:
            self.skip_conv = nn.Conv1d(ch_mid, ch_in, 1, bias=False)
        if freq_att:
            self.att = FrequencyAttention(ch_out, ch_out, ch_out)

    def forward(self, x):
        residual = x
        output = self.block(x)
        if self.skip:
            skip = self.skip_conv(output)
            output = self.att(self.pointwise_conv(output))
            if self.freq_att:
                output = self.att(output)
            output = output + residual
            return [output, skip]
        else:
            output = self.pointwise_conv(output)
            if self.freq_att:
                output = self.att(output)
            return output + residual


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, ch_in, ch_out, shortcut=True, i=1, hidden_ratio=0.5):
        super(Bottleneck, self).__init__()
        ch_mid = int(ch_out * hidden_ratio)
        d = 2 ** i
        p = d
        self.conv1 = Conv(ch_in, ch_mid, 1, 1)
        self.conv2 = Conv(ch_mid, ch_out, 3, 1, p, d)
        self.add = shortcut and ch_in == ch_out

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck
    def __init__(self, ch_in, ch_out, n=1, shortcut=True, hidden_ratio=0.5, ac_func='PRelu'):
        super(BottleneckCSP, self).__init__()
        ch_ = int(ch_in * hidden_ratio)  # hidden channels
        self.conv1 = Conv(ch_in, ch_, 1, 1)
        self.conv2 = nn.Conv1d(ch_in, ch_, 1, 1, bias=False)
        self.conv3 = nn.Conv1d(ch_, ch_, 1, 1, bias=False)
        self.conv4 = Conv(ch_out, ch_out, 1, 1)
        self.bn = nn.GroupNorm(32, 2 * ch_, eps=1e-08)
        if ac_func == 'PRelu':
            self.ac_func = nn.PReLU()
        elif ac_func == 'Mish':
            self.ac_func = Mish()
        else:
            self.ac_func = nn.ReLU()
        self.m = nn.Sequential(*[Bottleneck(ch_, ch_, shortcut, i, hidden_ratio=1.0) for i in range(n)])

    def forward(self, x):
        y1 = self.conv3(self.m(self.conv1(x)))
        y2 = self.conv2(x)
        return self.conv4(self.ac_func(self.bn(torch.cat([y1, y2], dim=1))))


class ChannelBlock(nn.Module):
    def __init__(self, in_chs):
        super(ChannelBlock, self).__init__()
        expansion = 4
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        chs_mid = in_chs // expansion
        self.channel_excitation = nn.Sequential(nn.Linear(in_chs, chs_mid, bias=False),
                                nn.PReLU(),
                                nn.Linear(chs_mid, in_chs, bias=False),
                                nn.Sigmoid())

    def forward(self, x):
        num_batch, chs, _ = x.size()
        chn_se = self.avg_pool(x).view(num_batch, chs)
        chn_se = self.channel_excitation(chn_se).view(num_batch, chs, 1)
        return torch.mul(x, chn_se)


class LayerFusion(nn.Module):
    def __init__(self, ch_inter, ch_intra, k, s=4, p=True):
        super(LayerFusion, self).__init__()
        # p = (k - 1)  // 2
        if not p:
            p = (k - 1) // 2
        else:
            p = 0
        self.se_block = ChannelBlock(ch_intra)
        self.conv1 = Conv(ch_intra, ch_intra, k=k, s=s, p=p)
        self.conv2 = Conv(ch_inter+ch_intra, ch_inter, 3, p=1)

    def forward(self, x_inter, x_intra):
        x_add = self.conv1(x_intra)
        x_add = self.se_block(x_add)
        x_inter = self.conv2(torch.cat((x_inter, x_add), 1))
        return [x_inter, x_intra]


class BottleneckTCN(nn.Module):
    def __init__(self, ch_in, ch_mid, ch_out, n=1, ac_func='PRelu', skip=True, freq_att=False):
        super(BottleneckTCN, self).__init__()
        self.skip = skip
        tcn_blocks = []
        for i in range(n):
            d = 2 ** i
            p = d
            tcn_blocks += [TCNBlock(ch_in, ch_mid, ch_out, k=3, p=p, d=d, skip=skip, ac_func=ac_func, freq_att=freq_att)]
        self.tcn_blocks = nn.Sequential(*tcn_blocks)

    def forward(self, x):
        if self.skip:
            skip_connect = 0
            for m in self.tcn_blocks:
                x, skip = m(x)
                skip_connect += skip
            return x, skip_connect
        else:
            for m in self.tcn_blocks:
                x = m(x)
            return x


class Conv1d(nn.Module):
    # Basic Conv block: conv + bn + ac_func
    def __init__(self, ch_in, ch_out, k, s=1, p=0, d=1, ac_func='PRelu'):
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
        expansion = 4
        chs_mid = chs_in // expansion
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.se = nn.Sequential(nn.Linear(int(chs_in), chs_mid, bias=False),
                                nn.PReLU(),
                                nn.Linear(chs_mid, chs_out, bias=False),
                                nn.Sigmoid())

    def forward(self, x):
        batch_size, f, t = x.shape
        se = self.avg(x).view(batch_size, f)
        se = self.se(se).view(batch_size, f, 1)
        return x * se.expand_as(x)


class FrequencyAttention_(nn.Module):
    def __init__(self, chs_in, chs_out, size):
        super(FrequencyAttention_, self).__init__()
        expansion = 4
        chs_mid = chs_in // expansion
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.se = nn.Sequential(nn.Linear(int(chs_in), chs_mid, bias=False),
                                nn.PReLU(),
                                nn.Linear(chs_mid, chs_out, bias=False),
                                nn.Sigmoid())

    def forward(self, x):
        batch_size, f, t = x.shape
        se = self.avg(x).view(batch_size, f)
        se = self.se(se).view(batch_size, f, 1)
        return x * se.expand_as(x), se


if __name__ == '__main__':
    a = BottleneckCSP(256, 256, 8)
    print(a)
    b = torch.zeros([1, 256, 16000])
    c = a(b)
    print(c.size())