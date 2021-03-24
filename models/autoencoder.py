import torch
import torch.nn as nn
from configs import cfg
from models.common import *


class Dense(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Dense, self).__init__()
        self.fc = nn.Linear(ch_in, ch_out, bias=False)
        self.bn = nn.BatchNorm1d(122)
        self.ac_func = nn.ReLU()

    def forward(self, x):
        return self.ac_func(self.bn(self.fc(x)))


class AutoEncoder(nn.Module):
    def __init__(self, ch_in=640):
        super(AutoEncoder, self).__init__()
        self.encoder_1 = Dense(ch_in, 128)
        self.encoder_2 = Dense(128, 128)
        self.encoder_3 = Dense(128, 128)
        self.encoder_4 = Dense(128, 128)
        self.bottlenect = Dense(128, 8)
        self.decoder_1 = Dense(8, 128)
        self.decoder_2 = Dense(128, 128)
        self.decoder_3 = Dense(128, 128)
        self.decoder_4 = Dense(128, ch_in)

    def forward(self, x):
        x = self.encoder_1(x)
        x = self.encoder_2(x)
        x = self.encoder_3(x)
        x = self.encoder_4(x)
        x = self.bottlenect(x)
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        x = self.decoder_4(x)
        return x

