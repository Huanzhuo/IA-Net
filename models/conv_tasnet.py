import torch
import torch.nn as nn
from configs import cfg
from models.common import *
from contextlib import redirect_stdout


class Encoder(nn.Module):
    def __init__(self, cfg):
        '''
        Encoder part for Y-Net
        Args:
            ch_in: input audio channels, default 1 for mono signal
            ch1: output channels for inter-channel path
            ch2: output channels for intra-channel path
            k1: kernel size for inter-channel encoder
            k2: kernel size for intra-channel encoder
        Inputs:
            padded signal [batch_size, ch_in, length of signal]

        Return:
            encoded inter-channel and intra-channel features
            [y_inter, y_intra]
        '''
        ch_in, ch = cfg.PARAMETERS.CHS_IN, cfg.ENCODER.C
        k = cfg.ENCODER.K
        super(Encoder, self).__init__()
        self.norm = cfg.ENCODER.NORM
        if cfg.PARAMETERS.PADDING:
            pad = 0
        else:
            pad = (k - 1)//2
        s = cfg.ENCODER.S
        self.encoder = nn.Conv1d(ch_in, ch, k, s, padding=pad, bias=False)
        if self.norm:
            self.norm = nn.GroupNorm(1, ch, eps=1e-8)

    def forward(self, x):
        y = self.encoder(x)
        if self.norm:
            y = self.norm(y)
        return y


class CSPSeparationBlock(nn.Module):
    '''
    Separation repeat for separation part using CSP-Net
    Args:
        ch_inter: output channels for inter-channel path
        ch_intra: output channels for intra-channel path
        x: number of bottleneck for BottleneckCSP block, default: 8
        k_trans: kernel size for layer fusion part
    Input:
        features from two paths or two encoder [x_inter, x_intra]
    Return:
        separated features from inter-channel and intra-channel path
        [y_inter, y_intra]
    '''

    def __init__(self, cfg, i):
        super(CSPSeparationBlock, self).__init__()
        ch_inter, ch_intra = cfg.ENCODER.C_INTER, cfg.ENCODER.C_INTRA
        x, k_trans = cfg.SEPARATION.X[i], cfg.SEPARATION.K_TRANS
        self.inter_block = BottleneckCSP(ch_inter, ch_inter, x)
        self.intra_block = BottleneckCSP(ch_intra, ch_intra, x)
        self.fuse = LayerFusion(ch_inter, ch_intra, k_trans, p=cfg.PARAMETERS.PADDING)

    def forward(self, x):
        x_inter, x_intra = x
        x_inter = self.inter_block(x_inter)
        x_intra = self.intra_block(x_intra)
        return self.fuse(x_inter, x_intra)


class TCNSeparationBlock(nn.Module):
    '''
    Separation repeat for separation part using bottleneck TCN
    Args:
        ch_inter: output channels for inter-channel path
        ch_intra: output channels for intra-channel path
        x: number of bottleneck for BottleneckCSP block, default: 8
        k_trans: kernel size for layer fusion part
    Input:
        features from two paths or two encoder [x_inter, x_intra]
    Return:
        separated features from inter-channel and intra-channel path
        [y_inter, y_intra]
    '''
    def __init__(self, cfg, i):
        super(TCNSeparationBlock, self).__init__()
        ch = cfg.ENCODER.C
        ch_mid = cfg.SEPARATION.C
        x = cfg.SEPARATION.X[i]
        self.skip = cfg.SEPARATION.SKIP
        self.freq_att = cfg.SEPARATION.FREQ_ATTENTION
        self.block = BottleneckTCN(ch, ch_mid, ch, x, skip=self.skip, freq_att=self.freq_att)

    def forward(self, x):
        if not self.skip:
            return self.block(x)
        else:
            x_mid, x_skip = x
            x_mid, skip = self.inter_block(x_mid)
            x_skip += skip
            return [x_mid, x_skip]


class Decoder(nn.Module):
    '''
    Decoder part using a ConvTranspose1d part
    Args:
        ch_in: input audio channels, default 1 for mono signal
        ch1: output channels for inter-channel path
        ch2: output channels for intra-channel path
        k1: kernel size for inter-channel encoder
        k2: kernel size for intra-channel encoder
        num_spk: number of target sources
    Return:
        separated signal using two decoder
        [inter_output, intra_output]
    '''

    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.ch = cfg.ENCODER.C
        k, s = cfg.ENCODER.K, cfg.ENCODER.S
        self.num_spk = cfg.PARAMETERS.NUM_SPK
        self.mask = nn.Sequential(nn.PReLU(),
                                  nn.Conv1d(self.ch, self.num_spk * self.ch, 1))

        if not cfg.PARAMETERS.PADDING:
            self.decoder = nn.Sequential(nn.Upsample(scale_factor=cfg.ENCODER.S),
                                         nn.Conv1d(self.ch, 1, 1, bias=False))
        else:
            self.decoder = nn.ConvTranspose1d(self.ch, 1, k, stride=s, bias=False)

    def forward(self, mixtures, masks):
        batch_size, _, _ = masks.size()
        mask = self.mask(mixtures)
        mask = torch.sigmoid(mask).view(batch_size, self.num_spk, self.ch, -1)
        outputs = mixtures.unsqueeze(1) * mask
        outputs = self.decoder(outputs.view(batch_size * self.num_spk, self.ch, -1))
        outputs = outputs.view(batch_size, self.num_spk, -1).clamp(-1, 1)
        return outputs


class CSPSeparationPart(nn.Module):
    '''
    dual path separation part using CSP-Net
    Args:
         n: number of separation block
    Input:
        features from two encoders
    Return:
        separated masks
        [mask_inter, mask_intra]
    '''

    def __init__(self, cfg):
        super(CSPSeparationPart, self).__init__()
        n = cfg.SEPARATION.N
        separation_blocks = []
        for i in range(n):
            separation_blocks += [CSPSeparationBlock(cfg, i)]
        self.separation_part = nn.Sequential(*separation_blocks)

    def forward(self, x):
        return self.separation_part(x)


class TCNSeparationPart(nn.Module):
    def __init__(self, cfg):
        super(TCNSeparationPart, self).__init__()
        self.skip = cfg.SEPARATION.SKIP
        n = cfg.SEPARATION.N
        separation_blocks = []
        for i in range(n):
            separation_blocks += [TCNSeparationBlock(cfg, i)]
        self.separation_part = nn.Sequential(*separation_blocks)

    def forward(self, x):
        if not self.skip:
            return self.separation_part(x)
        else:
            return self.separation_part([[x[0], torch.zeros_like(x[0]).to(x[0].device)], [x[1], torch.zeros_like(x[1]).to(x[1].device)]])


class Y_Net(nn.Module):
    '''
    Y-Net: dual path deep neural network for audio signal separation
    Args:
        config file
    Input:
        x: time domain mixed signal [batch_size, n_sample]
    Output:

    '''

    def __init__(self, cfg=cfg):
        super(Y_Net, self).__init__()
        self.clustering = cfg.PARAMETERS.CLUSTERING
        self.encoder = Encoder(cfg)
        if cfg.SEPARATION.BACKBONE == 'CSP-Net':
            self.separation = CSPSeparationPart(cfg)
        elif cfg.SEPARATION.BACKBONE == 'TCN':
            self.separation = TCNSeparationPart(cfg)
        else:
            raise NotImplementedError("the backbone is not included in the lib")
        self.decoder = Decoder(cfg)

    def forward(self, x):
        x, start, end = self._pad_signal(cfg, x)
        mixtures = self.encoder(x)
        masks = self.separation(mixtures)
        outputs = self.decoder(mixtures, masks)
        outputs = outputs[..., start:end]
        return outputs

    def _pad_signal(self, cfg, input):
        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nsample = input.size(2)
        if not cfg.PARAMETERS.PADDING:
            return input, 0, nsample
        k = cfg.ENCODER.K
        stride = k // 2
        rest = k - (stride + nsample % k) % k
        if rest > 0:
            pad = torch.zeros((batch_size, 1, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = torch.zeros((batch_size, 1, stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)
        start = stride
        end = stride + nsample
        return input, start, end


if __name__ == '__main__':
    model = Y_Net(cfg)
    print(cfg)
    print(model)
    input = torch.ones([4, 32000])
    output = model(input)
    print('test')
