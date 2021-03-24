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
        ch_in, ch1, ch2 = cfg.PARAMETERS.CHS_IN, cfg.ENCODER.C_INTER, cfg.ENCODER.C_INTRA
        k1, k2 = cfg.ENCODER.K_INTER, cfg.ENCODER.K_INTRA
        super(Encoder, self).__init__()
        self.norm = cfg.ENCODER.NORM
        if cfg.PARAMETERS.PADDING:
            pad1 = 0
            pad2 = 0
        else:
            pad1 = (k1 - 1)//2
            pad2 = (k2 - 1)//2
        s1, s2 = cfg.ENCODER.S_INTER, cfg.ENCODER.S_INTRA
        self.encoder_inter = nn.Conv1d(ch_in, ch1, k1, s1, padding=pad1, bias=False)
        self.encoder_intra = nn.Conv1d(ch_in, ch2, k2, s2, padding=pad2, bias=False)
        if self.norm:
            self.norm_inter = nn.GroupNorm(1, ch1, eps=1e-8)
            self.norm_intra = nn.GroupNorm(1, ch2, eps=1e-8)

    def forward(self, x):
        y_inter = self.encoder_inter(x)
        y_intra = self.encoder_intra(x)
        if self.norm:
            y_inter = self.norm_inter(y_inter)
            y_intra = self.norm_intra(y_intra)
        return [y_inter, y_intra]


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
        ch_inter, ch_intra = cfg.ENCODER.C_INTER, cfg.ENCODER.C_INTRA
        ch_inter_mid = cfg.SEPARATION.C_INTER_MID
        ch_intra_mid = cfg.SEPARATION.C_INTRA_MID
        x, k_trans = cfg.SEPARATION.X[i], cfg.SEPARATION.K_TRANS
        self.skip = cfg.SEPARATION.SKIP
        self.freq_att = cfg.SEPARATION.FREQ_ATTENTION
        self.inter_block = BottleneckTCN(ch_inter, ch_inter_mid, ch_inter, x, skip=self.skip, freq_att=self.freq_att)
        self.intra_block = BottleneckTCN(ch_intra, ch_intra_mid, ch_intra, x, skip=self.skip, freq_att=False)
        self.final = (i == cfg.SEPARATION.N - 1)
        if not self.skip:
            self.fuse = LayerFusion(ch_inter, ch_intra, k_trans, p=cfg.PARAMETERS.PADDING)
        else:
            if self.final:
                self.fuse = nn.Identity()
            else:
                self.fuse = LayerFusion(ch_inter, ch_intra, k_trans, p=cfg.PARAMETERS.PADDING)

    def forward(self, x):
        if not self.skip:
            x_inter, x_intra = x
            x_inter = self.inter_block(x_inter)
            x_intra = self.intra_block(x_intra)
            return self.fuse(x_inter, x_intra)
        else:
            inter, intra = x
            x_inter, inter_skip = inter
            x_intra, intra_skip = intra

            x_inter, skip = self.inter_block(x_inter)
            inter_skip += skip
            x_intra, skip = self.intra_block(x_intra)
            intra_skip += skip
            if not self.final:
                x_inter, x_intra = self.fuse(x_inter, x_intra)
            else:
                x_inter, x_intra = self.fuse([x_inter, x_intra])
            if self.final:
                return [inter_skip, intra_skip]
            else:
                return [[x_inter, inter_skip], [x_intra, intra_skip]]


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
        self.ch1, self.ch2 = cfg.ENCODER.C_INTER, cfg.ENCODER.C_INTRA
        k1, k2 = cfg.ENCODER.K_INTER, cfg.ENCODER.K_INTRA
        s1, s2 = cfg.ENCODER.S_INTER, cfg.ENCODER.S_INTRA
        self.double_docder = cfg.DECODER.DOUBLE_DECODER
        # self.clustering = cfg.PARAMETERS.CLUSTERING
        self.num_spk = cfg.PARAMETERS.NUM_SPK
        self.mask_inter = nn.Sequential(nn.PReLU(),
                                        nn.Conv1d(self.ch1, self.num_spk * self.ch1, 1))
        # self.att = FrequencyAttention_(self.ch1, self.ch1, self.ch1)
        # self.conv = Conv(self.num_spk * self.ch1, self.num_spk * self.ch1, 3, 1, 1)
        # self.avg = nn.AdaptiveAvgPool1d(1)
        # self.linear = nn.Linear(self.ch1, 1)

        if not cfg.PARAMETERS.PADDING:
            self.decoder_inter = nn.Sequential(nn.Upsample(scale_factor=cfg.ENCODER.S_INTER),
                                               nn.Conv1d(self.ch1, 1, 1, bias=False))
        else:
            self.decoder_inter = nn.ConvTranspose1d(self.ch1, 1, k1, stride=s1, bias=False)

        if self.double_docder:
            self.mask_intra = nn.Sequential(nn.PReLU(),
                                            nn.Conv1d(self.ch2, self.num_spk * self.ch2, 1))
            if not cfg.PARAMETERS.PADDING:
                self.decoder_intra = nn.Sequential(nn.Upsample(scale_factor=cfg.ENCODER.S_INTRA),
                                                   nn.Conv1d(self.ch1, 1, 1, bias=False))
            else:
                self.decoder_intra = nn.ConvTranspose1d(self.ch2, 1, k2, stride=s2, bias=False)
        if self.clustering:
            self.trans = Conv(self.num_spk * self.ch1, self.num_spk * self.ch1, 3, 1, 1)
            self.max = nn.AdaptiveAvgPool2d((self.ch1, 1))
            # self.decoder_intra = nn.ConvTranspose1d(self.ch2, 1, k2, stride=k2 // 2, bias=False)

    def forward(self, mixtures, masks):
        mask_inter, mask_intra = masks
        batch_size, _, _ = masks[0].size()
        mixture_inter, mixture_intra = mixtures
        mask_inter = self.mask_inter(mask_inter)
        # features = self.conv(mask_inter)
        # features = features.view(batch_size * self.num_spk, self.ch1, -1)
        # features = self.avg(features).squeeze()
        # features = self.linear(features).view(batch_size, self.num_spk)
        # mask_inter, se = self.att(mask_inter.view(batch_size*self.num_spk, self.ch1, -1))
        # se = se.view(batch_size, self.num_spk, self.ch1)
        if self.clustering:
            clutering_features = self.trans(mask_inter).view(batch_size, self.num_spk, self.ch1, -1)
            # clutering_features = mask_inter.view(batch_size, self.num_spk, self.ch1, -1)
            clutering_features = self.max(clutering_features).squeeze()
        mask_inter = torch.sigmoid(mask_inter).view(batch_size, self.num_spk, self.ch1, -1)
        inter_output = mixture_inter.unsqueeze(1) * mask_inter
        inter_output = self.decoder_inter(inter_output.view(batch_size * self.num_spk, self.ch1, -1))
        inter_output = inter_output.view(batch_size, self.num_spk, -1).clamp(-1, 1)

        if self.double_docder:
            mask_intra = torch.sigmoid(self.mask_intra(mask_intra)).view(batch_size, self.num_spk, self.ch2, -1)
            intra_output = mixture_intra.unsqueeze(1) * mask_intra
            intra_output = self.decoder_intra(intra_output.view(batch_size * self.num_spk, self.ch2, -1))
            intra_output = intra_output.view(batch_size, self.num_spk, -1).clamp(-1, 1)
            return [inter_output, intra_output, clutering_features]
        return [inter_output, clutering_features]
        #     if self.clustering:
        #         return [inter_output, intra_output, clutering_features]
        #     else:
        #         return [inter_output, intra_output, mask_inter]
        # else:
        #     if self.clustering:
        #         return [inter_output, clutering_features]
        #     else:
        #         return inter_output


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
        if isinstance(outputs, list):
            if len(outputs) == 3:
                outputs[0] = outputs[0][..., start:end]
                outputs[1] = outputs[1][..., start:end]
            elif len(outputs) == 2 and self.clustering:
                outputs[0] = outputs[0][..., start:end]
            elif len(outputs) == 2 and not self.clustering:
                outputs[0] = outputs[0][..., start:end]
                outputs[1] = outputs[1][..., start:end]
            else:
                outputs = outputs[..., start:end]
        elif isinstance(outputs, torch.Tensor):
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
        k = cfg.ENCODER.K_INTER
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
