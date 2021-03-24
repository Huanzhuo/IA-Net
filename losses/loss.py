import torch
import torch.nn as nn
import torch.nn.functional as F


class CosLoss(nn.Module):
    def __init__(self):
        super(CosLoss, self).__init__()
        self.loss = nn.CosineEmbeddingLoss()

    def forward(self, s1_features, s2_features, s1_labels, s2_labels):
        s1_loss = self.loss(s1_features[0], s1_features[1], s1_labels)
        s2_loss = self.loss(s2_features[0], s2_features[1], s2_labels)
        loss = s1_loss + s2_loss
        return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin=3):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-8

    def forward_once(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.squeeze() * distances +
                        (1 + -1 * target.squeeze()) * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

    def forward(self, s1_features, s2_features, s1_labels, s2_labels):
        s1_loss = self.forward_once(s1_features[0], s1_features[1], s1_labels)
        s2_loss = self.forward_once(s2_features[0], s2_features[1], s2_labels)
        return s1_loss + s2_loss


class ContrastiveLoss_(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin=3):
        super(ContrastiveLoss_, self).__init__()
        self.margin = margin
        self.eps = 1e-8

    def forward_once(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.squeeze() * distances +
                        (1 + -1 * target.squeeze()) * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

    def forward(self, features, features_, status):
        total_loss = 0.0
        n = status.shape[1]
        for i in range(n):
            target = status[:, i]
            loss = self.forward_once(features[i], features_[i], target)
            total_loss += loss
        return total_loss


def sdr_loss(estimation, origin, mask=None):
    sdr = sdr_object(estimation, origin, mask)
    loss = -sdr.sum()
    return loss


def sdr_object(estimation, origin, masks):
    """
    batch-wise SDR caculation for one audio file.
    estimation: (batch, chs, nsample)
    origin: (batch, chs, nsample)
    mask: optional, (batch, chs), binary  True if audio is silence
    """
    estimation = estimation.unsqueeze(2)  # (batch, chs, 1, nsample)
    origin = origin.unsqueeze(2)
    origin_power = torch.pow(origin, 2).sum(dim=-1, keepdim=True) + 1e-8  # shape: (B, 4, 1, 1)
    scale = torch.sum(origin * estimation, dim=-1, keepdim=True) / origin_power  # shape: (B, 4, 1, 1)

    est_true = scale * origin
    est_res = estimation - est_true

    true_power = torch.pow(est_true, 2).sum(dim=-1).clamp(min=1e-8)
    res_power = torch.pow(est_res, 2).sum(dim=-1).clamp(min=1e-8)

    sdr = 10 * (torch.log10(true_power) - torch.log10(res_power))

    if masks is not None:
        masks = masks.unsqueeze(2)
        sdr = (sdr * masks).sum(dim=(0, -1)) / masks.sum(dim=(0, -1)).clamp(min=1e-8)  # shape: (4)
        # masks = masks.unsqueeze(2).unsqueeze(3)
        # sdr = (sdr * masks).sum(dim=(0, -1)) / masks.sum(dim=(0, -1)).clamp(min=1e-8)  # shape: (4)
    else:
        sdr = sdr.mean(dim=(0, -1))  # shape: (4)

    return sdr

