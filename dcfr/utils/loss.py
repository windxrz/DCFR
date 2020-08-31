import torch
import torch.nn.functional as F


def weighted_mae(w, y, y_pred):
    return torch.sum(w * torch.abs(y - y_pred))


def weighted_smoothmae(w, y, y_pred):
    return torch.sum(w * F.smooth_l1_loss(y, y_pred, reduce=None))


def weighted_mse(w, y, y_pred):
    return torch.sum(w * (y - y_pred) * (y - y_pred))


def weighted_cross_entropy(w, y, y_pred, eps=1e-8):
    res = -torch.sum(
        w * (y * torch.log(y_pred + eps) + (1 - y) * torch.log(1 - y_pred + eps))
    )
    return res
