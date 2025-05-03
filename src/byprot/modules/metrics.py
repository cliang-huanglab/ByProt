import numpy as np
import torch


def accuracy(pred, target, mask=None, reduction="all"):
    assert pred.shape == target.shape
    if mask is None:
        mask = torch.ones_like(pred, dtype=torch.bool)

    return (pred[mask] == target[mask]).sum() / mask.sum()


def accuracy_per_sample(pred, target, mask=None):
    assert pred.shape == target.shape
    bsz = target.shape[0]

    if mask is None:
        mask = torch.ones_like(pred, dtype=torch.bool)

    pred = pred.view(bsz, -1)
    target = target.view(bsz, -1)
    mask = mask.view(bsz, -1)

    return ((pred == target) * mask).sum(1) / mask.sum(1)


from tmtools import tm_align


def calc_tm_score(pos_1, pos_2, seq_1, seq_2):
    tm_results = tm_align(np.float64(pos_1), np.float64(pos_2), seq_1, seq_2)
    return tm_results.tm_norm_chain1, tm_results.tm_norm_chain2
