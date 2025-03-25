import torch.nn.functional as F
import torch.nn as nn


def loss_fn_custom(pred, target, control):
    # L1 loss (primary term)
    l1_loss = F.l1_loss(pred, target)

    pred_dir = pred - control
    target_dir = target - control
    cos_loss = 1 - F.cosine_similarity(pred_dir, target_dir).mean()

    return l1_loss + 0.3 * cos_loss

def l2_loss(pred, target, control):
    rmse = nn.MSELoss()
    return rmse(pred, target)
