"""
loss.py
-------
A set of pre-compiled loss functions for training neural networks.
"""

from typing import Callable

import torch.nn as nn


__author__ = "Frank Odom"
__company__ = "Radiance Technologies, Inc."
__email__ = "frank.odom@radiancetech.com"
__classification__ = "UNCLASSIFIED"
__all__ = ["get_loss"]


loss_dict = {
    "bce": nn.BCELoss,
    "bce_with_logits": nn.BCEWithLogitsLoss,
    "cosine": nn.CosineEmbeddingLoss,
    "ctc": nn.CTCLoss,
    "cross_entropy": nn.CrossEntropyLoss,
    "hinge": nn.HingeEmbeddingLoss,
    "kl_div": nn.KLDivLoss,
    "l1": nn.L1Loss,
    "margin": nn.MarginRankingLoss,
    "mse": nn.MSELoss,
    "multi_label_margin": nn.MultiLabelMarginLoss,
    "multi_label_soft_margin": nn.MultiLabelSoftMarginLoss,
    "multi_margin": nn.MultiMarginLoss,
    "nll": nn.NLLLoss,
    "poisson_nll": nn.PoissonNLLLoss,
    "smooth_l1": nn.SmoothL1Loss,
    "soft_margin": nn.SoftMarginLoss,
    "triplet_margin": nn.TripletMarginLoss,
}


def get_loss(loss: str or Callable) -> Callable:
    """TODO: Documentation"""
    if isinstance(loss, str):
        if loss not in loss_dict.keys():
            raise ValueError(
                f"Loss function '{loss}' not recognized.  "
                f"Allowed values: {list(loss_dict.keys())}"
            )
        else:
            loss = loss_dict[loss]()

    return loss
