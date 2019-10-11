"""
model.py
--------
A set of helper functions/classes for converting`torch` models to/from `tiki`.
"""

from torch import Tensor
import torch.nn as nn
from tiki.models import Base


__author__ = "Frank Odom"
__company__ = "Radiance Technologies, Inc."
__email__ = "frank.odom@radiancetech.com"
__classification__ = "UNCLASSIFIED"
__all__ = ["from_module"]


class FromModule(Base):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


def from_module(module: nn.Module) -> Base:
    """Converts any torch.nn.Module into a tiki.models.Base object, which
    provides convenient methods for model training.

    Parameters
    ----------
    module: nn.Module
        PyTorch module to convert to tiki

    Returns
    -------
    Base
        Trainable tiki model
    """
    net = FromModule()
    net._modules = module._modules
    net.forward = module.forward

    return net
