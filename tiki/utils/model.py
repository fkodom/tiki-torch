"""
model.py
--------
A set of helper functions/classes for converting`torch` models to/from `tiki`.
"""

from torch import Tensor
from torch.nn import Module
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


def from_module(module: Module):
    """TODO: Documentation"""
    net = FromModule()
    net._modules = module._modules
    net.forward = module.forward

    return net
