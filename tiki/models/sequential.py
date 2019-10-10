"""
sequential.py
-------------
A simplified interface for building basic neural network models.
"""

from torch import Tensor
import torch.nn as nn
from tiki.models import Base


__author__ = "Frank Odom"
__company__ = "Radiance Technologies, Inc."
__email__ = "frank.odom@radiancetech.com"
__classification__ = "UNCLASSIFIED"
__all__ = ["Sequential"]


class Sequential(Base):
    """Simplified neural network model, which automatically defines the
    'forward' method for end users.  Allows users to define simple networks
    without needing to explicitly define a base class for the model.

    Examples
    --------
    >>> import torch
    >>> from tiki.models import Sequential
    >>> net = Sequential(torch.nn.Linear(10, 5))
    >>> hasattr(net, "forward") and hasattr(net, "fit")
    True
    >>> net(torch.rand(25, 10)).size()
    torch.Size([25, 5])
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def add_module(self, name: str, module: nn.Module):
        """Adds a new named module at the end of the current Sequential model.

        Parameters
        ----------
        name: str
            Name for the new module
        module: nn.Module
            Module to add to the Sequential model

        Examples
        --------
        >>> import torch.nn as nn
        >>> from tiki.models import Sequential
        >>> net = Sequential()
        >>> net.add_module("linear_1", nn.Linear(10, 5, bias=False))
        >>> len(net.state_dict())
        1
        """
        self.layers.add_module(name, module)

    def forward(self, x: Tensor) -> Tensor:
        """Pushes inputs 'x' through the Sequential model."""
        if self.device.type == "cuda":
            x = x.to(self.device)

        return self.layers(x)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
