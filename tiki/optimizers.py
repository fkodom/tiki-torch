"""
A set of pre-compiled optimizers for training neural networks.
"""

from typing import Iterator

from torch.nn import Parameter
import torch.optim as optim


__author__ = "Frank Odom"
__company__ = "Radiance Technologies, Inc."
__email__ = "frank.odom@radiancetech.com"
__classification__ = "UNCLASSIFIED"
__all__ = ["get_optimizer"]


optimizer_dict = {
    "adadelta": optim.Adadelta,
    "adagrad": optim.Adagrad,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "adamax": optim.Adamax,
    "asgd": optim.ASGD,
    "lbfgs": optim.LBFGS,
    "rmsprop": optim.RMSprop,
    "rprop": optim.Rprop,
    "sgd": optim.SGD,
    "sparse_adam": optim.SparseAdam,
}


def get_optimizer(
    optimizer: str or optim.Optimizer, parameters: Iterator[Parameter]
) -> optim.Optimizer:
    """Accepts a string or Optimizer, and returns an instantiated Optimizer for
    use during model training.  (Must accept both strings and Optimizers to
    accommodate users providing mixed values for optimizers classes).

    Parameters
    ----------
    optimizer: str or optim.Optimizer
        Specified callback function to retrieve.  If an Optimizer is provided,
        rather than a string, this function just returns the same Optimizer.
    parameters: Iterator[Parameter]
        Iterator of trainable parameters in the model

    Returns
    -------
    optim.Optimizer
        Optimizer object for use during model training

    Raises
    ------
    ValueError
        If a string is provided, which does not correspond to a known Optimizer.
    """
    if isinstance(optimizer, str):
        if optimizer not in optimizer_dict.keys():
            raise ValueError(
                f"Optimizer '{optimizer}' not recognized.  "
                f"Allowed values: {list(optimizer_dict.keys())}"
            )
        else:
            optimizer = optimizer_dict[optimizer](parameters, lr=1e-3)
    elif not isinstance(optimizer, optim.Optimizer):
        raise TypeError(
            f"Optimizer can have types: [str, Optimizer].  "
            f"Found: {type(optimizer)}."
        )

    return optimizer


if __name__ == "__main__":
    import doctest

    doctest.testmod()
