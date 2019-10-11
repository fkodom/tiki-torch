"""
metrics.py
----------
Defines a set of functions, which are used to compute performance metrics
during model training..
"""

from typing import Callable, Sequence, Iterable

import torch
from torch import Tensor


__author__ = "Frank Odom"
__company__ = "Radiance Technologies, Inc."
__email__ = "frank.odom@radiancetech.com"
__classification__ = "UNCLASSIFIED"
__all__ = [
    "mae",
    "mse",
    "acc",
    "bin_acc",
    "sparse_cat_acc",
    "cat_acc",
    "sparse_topk",
    "topk",
    "cosine",
    "get_metric",
]

# TODO: If RL classes are included:
# * mean_reward
# * mean_steps


def _prepare_tensor(inputs: Iterable[Tensor]) -> Tensor:
    """Converts an Iterable of Tensors to a single Tensor, which is more
    easily used for computing performance metrics.  If the input type is already
    Tensor, then it is simply returned.

    Parameters
    ----------
    inputs: Iterable[Tensor]
        Iterable of Tensors to compile into a single Tensor

    Returns
    -------
    Tensor
        A single Tensor for computing metrics
    """
    if isinstance(inputs, list) or isinstance(inputs, tuple):
        inputs = torch.cat(inputs, dim=0)
    elif not isinstance(inputs, Tensor):
        raise TypeError(f"Data type {type(inputs)} not supported for metrics.")

    return inputs


def _prepare_tensors(*args: Iterable[Tensor]) -> Sequence[Tensor]:
    """Prepares multiple sets of inputs as Tensors, which are more easily used
    for computing performance metrics.

    Parameters
    ----------
    args
        One or more Iterables of Tensors

    Returns
    -------
    Sequence[Tensor]
        Tuple of compiled Tensors

    See Also
    --------
    _prepare_tensor (above)
    """
    return tuple(_prepare_tensor(x) for x in args)


def mae(outputs: Iterable[Tensor], labels: Iterable[Tensor]) -> Tensor:
    """Mean absolute error between outputs and labels.

    Parameters
    ----------
    outputs: Iterable[Tensor]
        Outputs from model
    labels: Iterable[Tensor]
        Ground truth labels

    Returns
    -------
    Tensor
        Mean absolute error

    Examples
    --------
    >>> from tiki.metrics import mae
    >>> import torch
    >>> mae(torch.ones(4), torch.zeros(4))
    tensor(1.)
    >>> mae(torch.zeros(4), torch.arange(4))
    tensor(1.5000)
    """
    outputs, labels = _prepare_tensors(outputs, labels)
    return (outputs - labels.type(outputs.type())).type(torch.float).abs().mean()


def mse(outputs: Iterable[Tensor], labels: Iterable[Tensor]) -> Tensor:
    """Mean squared error between outputs and labels.

    Parameters
    ----------
    outputs: Iterable[Tensor]
        Outputs from model
    labels: Iterable[Tensor]
        Ground truth labels

    Returns
    -------
    Tensor
        Mean squared error

    Examples
    --------
    >>> from tiki.metrics import mse
    >>> import torch
    >>> mse(torch.ones(4), torch.zeros(4))
    tensor(1.)
    >>> mse(torch.zeros(4), torch.arange(4))
    tensor(3.5000)
    """
    outputs, labels = _prepare_tensors(outputs, labels)
    return mae(outputs ** 2, labels ** 2)


def acc(outputs: Iterable[Tensor], labels: Iterable[Tensor]) -> Tensor:
    """Computes accuracy by comparing outputs directly to labels.  Outputs and
    labels should have the same sizes.

    Parameters
    ----------
    outputs: Iterable[Tensor]
        Outputs from model
    labels: Iterable[Tensor]
        Ground truth labels

    Returns
    -------
    Tensor
        Accuracy

    Examples
    --------
    >>> from tiki.metrics import acc
    >>> import torch
    >>> acc(torch.zeros(4), torch.arange(4))
    tensor(0.2500)
    >>> acc(torch.ones(4), torch.arange(4).ge(1))
    tensor(0.7500)
    """
    outputs, labels = _prepare_tensors(outputs, labels)
    return outputs.eq(labels.type(outputs.type())).type(torch.float).mean()


def bin_acc(
    outputs: Iterable[Tensor], labels: Iterable[Tensor], threshold: float = 0.5
) -> Tensor:
    """Computes binary accuracy by comparing outputs directly to labels.
    Outputs and labels should have the same sizes and contain values in the
    range [0, 1].

    Parameters
    ----------
    outputs: Iterable[Tensor]
        Outputs from model
    labels: Iterable[Tensor]
        Ground truth labels

    Returns
    -------
    Tensor
        Binary accuracy

    Examples
    --------
    >>> from tiki.metrics import bin_acc
    >>> import torch
    >>> bin_acc(torch.zeros(4), torch.ones(4))
    tensor(0.)
    >>> bin_acc(torch.ones(4), torch.arange(4).ge(1))
    tensor(0.7500)
    """
    outputs, labels = _prepare_tensors(outputs, labels)
    return torch.mean(
        (outputs > threshold).type(labels.type()).eq(labels).type(torch.float)
    )


def sparse_cat_acc(outputs: Iterable[Tensor], labels: Iterable[Tensor]) -> Tensor:
    """Computes accuracy by comparing outputs to sparse categorical labels.
    Labels should contain only integer values corresponding to the class label.
    Outputs and labels should have the same sizes, except in the final
    dimension, where outputs have extent `num_classes` and labels have extent 0.

    Parameters
    ----------
    outputs: Iterable[Tensor]
        Outputs from model
    labels: Iterable[Tensor]
        Ground truth labels

    Returns
    -------
    Tensor
        Accuracy

    Examples
    --------
    >>> from tiki.metrics import sparse_cat_acc
    >>> import torch
    >>> sparse_cat_acc(torch.tensor([0.2, 0.8]), torch.tensor(0))
    tensor(0.)
    >>> sparse_cat_acc(torch.tensor([0.2, 0.8]), torch.tensor(1))
    tensor(1.)
    """
    outputs, labels = _prepare_tensors(outputs, labels)
    return acc(outputs.argmax(dim=-1), labels)


def cat_acc(outputs: Iterable[Tensor], labels: Iterable[Tensor]) -> Tensor:
    """Computes accuracy by comparing outputs to categorical labels.  Outputs
    and labels should have the same sizes.

    Parameters
    ----------
    outputs: Iterable[Tensor]
        Outputs from model
    labels: Iterable[Tensor]
        Ground truth labels

    Returns
    -------
    Tensor
        Accuracy

    Examples
    --------
    >>> from tiki.metrics import cat_acc
    >>> import torch
    >>> cat_acc(torch.tensor([0.2, 0.8]), torch.tensor([1.0, 0.0]))
    tensor(0.)
    >>> cat_acc(torch.tensor([0.2, 0.8]), torch.tensor([0.0, 1.0]))
    tensor(1.)
    """
    outputs, labels = _prepare_tensors(outputs, labels)
    return sparse_cat_acc(outputs, labels.argmax(dim=-1))


def sparse_topk(
    outputs: Iterable[Tensor], labels: Iterable[Tensor], k: int = 5
) -> Tensor:
    """Computes top-k accuracy by comparing outputs to sparse categorical labels.
    Labels should contain only integer values corresponding to the class label.
    Outputs and labels should have the same sizes, except in the final
    dimension, where outputs have extent `num_classes` and labels have extent 0.

    Parameters
    ----------
    outputs: Iterable[Tensor]
        Outputs from model
    labels: Iterable[Tensor]
        Ground truth labels
    k: int, optional
        Output is considered correct if the labeled class is among the top `k`
        values in the network predictions.

    Returns
    -------
    Tensor
        Accuracy

    Examples
    --------
    >>> from tiki.metrics import sparse_topk
    >>> import torch
    >>> sparse_topk(torch.tensor([0.2, 0.8]), torch.tensor(0.0), k=1)
    tensor(0.)
    >>> sparse_topk(torch.tensor([0.2, 0.8]), torch.tensor(0.0), k=2)
    tensor(1.)
    """
    outputs, labels = _prepare_tensors(outputs, labels)
    top_k = outputs.topk(k, dim=-1).indices
    correct = torch.any(top_k.eq(labels.type(top_k.type())), dim=-1)
    return correct.type(torch.float).mean()


def topk(outputs: Iterable[Tensor], labels: Iterable[Tensor], k: int = 5) -> Tensor:
    """Computes top-k accuracy by comparing outputs to categorical labels.
    Outputs and labels should have the same sizes.

    Parameters
    ----------
    outputs: Iterable[Tensor]
        Outputs from model
    labels: Iterable[Tensor]
        Ground truth labels
    k: int, optional
        Output is considered correct if the labeled class is among the top `k`
        values in the network predictions.

    Returns
    -------
    Tensor
        Accuracy

    Examples
    --------
    >>> from tiki.metrics import topk
    >>> import torch
    >>> topk(torch.tensor([0.2, 0.8]), torch.tensor([1.0, 0.0]), k=1)
    tensor(0.)
    >>> topk(torch.tensor([0.2, 0.8]), torch.tensor([1.0, 0.0]), k=2)
    tensor(1.)
    """
    outputs, labels = _prepare_tensors(outputs, labels)
    return sparse_topk(outputs, labels.argmax(dim=-1), k=k)


def cosine(outputs: Iterable[Tensor], labels: Iterable[Tensor]) -> Tensor:
    """Cosine similarity between outputs and labels.

    Parameters
    ----------
    outputs: Iterable[Tensor]
        Outputs from model
    labels: Iterable[Tensor]
        Ground truth labels

    Returns
    -------
    Tensor
        Cosine similarity

    Examples
    --------
    >>> from tiki.metrics import cosine
    >>> import torch
    >>> cosine(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
    tensor(1.)
    >>> cosine(torch.tensor([0.2, 0.8]), torch.tensor([1.0, 0.0]))
    tensor(0.7364)
    """
    outputs, labels = _prepare_tensors(outputs, labels)
    labels_ = labels.type(outputs.type())
    outputs_mag = outputs.pow(2).sum(dim=-1).sqrt()
    labels_mag = labels_.pow(2).sum(dim=-1).sqrt()

    return torch.sqrt((outputs * labels_).sum().sqrt() / (outputs_mag * labels_mag))


metric_dict = {
    "mae": mae,
    "mse": mse,
    "acc": acc,
    "bin_acc": bin_acc,
    "sparse_cat_acc": sparse_cat_acc,
    "cat_acc": cat_acc,
    "sparse_topk": sparse_topk,
    "topk": topk,
    "cosine": cosine,
}


def get_metric(metric: str or Callable) -> Callable:
    """Accepts a string or Callable, and returns an Callable function for
    computing performance metrics.

    NOTE:  Must accept both strings and Callables to accommodate users providing
    mixed values for metric functions.

    Parameters
    ----------
    metric: str or Callable
        Specified performance metric to compute.  If a Callable is provided,
        rather than a string, this function just returns the same Callable.

    Returns
    -------
    Callable
        Callable function for computing performance metric

    Raises
    ------
    ValueError
        If a string is provided, which does not correspond to a known Callable.

    Examples
    --------
    >>> from typing import Callable
    >>> from tiki.metrics import get_metric
    >>> mse = get_metric("mse")
    >>> isinstance(mse, Callable)
    True
    >>> mse.__name__
    'mse'
    """
    if isinstance(metric, str):
        if metric not in metric_dict.keys():
            raise ValueError(
                f"Metric '{metric}' not recognized.  "
                f"Allowed values: {list(metric_dict.keys())}"
            )
        else:
            metric = metric_dict[metric]

    return metric


if __name__ == "__main__":
    import doctest

    doctest.testmod()
