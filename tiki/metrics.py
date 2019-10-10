"""
metrics.py
----------
Defines a set of functions, which are used to compute performance metrics
during model training..
"""

from typing import Callable, Tuple, Iterable

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
    """TODO: Documentation"""
    if isinstance(inputs, list) or isinstance(inputs, tuple):
        inputs = torch.cat(inputs, dim=0)
    elif not isinstance(inputs, Tensor):
        raise TypeError(f"Data type {type(inputs)} not supported for metrics.")

    return inputs


def _prepare_tensors(outputs, labels) -> Tuple[Tensor, Tensor]:
    """TODO: Documentation"""
    if isinstance(outputs, list):
        outputs = torch.cat(outputs, 0)
    if isinstance(labels, list):
        labels = torch.cat(labels, 0)

    return outputs, labels


def mae(outputs: Iterable[Tensor], labels: Iterable[Tensor]) -> Tensor:
    """TODO: Documentation"""
    outputs, labels = _prepare_tensors(outputs, labels)
    return torch.mean((outputs - labels).type(torch.float).abs())


def mse(outputs: Iterable[Tensor], labels: Iterable[Tensor]) -> Tensor:
    """TODO: Documentation"""
    outputs, labels = _prepare_tensors(outputs, labels)
    return torch.mean((outputs ** 2 - labels ** 2).type(torch.float).abs())


def acc(outputs: Iterable[Tensor], labels: Iterable[Tensor]) -> Tensor:
    """TODO: Documentation"""
    outputs, labels = _prepare_tensors(outputs, labels)
    return torch.mean((outputs == labels.type(outputs.type())).type(torch.float))


def bin_acc(
    outputs: Iterable[Tensor], labels: Iterable[Tensor], threshold: float = 0.5
) -> Tensor:
    """TODO: Documentation"""
    outputs, labels = _prepare_tensors(outputs, labels)
    return torch.mean(
        ((outputs > threshold).type(labels.type()) == labels).type(torch.float)
    )


def sparse_cat_acc(outputs: Iterable[Tensor], labels: Iterable[Tensor]) -> Tensor:
    """TODO: Documentation"""
    outputs, labels = _prepare_tensors(outputs, labels)
    return acc(outputs.argmax(dim=-1), labels)


def cat_acc(outputs: Iterable[Tensor], labels: Iterable[Tensor]) -> Tensor:
    """TODO: Documentation"""
    outputs, labels = _prepare_tensors(outputs, labels)
    return sparse_cat_acc(outputs, labels.argmax(dim=-1))


def sparse_topk(
    outputs: Iterable[Tensor], labels: Iterable[Tensor], k: int = 5
) -> Tensor:
    """TODO: Documentation"""
    outputs, labels = _prepare_tensors(outputs, labels)
    pred_value = outputs[labels.unsqueeze(-1)]
    top_k = pred_value.type(torch.float).sum(-1) <= k
    return top_k.type(torch.float).mean()


def topk(outputs: Iterable[Tensor], labels: Iterable[Tensor], k: int = 5) -> Tensor:
    """TODO: Documentation"""
    outputs, labels = _prepare_tensors(outputs, labels)
    return sparse_topk(outputs, labels.argmax(dim=-1), k=k)


def cosine(outputs: Iterable[Tensor], labels: Iterable[Tensor]) -> Tensor:
    """TODO: Documentation"""
    outputs, labels = _prepare_tensors(outputs, labels)
    labels_ = labels.type(outputs.type())
    outputs_mag = outputs.pow(2).mean().sqrt()
    labels_mag = labels_.pow(2).mean().sqrt()

    return torch.sqrt((outputs * labels_).mean().sqrt() / (outputs_mag * labels_mag))


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
