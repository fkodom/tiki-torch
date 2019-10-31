"""
trainers.py
-------
Base trainer module for all models in `tiki`.
"""

import os
from typing import Iterable, Sequence, Callable, List

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel, DistributedDataParallel

from tiki.losses import get_loss
from tiki.metrics import get_metric
from tiki.optimizers import get_optimizer
from tiki.callbacks import get_callback, Callback
from tiki.utils.device import get_device_ids


__author__ = "Frank Odom"
__company__ = "Radiance Technologies, Inc."
__email__ = "frank.odom@radiancetech.com"
__classification__ = "UNCLASSIFIED"
__all__ = ["setup", "cleanup", "batch_to_device"]

# Define batch datatype (used for internal methods).
# Each batch is an iterable (over train, validation sets) of Tensors.
# If the inputs have inconsistent sizes, lists of Tensors are used instead.
Batch = Sequence[Tensor or List[Tensor]]


def _setup_data_parallel(
    model: nn.Module, gpus: int or Sequence[int], seed: int = None
) -> nn.Module:
    """Wraps the input model in either DistributedDataParallel (Linux/Unix) or
    DataParallel (Windows). If the model has already been wrapped, then just
    returns the input model.

    NOTE: DistributedDataParallel (Linux/Unix) achieves higher performance,
    since it uses multiprocessing in the background.

    Parameters
    ----------
    model: nn.Module
        PyTorch module to train
    gpus: int or Sequence[int]
        If an `int` is provided, specifies the number of GPUs to use
        during training.  GPUs are chosen in order of ascending device ID.
        If a sequence of ints is provided, specifies the exact device IDs of
        the devices to use during training.
    seed: int, optional
        Random seed for the parallel model.  Used only if the input model is not
        already an instance of DistributedDataParallel or DataParallel.

    Returns
    -------
    nn.Module
        If more than one GPU is specified, returns an instance of either
        DataParallel or DistributedDataParallel (subclasses of nn.Module).
        Otherwise, just returns the input nn.Module object.

    Raises
    ------
    ValueError
        If any of the specified GPUs are not available on the local system
    """
    if (
        not gpus
        or isinstance(model, DataParallel)
        or isinstance(model, DistributedDataParallel)
    ):
        dp_model = model
    else:
        # Set the address and port for the cluster (localhost).
        # Only used if DistributedDataParallel is available (Linux/Unix).
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        gpus = get_device_ids(gpus)

        if seed is None:
            seed = torch.randint(0, int(1e6), size=(1,)).item()
        torch.manual_seed(seed)

        try:
            from torch.distributed import init_process_group

            data_parallel = DistributedDataParallel
        except ImportError:
            # noinspection PyUnusedLocal
            def init_process_group(backend: str):
                pass

            data_parallel = DataParallel

        init_process_group("gloo")
        dp_model = data_parallel(model, device_ids=gpus)

    return dp_model


def setup(
    model: nn.Module = None,
    loss: str or Callable = None,
    optimizer: str or optim.Optimizer = "adam",
    callbacks: Iterable[str or Callback] = (),
    metrics: Iterable[str or Callable] = (),
    gpus: int or Sequence[int] = (),
    seed: int = None,
):
    """Fetches user-specified modules for model training.  Converts the input
    model to a parallel model, fetches loss/callback/metric functions, and
    prepares an optimizer for model training.

    Parameters
    ----------
    model: nn.Module
        PyTorch module to train
    loss: Callable, optional
        Loss function used for computing training and validation error.
        **If not specified, this function will raise an exception.**
    optimizer: optim.Optimizer, optional
        Optimization algorithm to use for network training.  If not specified,
        defaults to the class property 'self.optimizer', which will cause
        an error if it has not been set.
    metrics: Iterable[str or Callable], optional
        Iterable of performance metrics to compute for each batch of
        validation data.  If strings are provided, will attempt to retrieve
        the corresponding metric function from tiki.metrics.
    callbacks: Iterable[Callable]
        Iterable of callable functions to execute after computing outputs,
        but before updating the network parameters.
    gpus: int or Sequence[int]
        If an `int` is provided, specifies the number of GPUs to use
        during training.  GPUs are chosen in order of ascending device ID.
        If a sequence of ints is provided, specifies the exact device IDs of
        the devices to use during training.
    seed: int, optional
        Random seed for the parallel model.  Used only if the input model is not
        already an instance of DistributedDataParallel or DataParallel.

    Raises
    ------
    ValueError
        * If the `model` keyword argument is not specified
        * If the `loss` keyword argument is not specified
        * If any of the specified GPUs are not available on the local system
    """
    if model is None:
        raise ValueError("'model' is a required keyword argument.")
    if loss is None:
        raise ValueError("'loss' is a required keyword argument.")

    model = _setup_data_parallel(model, gpus=gpus, seed=seed)
    loss = get_loss(loss)
    optimizer = get_optimizer(optimizer, model.parameters())
    callbacks = [get_callback(c) for c in callbacks]
    metrics = [get_metric(m) for m in metrics]

    return model, loss, optimizer, callbacks, metrics


def cleanup():
    """Destroys any process groups spawned by DistributedDataParallel during
    setup of the parallel model.
    """
    try:
        from torch.distributed import destroy_process_group

        destroy_process_group()
    except ImportError:
        pass


def batch_to_device(batch: Batch, device: str or torch.device) -> Batch:
    """Pushes a batch of inputs to a particular device.  Designed to handle
    both Tensor and List[Tensor] inputs.

    Parameters
    ----------
    batch: Batch
        Batch of inputs to push to the device
    device: str or torch.device
        Device to move the inputs to

    Returns
    -------
    Batch
        Batch of inputs located on the specified device
    """
    if isinstance(batch[0], Tensor):
        batch = tuple(x.to(device) for x in batch)
    elif isinstance(batch[0], list):
        batch = tuple([y.to(device) for y in x] for x in batch)
    else:
        raise ValueError(f"Unallowed data type: {type(batch[0])}.")

    return batch
