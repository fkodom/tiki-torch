from typing import Sequence, List
from warnings import warn

import torch
from torch import nn, Tensor
from numba import cuda


# Define batch datatype (used for internal methods). Each batch is an
# iterable (over train, validation sets) of Tensors.
Batch = Sequence[Tensor or List[Tensor]]


NUM_GPUS_WARNING = """
Requested {requested} GPUs, but only {available} available.
Defaulting to {available} GPU(s).
"""

GPU_ID_WARNING = """
Devices {unavailable} are not available.  Defaulting to devices {available}.
"""


def get_device_ids(gpus: int or Sequence[int]):
    """Gets a list of IDs for devices that will be used during model training."""
    gpu_ids = tuple(d.id for d in cuda.list_devices())
    if isinstance(gpus, int):
        available = len(gpu_ids)
        if available < gpus:
            warn(NUM_GPUS_WARNING.format(requested=gpus, available=available))
            gpu_ids = gpu_ids[:gpus]
    else:
        gpu_ids = set(gpu_ids).intersection(gpus)
        unavailable = set(gpus).difference(gpu_ids)
        if unavailable:
            warn(GPU_ID_WARNING.format(unavailable=unavailable, available=gpu_ids))

    return list(gpu_ids)


def get_module_device(module: nn.Module) -> torch.device:
    """Gets the device where the module is currently stored. Requires that
    the model has at least 1 trainable parameter, which is located on the
    same device as the rest of the model.
    """
    for param in module.parameters():
        if hasattr(param, "device"):
            return param.device

    raise ValueError(
        "Model has no trainable parameters.  Could not determine the device"
        " location for the model."
    )
