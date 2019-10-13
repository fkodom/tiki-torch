from typing import Sequence, List

import torch
from torch import Tensor
import torch.nn as nn
from GPUtil import getAvailable


# Define batch datatype (used for internal methods). Each batch is an
# iterable (over train, validation sets) of Tensors.
Batch = Sequence[Tensor or List[Tensor]]


def get_device_ids(gpus: int or Sequence[int]):
    """TODO: Documentation"""
    available = getAvailable()
    if isinstance(gpus, int):
        return available[:gpus]
    elif all(gpu in available for gpu in gpus):
        return gpus
    else:
        unavailable = [gpu for gpu in gpus if gpu not in available]
        raise ValueError(f"Devices {unavailable} not available.")


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
