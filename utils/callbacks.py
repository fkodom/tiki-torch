from typing import AnyStr, Callable
from collections import OrderedDict

import torch
import torch.nn as nn


def model_checkpoint(save_path: AnyStr) -> Callable:
    """Defines a callback for saving PyTorch models during training.  The
    callback function will always accept `self` as the first argument.  We
    include `*args` positional arguments for added flexibility, so that a
    network could pass multiple arguments (e.g. training/validation loss,
    epoch number, etc.) without breaking it.

    :param save_path:  Absolute path to the model's save file
    :return:  Callback function for model saving
    """

    def callback(model: nn.Module, *args):
        state_dict = model.state_dict()

        # Push all parameters to CPU for cross-compatibility
        cpu_state_dict = OrderedDict()
        for key, value in state_dict.items():
            cpu_state_dict[key] = value.detach().cpu()

        torch.save(cpu_state_dict, save_path)

    return callback
