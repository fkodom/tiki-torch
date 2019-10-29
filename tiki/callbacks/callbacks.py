"""
callbacks.py
------------
Defines a set of "callback" functions, which are executed automatically
at scheduled times during model training.
"""

import os
from typing import Iterable, List
from math import isnan, isinf
from time import time
import pickle

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DataParallel, DistributedDataParallel
from codenamize import codenamize
from torchviz import make_dot

from tiki.callbacks.base import Callback
from tiki.utils.path import custom_path


__author__ = "Frank Odom"
__company__ = "Radiance Technologies, Inc."
__email__ = "frank.odom@radiancetech.com"
__classification__ = "UNCLASSIFIED"
__codename__ = codenamize(str(time()), join="", capitalize=True)
__all__ = [
    "TerminateOnNan",
    "EarlyStopping",
    "ModelCheckpoint",
    "TensorBoard",
    "TikiHut",
    "get_callback",
    "compile_callbacks",
]


class TerminateOnNan(Callback):
    """Stops training when a `NaN`/`Inf` value is encountered during
    training.  Checks for `NaN`/`Inf` during each batch, but before network
    parameters are updated.  (So parameters don't diverge to `NaN` or `Inf`.)

    Parameters
    ----------
    verbose: bool (optional)
        If True, prints a message to the console when an action is performed.
        Nothing is printed if the callback does nothing.  Default: True
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_forward(self, trainer: object = None, **kwargs):
        check_nan_keys = ["tr_loss", "va_loss"]
        for key in check_nan_keys:
            if key not in trainer.metrics.keys():
                continue
            val = trainer.metrics[key]
            if isnan(val) or isinf(val):
                if self.verbose:
                    print(f"\nEncountered {val} value.  Terminating training.")
                return True

        return False


class EarlyStopping(Callback):
    """Stops training when a monitored quantity has stopped improving.

    Parameters
    ----------
    monitor: str (optional)
        Quantity to be monitored.  Allowed values: ["va_loss", "tr_loss"].
        Default:  "va_loss"
    min_delta: float (optional)
        Minimum change in the monitored quantity to qualify as an improvement.
        I.e. an absolute change of less than min_delta, will count as no
        improvement.  Default: 0.0
    patience: int (optional)
        Number of epochs that produced the monitored quantity with no improvement
        after which training will be stopped. Validation quantities may not be
        produced for every epoch if the validation frequency
        (model.fit(validation_freq=5)) is greater than one.  Default: 2
    verbose: bool (optional)
        If True, prints a message to the console when an action is performed.
        Nothing is printed if the callback does nothing.  Default: True

    Raises
    ------
    ValueError
        If monitor value is not contained in 'model.metrics.keys()'
    """

    def __init__(
        self,
        monitor: str = "va_loss",
        patience: int = 2,
        min_epochs: int = 5,
        min_delta: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.execution_times = ["epochs"]
        self.monitor = monitor
        self.patience = patience
        self.min_epochs = max(patience, min_epochs)
        self.min_delta = min_delta

    def on_epoch(self, trainer: object = None, **kwargs):
        if self.monitor in trainer.history.keys():
            monitor = trainer.history[self.monitor]
        else:
            monitor = []

        if len(monitor) > self.min_epochs and all(
            x + self.min_delta > monitor[-self.patience - 1]
            for x in monitor[-self.patience:]
        ):
            if self.verbose:
                print("\nLoss stopped decreasing. Terminating training.")
            return True

        return False


class ModelCheckpoint(Callback):
    """Saves a state dictionary for the trained model to file after each epoch.

    Parameters
    ----------
    path: str (optional)
        Path to the save file for this model.  If none is provided, a random
        codename will be generated with file extension '.dict', which will be
        placed inside the 'models' subfolder.
    verbose: bool (optional)
        If True, prints a message to the console when an action is performed.
        Nothing is printed if the callback does nothing.  Default: True
    """

    def __init__(self, path: str = os.path.join("models", "{codename}.dict"), **kwargs):
        super().__init__(**kwargs)
        self.path = path
        directory = os.path.join(*os.path.split(path)[:-1])
        if not os.path.exists(directory):
            os.makedirs(directory)

    def on_epoch(
            self,
            trainer: object = None,
            model: nn.Module = None,
            **kwargs
    ):
        path = custom_path(
            self.path, codename=__codename__, epoch=trainer.info["epochs"]
        )
        if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
            torch.save(model.module.state_dict(), path)
        else:
            torch.save(model.state_dict(), path)
        return False


class TensorBoard(Callback):
    """Logs training/validation data to TensorBoard for visualization.

    Parameters
    ----------
    path: str (optional)
        Path to the log directory for this run.  If none is provided, it will be
        placed in a subfolder of 'logs' using a random codename.
    verbose: bool (optional)
        If True, prints a message to the console when an action is performed.
        Nothing is printed if the callback does nothing.  Default: True
    """

    def __init__(
        self,
        path: str = os.path.join("logs", "{codename}"),
        comment: str = "",
        write_scalars: bool = True,
        write_graph: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.write_scalars = write_scalars
        self.write_graph = write_graph

        self.writer = SummaryWriter(
            log_dir=custom_path(path, codename=__codename__, **kwargs),
            comment=comment,
            **kwargs,
        )

    def on_start(self, model: nn.Module = None, **kwargs):
        if self.write_graph:
            self.writer.add_graph(
                model=model,
                input_to_model=model.info["input_to_model"]
            )

    def on_epoch(self, trainer: object = None, **kwargs):
        if self.write_scalars:
            for key, val in trainer.metrics.items():
                self.writer.add_scalar(key, val, trainer.info["epochs"])

            self.writer.flush()

        return False


class TikiHut(Callback):
    """Logs training/validation data to `tiki hut` for visualization.

    Parameters
    ----------
    path: str (optional)
        Path to the log directory for this run.  If none is provided, it will be
        placed in a subfolder of 'logs' using a random codename.
    verbose: bool (optional)
        If True, prints a message to the console when an action is performed.
        Nothing is printed if the callback does nothing.  Default: True
    """

    def __init__(
        self,
        path: str = os.path.join("logs", "{codename}"),
        write_scalars: bool = True,
        write_graph: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        path = custom_path(path, codename=__codename__, **kwargs)
        self.path = path + ".hut"
        self.write_scalars = write_scalars
        self.write_graph = write_graph
        self.name = os.path.split(path)[-1]

        directory = os.path.join(*os.path.split(path)[:-1])
        if not os.path.exists(directory):
            os.makedirs(directory)

    def on_batch(self, model: nn.Module = None, outputs: Tensor = None, **kwargs):
        if self.write_graph:
            if os.path.exists(self.path):
                log = pickle.load(open(self.path, "rb"))
            else:
                log = {"name": self.name}
            if "graph" not in log.keys():
                log["graph"] = make_dot(outputs, params=dict(model.named_parameters()))

            pickle.dump(log, open(self.path, "wb"))

            return False

    def on_epoch(self, trainer: object = None, **kwargs):
        if self.write_scalars:
            if os.path.exists(self.path):
                log = pickle.load(open(self.path, "rb"))
            else:
                log = {"name": self.name}

            log["history"] = trainer.history
            pickle.dump(log, open(self.path, "wb"))

            return False


callback_dict = {
    "terminate_on_nan": TerminateOnNan,
    "early_stopping": EarlyStopping,
    "model_checkpoint": ModelCheckpoint,
    "tensorboard": TensorBoard,
    "tiki_hut": TikiHut,
}


def get_callback(callback: str or Callback) -> Callback:
    """Accepts a string or Callback, and returns an instantiated Callback for
    use during model training.  (Must accept both strings and Callbacks to
    accommodate users providing mixed values for callback functions).

    Parameters
    ----------
    callback: str or Callback
        Specified callback function to retrieve.  If a Callback is provided,
        rather than a string, this function just returns the same Callback.

    Returns
    -------
    Callback
        Callback object for use during model training

    Raises
    ------
    ValueError
        If a string is provided, which does not correspond to a known Callback.

    Examples
    --------
    >>> from tiki.callbacks import Callback, get_callback
    >>> get_callback("model_checkpoint")
    ModelCheckpoint()
    >>> get_callback("tensorboard")
    TensorBoard()
    """
    if isinstance(callback, str):
        if callback not in callback_dict.keys():
            raise ValueError(
                f"Loss function '{callback}' not recognized.  "
                f"Allowed values: {list(callback_dict.keys())}"
            )
        else:
            callback = callback_dict[callback]()

    return callback


def compile_callbacks(
    callbacks: Iterable[str or Callback] = (), execution_times: Iterable[str] = ()
) -> List[Callback]:
    """Compiles a list of callback functions, which are executed at any of the
    specified execution times.  If strings are provided, this function also
    fetches the corresponding Callback objects.

    Parameters
    ----------
    callbacks: Iterable[str or Callback]
        Specified callback function to retrieve.  If a Callback is provided,
        rather than a string, this function just returns the same Callback.
    execution_times: Iterable[str]
        Iterable of execution times for callbacks during model training.
        Allowed: ["on_start", "on_end", "on_epoch", "on_batch", "on_forward"]

    Returns
    -------
    List[Callback]
        A list of Callback objects, which can be called during model training.

    Raises
    ------
    ValueError
        If a string is provided, which does not correspond to a known Callback.

    Examples
    --------
    >>> from tiki.callbacks import Callback, compile_callbacks
    >>> compile_callbacks(["model_checkpoint"], ["on_epoch"])
    [ModelCheckpoint()]
    >>> compile_callbacks(["model_checkpoint"], ["on_start"])
    []
    """
    compiled = []
    for callback in callbacks:
        callback = get_callback(callback)
        if any(hasattr(callback, exec_time) for exec_time in execution_times):
            compiled.append(callback)

    return compiled


if __name__ == "__main__":
    import doctest

    doctest.testmod()