"""
trainers.py
-------
Base trainer module for all models in `tiki`.
"""

from typing import Iterable, Sequence, Callable, List

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim

from tiki.base.utils import setup, batch_to_device
from tiki.callbacks import Callback
from tiki.utils.device import get_module_device


__author__ = "Frank Odom"
__company__ = "Radiance Technologies, Inc."
__email__ = "frank.odom@radiancetech.com"
__classification__ = "UNCLASSIFIED"
__all__ = ["BaseTrainTest"]


# TODO:
# * multi-GPU loss
# * multi-GPU performance metrics

# Define batch datatype (used for internal methods).
# Each batch is an iterable (over train, validation sets) of Tensors.
# If the inputs have inconsistent sizes, lists of Tensors are used instead.
Batch = Sequence[Tensor or List[Tensor]]


class BaseTrainTest(object):
    """Basic neural network trainer for supervised and unsupervised applications.
    Supports a wide variety of neural network types, including fully-connected,
    CNN, and RNN.

    Attributes
    ----------
    info: Dict
        Dictionary of training information, used for logging and callbacks
    metrics: Dict
        Dictionary of performance metrics and training/validation losses.
        Keys are strings, and values are `float` numbers for each metric.
        All metrics are updated after each training batch.
    history: Dict
        Dictionary of all historical performance metrics.  Historical metrics
        are logged at the end of each epoch.
    """

    def __init__(self):
        # Training information for logging purposes
        self.info = {"epochs": 0, "batches": 0}
        # Current and historical performance metrics
        self.metrics = {}
        self.history = {}

    def _execute_callbacks(
        self,
        model: nn.Module = None,
        callbacks: Iterable[Callback] = (),
        execution_times: Sequence[str] = (),
        **kwargs,
    ) -> bool:
        """Executes an Iterable of callback functions in order and returns a
        boolean flag, which specifies whether training needs to be terminated.
        If any callback requires training to terminate, immediately returns and
        sets 'break_flag' to True.

        Parameters
        ----------
        callbacks: Iterable[Callable]
            Iterable of callable functions to execute at the end of each epoch
        execution_times: Iterable[str]
            Iterable of execution times for callbacks during model training.
            Allowed: ["on_start", "on_end", "on_epoch", "on_batch", "on_forward"]

        Returns
        -------
        bool
            Indicates whether training should be terminated using 'break'
            statements.  Typically only happens on early stopping or Exceptions.
        """
        break_flag = False
        for callback in callbacks:
            exec_times = [e for e in execution_times if hasattr(callback, e)]
            for execution_time in exec_times:
                if hasattr(callback, execution_time):
                    func = getattr(callback, execution_time)
                    break_flag = func(trainer=self, model=model, **kwargs)
                if break_flag:
                    return True

        return False

    def run_on_batch(
        self,
        model: nn.Module,
        tr_batch: Sequence[Tensor] = (None,),
        va_batch: Sequence[Tensor] = (None,),
        loss: object = None,
        optimizer: str or optim.Optimizer = "sgd",
        gpus: int or Sequence[int] = (),
        alpha: float = 0.95,
        metrics: Iterable[str or Callable] = (),
        callbacks: Iterable[str or Callback] = (),
    ) -> bool:
        """Performs a single batch of network training.  Returns the training,
        validation losses for the batch.

        Parameters
        ----------
        model: nn.Module
            PyTorch module to train
        tr_batch: Iterable[Tensor]
            Tensors of inputs (one or more) and labels for training.  Labels
            should be provided as the *last* argument to the Dataset.
        va_batch: Iterable[Tensor], optional
            Dataset of inputs (one or more) and labels for validation.  Labels
            should be provided as the *last* argument to the Dataset.
        loss: Callable, optional
            Loss function used for computing training and validation error.
            **If not specified, this function will raise an exception.**
        optimizer: optim.Optimizer, optional
            Optimization algorithm to use for network training.  If not specified,
            defaults to the class property 'self.optimizer', which will cause
            an error if it has not been set.
        gpus: int or Sequence[int]
            If an `int` is provided, specifies the number of GPUs to use
            during training.  GPUs are chosen in order of ascending device ID.
            If a sequence of ints is provided, specifies the exact device IDs of
            the devices to use during training.
        alpha: float, optional
            Controls how quickly loss values are updated using an IIR filter.
            Range: [0, 1].  Close to 1 gives fast update, but low dependence on
            previous batches.  Close to 0 gives slow update, but incorporates
            information from many previous batches.  Default: 0.95
        metrics: Iterable[str or Callable], optional
            Iterable of performance metrics to compute for each batch of
            validation data.  If strings are provided, will attempt to retrieve
            the corresponding metric function from tiki.metrics.
        callbacks: Iterable[Callable]
            Iterable of callable functions to execute after computing outputs,
            but before updating the network parameters.

        Returns
        -------
        bool
            Indicates whether training should be terminated using 'break'
            statements.  Typically only happens on early stopping or Exceptions.

        Raises
        ------
        ValueError
            If keyword argument 'loss' is not provided
        NotImplementedError
            If the 'forward' method has not been implemented for sub-classes
        """
        device = get_module_device(model)
        self.info["batches"] += 1
        dp_model, loss, optimizer, callbacks, metrics = setup(
            model=model,
            loss=loss,
            optimizer=optimizer,
            callbacks=callbacks,
            metrics=metrics,
            gpus=gpus,
        )

        if not any(x is None for x in va_batch):
            batch = batch_to_device(va_batch, device)
            # Allow datasets with only one input Tensor (unsupervised)
            num_input_tensors = max(1, len(batch) - 1)
            with torch.no_grad():
                out = dp_model(*batch[:num_input_tensors])
                # Always pass final Tensor for "labels" (including unsupervised)
                va_loss = loss(out, batch[-1])
                metrics_info = {m.__name__: m(out, batch[-1]).item() for m in metrics}
                metrics_info["va_loss"] = va_loss.item()
        else:
            metrics_info = {}

        if not any(x is None for x in tr_batch):
            optimizer.zero_grad()
            batch = batch_to_device(tr_batch, device)
            out = dp_model(*batch[:-1])
            tr_loss = loss(out, batch[-1])

            # Execute callbacks before model update, and if necessary, stop training
            if self._execute_callbacks(
                model=model, callbacks=callbacks, execution_times=["on_forward"]
            ):
                return True

            # Compute gradients and update model parameters
            tr_loss.backward()
            optimizer.step()
            metrics_info["tr_loss"] = tr_loss.detach().item()
        else:
            out = torch.tensor(0.0)

        for key, val in metrics_info.items():
            if key not in self.metrics.keys():
                self.metrics[key] = val
            else:
                self.metrics[key] = alpha * self.metrics[key] + (1 - alpha) * val

        return self._execute_callbacks(
            model=model,
            callbacks=callbacks,
            execution_times=["on_batch"],
            outputs=out,
        )
