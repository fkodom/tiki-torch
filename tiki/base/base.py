"""
base.py
-------
Base train/test module for all models in `tiki`.
"""

from typing import Iterable, Sequence, Callable, List
from itertools import chain
from datetime import datetime

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


# Define batch datatype (used for internal methods).
# Each batch is an iterable (over train, validation sets) of Tensors.
# If the inputs have inconsistent sizes, lists of Tensors are used instead.
Batch = Sequence[Tensor or List[Tensor]]
default_batch = (None,)


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
    history: Dict
        Dictionary of training hyperparameters.
    """

    def __init__(self):
        # Training information for logging purposes
        self.info = {"epochs": 0, "batches": 0, "time": datetime.now()}
        # Current and historical performance metrics
        self.metrics = {}
        self.history = {}
        self.hyperparams = {}

    def _log_hyperparams(
        self,
        batch: Sequence[Tensor] = (None,),
        loss: object = None,
        optimizer: optim.Optimizer = None,
        gpus: int or Sequence[int] = (),
        metrics: Iterable[Callable] = (),
        callbacks: Iterable[Callback] = (),
    ):
        hyperparams = {
            "batch_size": batch[0].shape[0],
            "loss": loss.__name__ if hasattr(loss, "__name__") else str(loss),
            "optimizer": optimizer.__class__.__name__,
            "gpus": gpus,
            "metrics": [
                metric.__name__ if hasattr(metric, "__name__") else str(metric)
                for metric in metrics
            ],
            "callbacks": [str(callback) for callback in callbacks],
        }
        optimizer_hyperparams = optimizer.state_dict()["param_groups"][0]
        self.hyperparams = {
            key: val
            for key, val in chain(hyperparams.items(), optimizer_hyperparams.items())
        }

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

    def _update_history(self) -> None:
        for key, val in chain(self.info.items(), self.metrics.items()):
            if key not in self.history.keys():
                self.history[key] = []
            self.history[key].append(val)

    def run_on_batches(
        self,
        model: nn.Module,
        tr_batches: Batch = (default_batch,),
        va_batches: Batch = (default_batch,),
        loss: object = None,
        optimizer: str or optim.Optimizer = "sgd",
        gpus: int or Sequence[int] = (),
        alpha: float = 0.98,
        metrics: Iterable[str or Callable] = (),
        callbacks: Iterable[str or Callback] = (),
    ) -> bool:
        """Sums the loss for multiple input batches, updates the model
        parameters. Returns the training, validation losses for the batch.

        Parameters
        ----------
        model: nn.Module
            PyTorch module to train
        tr_batches: Iterable[Batch], optional
            Tensors of inputs (one or more) and labels for training.  Labels
            should be provided as the *last* argument to the Dataset.
        va_batches: Iterable[Batch], optional
            Dataset of inputs (one or more) and labels for validation.  Labels
            should be provided as the *last* argument to the Dataset.
        loss: Callable
            Loss function used for computing training and validation error.
            **If not specified, this function will raise an exception.**
        optimizer: optim.Optimizer, optional
            Optimization algorithm to use for network training.
            Default: "adam"
        gpus: int or Sequence[int]
            If an `int` is provided, specifies the number of GPUs to use
            during training.  GPUs are chosen in order of ascending device ID.
            If a sequence of ints is provided, specifies the exact device IDs of
            the devices to use during training. Default: ()
        alpha: float, optional
            Controls how quickly loss values are updated using an IIR filter.
            Range: [0, 1].  Close to 1 gives fast update, but low dependence on
            previous batches.  Close to 0 gives slow update, but incorporates
            information from many previous batches.  Default: 0.95
        metrics: Iterable[str or Callable], optional
            Iterable of performance metrics to compute for each batch of
            validation data.  If strings are provided, will attempt to retrieve
            the corresponding metric function from tiki.metrics.  Default: ()
        callbacks: Iterable[Callable]
            Iterable of callable functions to execute after computing outputs,
            but before updating the network parameters.  Default: ()

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
        self.info["time"] = datetime.now()
        dp_model, loss, optimizer, callbacks, metrics = setup(
            model=model,
            loss=loss,
            optimizer=optimizer,
            callbacks=callbacks,
            metrics=metrics,
            gpus=gpus,
        )

        if not self.hyperparams:
            if tr_batches[0][0] is not None:
                batch = tr_batches[0]
            else:
                batch = va_batches[0]

            self._log_hyperparams(
                batch=batch,
                loss=loss,
                optimizer=optimizer,
                gpus=gpus,
                metrics=metrics,
                callbacks=callbacks,
            )

        for va_batch in va_batches:
            va_loss = 0.0
            if not any(x is None for x in va_batch):
                batch = batch_to_device(va_batch, device)
                # Allow datasets with only one input Tensor (unsupervised)
                num_input_tensors = max(1, len(batch) - 1)
                with torch.no_grad():
                    out = dp_model(*batch[:num_input_tensors])
                    # Always pass final Tensor for "labels" (including unsupervised)
                    va_loss = va_loss + loss(out, batch[-1])
                    metrics_info = {m.__name__: m(out, batch[-1]).item() for m in metrics}
            else:
                metrics_info = {}

        if va_loss != 0:
            metrics_info["va_loss"] = va_loss.item()

        for tr_batch in tr_batches:
            tr_loss = 0.0
            out = torch.empty(0, device=device)
            optimizer.zero_grad()
            if not any(x is None for x in tr_batch):
                batch = batch_to_device(tr_batch, device)
                num_inputs = max(1, len(batch) - 1)
                out = torch.cat((out, dp_model(*batch[:num_inputs])), dim=0)
                tr_loss = tr_loss + loss(out, batch[-1])

                # Execute callbacks before model update, and if necessary, stop training
                if self._execute_callbacks(
                        model=model, callbacks=callbacks, execution_times=["on_forward"]
                ):
                    return True
            else:
                out = torch.tensor(0.0)

            if tr_loss != 0:
                # Compute gradients and update model parameters
                tr_loss.backward()
                optimizer.step()
                metrics_info["tr_loss"] = tr_loss.detach().item()

            for key, val in metrics_info.items():
                if key not in self.metrics.keys():
                    self.metrics[key] = val
                else:
                    self.metrics[key] = alpha * self.metrics[key] + (1 - alpha) * val

            return self._execute_callbacks(
                model=model, callbacks=callbacks, execution_times=["on_batch"], outputs=out
            )

    def run_on_batch(
        self,
        model: nn.Module,
        tr_batch: Batch = default_batch,
        va_batch: Batch = default_batch,
        loss: object = None,
        optimizer: str or optim.Optimizer = "sgd",
        gpus: int or Sequence[int] = (),
        alpha: float = 0.98,
        metrics: Iterable[str or Callable] = (),
        callbacks: Iterable[str or Callback] = (),
    ) -> bool:
        """Performs a single batch of network training.  Returns the training,
        validation losses for the batch.

        Parameters
        ----------
        model: nn.Module
            PyTorch module to train
        tr_batch: Batch, optional
            Tensors of inputs (one or more) and labels for training.  Labels
            should be provided as the *last* argument to the Dataset.
        va_batch: Batch, optional
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
        return self.run_on_batches(
            model,
            tr_batches=(tr_batch,),
            va_batches=(va_batch,),
            loss=loss,
            optimizer=optimizer,
            gpus=gpus,
            alpha=alpha,
            metrics=metrics,
            callbacks=callbacks,
        )
