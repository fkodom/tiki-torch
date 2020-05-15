"""
base.py
-------
Base train/test module for all models in `tiki`.
"""

from typing import Iterable, Sequence, List
from itertools import chain
from datetime import datetime

import torch
from torch import Tensor
import torch.nn as nn

from tiki.base.utils import batch_to_device
from tiki.utils.device import get_module_device


__author__ = "Frank Odom"
__email__ = "frank.odom.iii@gmail.com"
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

        # Stateful (or potentially stateful) training objects
        self.optimizer = None
        self.loss = None
        self.metric_fns = []
        self.callbacks = []
        self.outputs = torch.tensor(0.0)
        self.alpha = 0.98

    def _log_hyperparams(
        self, batch: Sequence[Tensor] = (None,), gpus: int or Sequence[int] = ()
    ):
        hyperparams = {
            "batch_size": len(batch[0]),
            "loss": self.loss.__name__ if hasattr(self.loss, "__name__") else str(self.loss),
            "optimizer": self.optimizer.__class__.__name__,
            "gpus": gpus,
            "metrics": [
                m.__name__ if hasattr(m, "__name__") else str(m)
                for m in self.metric_fns
            ],
            "callbacks": [str(c) for c in self.callbacks],
        }
        opt_hyperparams = self.optimizer.state_dict()["param_groups"][0]
        self.hyperparams = {
            k: v for k, v in chain(hyperparams.items(), opt_hyperparams.items())
        }

    def _execute_callbacks(
        self,
        model: nn.Module = None,
        execution_times: Sequence[str] = (),
        **kwargs,
    ) -> bool:
        """Executes an Iterable of callback functions in order and returns a
        boolean flag, which specifies whether training needs to be terminated.
        If any callback requires training to terminate, immediately returns and
        sets 'break_flag' to True.

        Parameters
        ----------
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
        for callback in self.callbacks:
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

    def _update_metric(self, key: str, val: float):
        if val == 0:
            return
        elif key not in self.metrics.keys():
            self.metrics[key] = val
        else:
            self.metrics[key] *= self.alpha
            self.metrics[key] += (1 - self.alpha) * val

    def _run_on_batch(
        self,
        model: nn.Module,
        train: Batch = None,
        val: Batch = None,
        gpus: int or Sequence[int] = (),
    ) -> bool:
        """Sums the loss for multiple input batches, updates the model
        parameters. Returns the training, validation losses for the batch.

        Parameters
        ----------
        model: nn.Module
            PyTorch module to train
        train: Iterable[Batch], optional
            Tensors of inputs (one or more) and labels for training.  Labels
            should be provided as the *last* argument to the Dataset.
        val: Iterable[Batch], optional
            Dataset of inputs (one or more) and labels for validation.  Labels
            should be provided as the *last* argument to the Dataset.
        gpus: int or Sequence[int]
            If an `int` is provided, specifies the number of GPUs to use
            during training.  GPUs are chosen in order of ascending device ID.
            If a sequence of ints is provided, specifies the exact device IDs of
            the devices to use during training. Default: ()

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

        if not self.hyperparams:
            batch = train[0] if train is not None else val[0]
            self._log_hyperparams(batch=batch, gpus=gpus)

        if not (val is None or any(x is None for x in val)):
            batch = batch_to_device(val, device)
            out, loss = self.test_step(model=model, batch=batch)
            self._update_metric("va_loss", float(loss))
            with torch.no_grad():
                for m in self.metric_fns:
                    self._update_metric(m.__name__, m(out, batch[-1]).item())

        self.optimizer.zero_grad()
        if not (train is None or any(x is None for x in train)):
            batch = batch_to_device(train, device)
            out, loss = self.train_step(model=model, batch=batch)
            self._update_metric("tr_loss", float(loss))

        out = locals().get("out", torch.tensor(0.0))
        return self._execute_callbacks(
            model=model, execution_times=["on_batch"], outputs=out
        )

    def _update(self, model: nn.Module, loss: Tensor):
        # Execute callbacks before model update, and if necessary, stop training
        if self._execute_callbacks(model=model, execution_times=["on_forward"]):
            return True

        self._update_metric("tr_loss", loss.item())

        try:
            # Compute gradients and update model parameters
            loss.backward()
            self.optimizer.step()
        except RuntimeError:
            pass

    def _default_step(self, model: nn.Module, batch: Batch) -> (Tensor, float):
        self.optimizer.zero_grad()
        num_inputs = max(1, len(batch) - 1)
        out = model(*batch[:num_inputs])
        loss = self.loss(out, batch[-1])
        self._update(model=model, loss=loss)

        return out, loss

    def train_step(self, model: nn.Module, batch: Batch) -> (Tensor, float):
        return self._default_step(model=model, batch=batch)

    def validation_step(self, model: nn.Module, batch: Batch) -> (Tensor, float):
        with torch.no_grad():
            return self._default_step(model=model, batch=batch)

    def test_step(self, model: nn.Module, batch: Batch) -> (Tensor, float):
        return self.validation_step(model=model, batch=batch)
