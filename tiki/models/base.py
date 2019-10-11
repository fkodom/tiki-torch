"""
base.py
-------
Parent class for all neural network models in `tiki`.
"""

from abc import ABC, abstractmethod
from typing import Iterable, Sequence, Callable, List
from collections import OrderedDict

from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from tiki.losses import get_loss
from tiki.metrics import get_metric
from tiki.optimizers import get_optimizer
from tiki.callbacks import get_callback, compile_callbacks, Callback
from tiki.utils.data import get_data_loaders


__author__ = "Frank Odom"
__company__ = "Radiance Technologies, Inc."
__email__ = "frank.odom@radiancetech.com"
__classification__ = "UNCLASSIFIED"
__all__ = ["Base"]


# Define batch datatype (used for internal methods).
# Each batch is an iterable (over train, validation sets) of Tensors.
# If the inputs have inconsistent sizes, lists of Tensors are used instead.
Batch = Sequence[Tensor or List[Tensor]]


class Base(nn.Module, ABC):
    """Base class for neural network models in PyTorch, which greatly reduces
    start-up time for PyTorch development. Specific neural nets are defined by
    inheriting the 'Network' class and defining the 'forward' method. Supports
    a wide variety of model types, including fully-connected, CNN, and RNN.
    """

    def __init__(self):
        super().__init__()
        self.metrics = OrderedDict()
        self.metrics["tr_loss"] = 0.0
        self.metrics["va_loss"] = 0.0

        self.all_metrics = OrderedDict()
        self.all_metrics["tr_loss"] = []
        self.all_metrics["va_loss"] = []

        self.info = OrderedDict()
        self.info["epoch"] = 0
        self.infp["input_to_model"] = None

    @property
    def device(self) -> torch.device:
        """Gets the device where this model is currently stored. Requires that
        the model has at least 1 trainable parameter, which is located on the
        same device as the rest of the model.
        """
        for param in self.parameters():
            if hasattr(param, "device"):
                return param.device

        raise ValueError(
            "Model has no trainable parameters.  Could not determine the device"
            " location for the model."
        )

    # noinspection PyTypeChecker,PyUnusedLocal
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Pushes a set of inputs (x) through the network.

        Parameters
        ----------
        x: Tensor
            Input values

        Returns
        -------
        Tensor
            Network outputs

        Raises
        ------
        NotImplementedError
            If this method has not been implemented for sub-classes.
        """
        msg = f"forward has not been implemented for {self.__class__.__name__}"
        raise NotImplementedError(msg)

    def _get_helper_functions(
        self,
        loss: str or Callable = None,
        optimizer: str or optim.Optimizer = "adam",
        callbacks: Iterable[str or Callback] = (),
        metrics: Iterable[str or Callable] = (),
    ):
        """"""
        if loss is None:
            raise ValueError("Keyword argument 'loss' must be specified.")

        loss = get_loss(loss)
        optimizer = get_optimizer(optimizer, self.parameters())
        callbacks = [get_callback(c) for c in callbacks]
        metrics = [get_metric(m) for m in metrics]

        return loss, optimizer, callbacks, metrics

    def _execute_callbacks(
        self,
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
            if not any(hasattr(callback, ex_time) for ex_time in execution_times):
                continue

            for execution_time in execution_times:
                if hasattr(callback, execution_time):
                    func = getattr(callback, execution_time)
                    break_flag = func(self, **kwargs)

                if break_flag:
                    return True

        return False

    def batch_to_device(self, batch: Batch) -> Batch:
        if isinstance(batch[0], Tensor):
            batch = tuple(x.to(self.device) for x in batch)
        elif isinstance(batch[0], list):
            batch = tuple([y.to(self.device) for y in x] for x in batch)
        else:
            raise ValueError(f"Unallowed data type: {type(batch)}.")

        return batch

    def fit_batch(
        self,
        tr_batch: Sequence[Tensor] = (None, None),
        va_batch: Sequence[Tensor] = (None, None),
        loss: str or Callable or None = None,
        optimizer: str or optim.Optimizer = "sgd",
        alpha: float = 0.95,
        metrics: Iterable[str or Callable] = (),
        callbacks: Iterable[str or Callback] = (),
    ) -> bool:
        """Performs a single batch of network training.  Returns the training,
        validation losses for the batch.

        Parameters
        ----------
        tr_batch: Iterable[Tensor]
            Tensors of inputs (one or more) and labels for training.  Labels
            should be provided as the *last* argument to the Dataset.
        va_batch: Iterable[Tensor] (optional)
            Dataset of inputs (one or more) and labels for validation.  Labels
            should be provided as the *last* argument to the Dataset.
        loss: Callable (optional)
            Loss function used for computing training and validation error.
            **If not specified, this function will raise an exception.**
        optimizer: optim.Optimizer (optional)
            Optimization algorithm to use for network training.  If not specified,
            defaults to the class property 'self.optimizer', which will cause
            an error if it has not been set.
        alpha: float (optional)
            Controls how quickly loss values are updated using an IIR filter.
            Range: [0, 1].  Close to 1 gives fast update, but low dependence on
            previous batches.  Close to 0 gives slow update, but incorporates
            information from many previous batches.  Default: 0.95
        metrics: Iterable[str or Callable] (optional)
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
        loss, optimizer, callbacks, metrics = self._get_helper_functions(
            loss=loss, optimizer=optimizer, callbacks=callbacks, metrics=metrics
        )

        if not any(x is None for x in va_batch):
            batch = self.batch_to_device(va_batch)
            with torch.no_grad():
                out = self.forward(*batch[:-1])
                va_loss = loss(out, batch[-1])
                metrics_info = {m.__name__: m(out, batch[-1]).item() for m in metrics}
        else:
            va_loss = torch.tensor(0.0)
            metrics_info = {}

        optimizer.zero_grad()
        batch = self.batch_to_device(tr_batch)
        out = self.forward(*batch[:-1])
        tr_loss = loss(out, batch[-1])

        # Execute callbacks before model update, and if necessary, stop training
        if self._execute_callbacks(
            callbacks=callbacks,
            execution_times=["on_forward"],
            tr_loss=tr_loss,
            va_loss=va_loss,
        ):
            return True

        # Compute gradients and update model parameters
        tr_loss.backward()
        optimizer.step()
        tr_loss.detach()

        if self.metrics["tr_loss"] == 0.0:
            self.metrics["tr_loss"] = tr_loss.item()
            self.metrics["va_loss"] = va_loss.item()
        else:
            self.metrics["tr_loss"] = (
                alpha * self.metrics["tr_loss"] + (1 - alpha) * tr_loss.item()
            )
            self.metrics["va_loss"] = (
                alpha * self.metrics["va_loss"] + (1 - alpha) * va_loss.item()
            )

        for key, val in metrics_info.items():
            if key not in self.metrics.keys():
                self.metrics[key] = val
            else:
                self.metrics[key] = alpha * self.metrics[key] + (1 - alpha) * val

        return self._execute_callbacks(
            callbacks=callbacks, execution_times=["on_batch"]
        )

    def fit_epoch(
        self,
        tr_dataset: Dataset or None = None,
        va_dataset: Dataset or None = None,
        loss: str or Callable or None = None,
        optimizer: str or optim.Optimizer = "adam",
        batch_size: int = 20,
        shuffle: bool = True,
        num_workers: int = 4,
        alpha: float = 0.95,
        metrics: Iterable[str or Callable] = (),
        callbacks: Iterable[str or Callback] = (),
        progress_bar: bool = True,
        verbosity: int = 1,
    ) -> bool:
        """Performs a full epoch of training and validation.

        Parameters
        ----------
        tr_dataset: Dataset
            Dataset of inputs (one or more) and labels for training.  Labels
            should be provided as the *last* argument to the Dataset.
        va_dataset: Dataset
            Dataset of inputs (one or more) and labels for validation.  Labels
            should be provided as the *last* argument to the Dataset.
        loss: Callable
            Loss function used for computing training and validation error.
            **If not specified, this function will raise an exception.**
        optimizer: optim.Optimizer (optional)
            Optimization algorithm to use for network training.  If not specified,
            defaults to the class property 'self.optimizer', which will cause
            an error if it has not been set.
        batch_size: int (optional)
            Batch size for training.  Default: 25
        shuffle: bool (optional)
            If True, automatically shuffles training and validation examples
            at the start of each epoch.  Default: True
        num_workers: int (optional)
            Number of 'torch.multiprocessing' workers used for data loading.
            Used by the DataLoader object to avoid the GIL.  Default: 2
        epoch: int
            Training epoch number.  Used only for progress bar display.
        alpha: float (optional)
            Controls how quickly loss values are updated using an IIR filter.
            Range: [0, 1].  Close to 1 gives fast update, but low dependence on
            previous batches.  Close to 0 gives slow update, but incorporates
            information from many previous batches.  Default: 0.95
        metrics: Iterable[str or Callable] (optional)
            Iterable of performance metrics to compute for each batch of
            validation data.  If strings are provided, will attempt to retrieve
            the corresponding metric function from tiki.metrics.
        callbacks: Iterable[Callable] (optional)
            Iterable of callable functions to execute during training
        progress_bar: bool
            If True, a progress bar with performance metrics is displayed
            during each epoch.
        verbosity: int
            If progress_bar == False, determines the frequency at which
            performance metrics are printed.  Ignored if progress_bar == True.

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
        loss, optimizer, callbacks, metrics = self._get_helper_functions(
            loss=loss, optimizer=optimizer, callbacks=callbacks, metrics=metrics
        )

        tr_loader, va_loader = get_data_loaders(
            (tr_dataset, va_dataset),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

        epoch = self.info["epoch"]
        epoch_str = f"Ep {epoch}"
        if progress_bar:
            prog_bar = tqdm(
                total=min(len(tr_loader), len(va_loader)),
                desc=epoch_str,
                dynamic_ncols=True,
            )
        else:
            prog_bar = None

        for tr_batch, va_batch in zip(tr_loader, va_loader):
            break_flag = self.fit_batch(
                tr_batch,
                va_batch=va_batch,
                loss=loss,
                optimizer=optimizer,
                alpha=alpha,
                metrics=metrics,
                callbacks=compile_callbacks(callbacks, ["on_batch", "on_forward"]),
            )

            if break_flag:
                if progress_bar:
                    prog_bar.close()
                return True

            postfix = ""
            for key, val in self.metrics.items():
                postfix += f"{key}: {val:+.3e}, "

            if progress_bar:
                prog_bar.update()
                prog_bar.set_postfix_str(postfix[:-2])

        for key, val in self.metrics.items():
            if key not in self.all_metrics.keys():
                self.all_metrics[key] = []
            self.all_metrics[key].append(val)

        if progress_bar:
            prog_bar.close()
        elif epoch % verbosity == 0:
            desc = ", ".join([f"{k}: {v:+.3e}" for k, v in self.metrics.items()])
            print(f"{epoch_str} : {desc[:-2]}")

        self.info["epoch"] += 1

        return self._execute_callbacks(
            callbacks=callbacks, execution_times=["on_epoch"], epoch=epoch
        )

    def fit(
        self,
        tr_dataset: Dataset or None = None,
        va_dataset: Dataset or None = None,
        loss: str or Callable or None = None,
        optimizer: str or optim.Optimizer = "adam",
        epochs: int = 10,
        batch_size: int = 25,
        shuffle: bool = True,
        num_workers: int = 4,
        alpha: float = 0.95,
        metrics: Iterable[str or Callable] = (),
        callbacks: Iterable[str or Callback] = (),
        progress_bar: bool = True,
        verbosity: int = 1,
    ) -> None:
        """Initiates a complete training sequence for the network.

        Parameters
        ----------
        tr_dataset: Dataset
            Dataset of inputs (one or more) and labels for training.  Labels
            should be provided as the *last* argument to the Dataset.
        va_dataset: Dataset
            Dataset of inputs (one or more) and labels for validation.  Labels
            should be provided as the *last* argument to the Dataset.
        loss: Callable
            Loss function used for computing training and validation error.
            **If not specified, this function will raise an exception.**
        optimizer: str or optim.Optimizer (optional)
            Optimization algorithm to use for network training.  Can be provided
            either as an 'optim.Optimizer' instance, or a string specifier. If
            not specified, defaults to 'optim.Adam' with its default arguments.
        epochs: int (optional)
            Number of epochs for training.  Default: 10
        batch_size: int (optional)
            Batch size for training.  Default: 25
        shuffle: bool (optional)
            If True, automatically shuffles training and validation examples
            at the start of each epoch.  Default: True
        num_workers: int (optional)
            Number of 'torch.multiprocessing' workers used for data loading.
            Used by the DataLoader object to avoid the GIL.  Default: 2
        alpha: float (optional)
            Controls how quickly loss values are updated using an IIR filter.
            Range: [0, 1].  Close to 1 gives fast update, but low dependence on
            previous batches.  Close to 0 gives slow update, but incorporates
            information from many previous batches.  Default: 0.95
        metrics: Iterable[str or Callable] (optional)
            Iterable of performance metrics to compute for each batch of
            validation data.  If strings are provided, will attempt to retrieve
            the corresponding metric function from tiki.metrics.
        callbacks: Iterable[Callable] (optional)
            Iterable of callable functions to automatically execute during training
        progress_bar: bool
            If True, a progress bar with performance metrics is displayed
            during each epoch.
        verbosity: int
            If progress_bar == False, determines the frequency at which
            performance metrics are printed.  Ignored if progress_bar == True.

        Raises
        ------
        ValueError
            If keyword argument 'loss' is not provided
        NotImplementedError
            If the 'forward' method has not been implemented for sub-classes
        """
        loss, optimizer, callbacks, metrics = self._get_helper_functions(
            loss=loss, optimizer=optimizer, callbacks=callbacks, metrics=metrics
        )
        self.info["input_to_model"] = tr_dataset[0].unsqueeze(0)
        self._execute_callbacks(callbacks, execution_times=["on_start"])

        for _ in range(epochs):
            break_flag = self.fit_epoch(
                tr_dataset,
                va_dataset=va_dataset,
                loss=loss,
                optimizer=optimizer,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                alpha=alpha,
                metrics=metrics,
                callbacks=compile_callbacks(
                    callbacks, ["on_epoch", "on_batch", "on_forward"]
                ),
                progress_bar=progress_bar,
                verbosity=verbosity,
            )
            if break_flag:
                break

        self._execute_callbacks(callbacks=callbacks, execution_times=["on_final"])
