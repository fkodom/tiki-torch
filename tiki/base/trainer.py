"""
trainers.py
-------
Base trainer module for all models in `tiki`.
"""

from typing import Iterable, Sequence, Callable, List
from itertools import chain

from tqdm import tqdm
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from tiki.callbacks import compile_callbacks, Callback
from tiki.utils.data import get_data_loaders
from tiki.base.utils import setup
from tiki.base.base import BaseTrainTest


__author__ = "Frank Odom"
__company__ = "Radiance Technologies, Inc."
__email__ = "frank.odom@radiancetech.com"
__classification__ = "UNCLASSIFIED"
__all__ = ["Trainer"]


# TODO:
# * multi-GPU loss
# * multi-GPU performance metrics

# Define batch datatype (used for internal methods).
# Each batch is an iterable (over train, validation sets) of Tensors.
# If the inputs have inconsistent sizes, lists of Tensors are used instead.
Batch = Sequence[Tensor or List[Tensor]]


class Trainer(BaseTrainTest):
    """Basic neural network trainer for supervised and unsupervised applications.
    Supports a wide variety of neural network types, including fully-connected,
    CNN, and RNN.
    """

    def __init__(self):
        super().__init__()

    def train_on_batch(self, *args, **kwargs):
        return self.run_on_batch(*args, **kwargs)

    def train_on_epoch(
        self,
        model: nn.Module,
        tr_dataset: Dataset = (),
        va_dataset: Dataset = (),
        loss: object = None,
        optimizer: str or optim.Optimizer = "adam",
        gpus: int or Sequence[int] = (),
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
        model: nn.Module
            PyTorch module to train
        tr_dataset: Dataset
            Dataset of inputs (one or more) and labels for training.  Labels
            should be provided as the *last* argument to the Dataset.
        va_dataset: Dataset
            Dataset of inputs (one or more) and labels for validation.  Labels
            should be provided as the *last* argument to the Dataset.
        loss: Callable
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
        batch_size: int, optional
            Batch size for training.  Default: 25
        shuffle: bool, optional
            If True, automatically shuffles training and validation examples
            at the start of each epoch.  Default: True
        num_workers: int, optional
            Number of 'torch.multiprocessing' workers used for data loading.
            Used by the DataLoader object to avoid the GIL.  Default: 2
        alpha: float, optional
            Controls how quickly loss values are updated using an IIR filter.
            Range: [0, 1].  Close to 1 gives fast update, but low dependence on
            previous batches.  Close to 0 gives slow update, but incorporates
            information from many previous batches.  Default: 0.95
        metrics: Iterable[str or Callable], optional
            Iterable of performance metrics to compute for each batch of
            validation data.  If strings are provided, will attempt to retrieve
            the corresponding metric function from tiki.metrics.
        callbacks: Iterable[Callable], optional
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
        self.info["epochs"] += 1
        dp_model, loss, optimizer, callbacks, metrics = setup(
            model=model,
            loss=loss,
            optimizer=optimizer,
            callbacks=callbacks,
            metrics=metrics,
            gpus=gpus,
        )

        tr_loader, va_loader = get_data_loaders(
            (tr_dataset, va_dataset),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

        epoch = self.info["epochs"]
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
            break_flag = self.run_on_batch(
                dp_model,
                tr_batch=tr_batch,
                va_batch=va_batch,
                loss=loss,
                optimizer=optimizer,
                gpus=gpus,
                alpha=alpha,
                metrics=metrics,
                callbacks=compile_callbacks(callbacks, ["on_batch", "on_forward"]),
            )

            if break_flag:
                if progress_bar:
                    prog_bar.close()
                return True

            postfixes = [f"{k}: {v:+.3e}" for k, v in self.metrics.items()]
            postfix = ", ".join(postfixes)

            if progress_bar:
                prog_bar.update()
                prog_bar.set_postfix_str(postfix)

        for key, val in chain(self.info.items(), self.metrics.items()):
            if key not in self.history.keys():
                self.history[key] = []
            self.history[key].append(val)

        if progress_bar:
            prog_bar.close()
        elif epoch % verbosity == 0:
            desc = ", ".join([f"{k}: {v:+.3e}" for k, v in self.metrics.items()])
            print(f"{epoch_str} : {desc}")

        return self._execute_callbacks(
            model, callbacks=callbacks, execution_times=["on_epoch"]
        )

    # TODO: Fix type annotation for `loss`
    # (Dependent on MyPy type checking)
    def train(
        self,
        model: nn.Module,
        tr_dataset: Dataset = (),
        va_dataset: Dataset = (),
        loss: object = None,
        optimizer: str or optim.Optimizer = "adam",
        gpus: int or Sequence[int] = 0,
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
        model: nn.Module
            PyTorch module to train
        tr_dataset: Dataset
            Dataset of inputs (one or more) and labels for training.  Labels
            should be provided as the *last* argument to the Dataset.
        va_dataset: Dataset, optional
            Dataset of inputs (one or more) and labels for validation.  Labels
            should be provided as the *last* argument to the Dataset.
        loss: Callable
            Loss function used for computing training and validation error.
            **If not specified, this function will raise an exception.**
        optimizer: str or optim.Optimizer, optional
            Optimization algorithm to use for network training.  Can be provided
            either as an 'optim.Optimizer' instance, or a string specifier. If
            not specified, defaults to 'optim.Adam' with its default arguments.
        gpus: int or Sequence[int]
            If an `int` is provided, specifies the number of GPUs to use
            during training.  GPUs are chosen in order of ascending device ID.
            If a sequence of ints is provided, specifies the exact device IDs of
            the devices to use during training.
        epochs: int, optional
            Number of epochs for training.  Default: 10
        batch_size: int, optional
            Batch size for training.  Default: 25
        shuffle: bool, optional
            If True, automatically shuffles training and validation examples
            at the start of each epoch.  Default: True
        num_workers: int, optional
            Number of 'torch.multiprocessing' workers used for data loading.
            Used by the DataLoader object to avoid the GIL.  Default: 2
        alpha: float, optional
            Controls how quickly loss values are updated using an IIR filter.
            Range: [0, 1].  Close to 1 gives fast update, but low dependence on
            previous batches.  Close to 0 gives slow update, but incorporates
            information from many previous batches.  Default: 0.95
        metrics: Iterable[str or Callable], optional
            Iterable of performance metrics to compute for each batch of
            validation data.  If strings are provided, will attempt to retrieve
            the corresponding metric function from tiki.metrics.
        callbacks: Iterable[Callable], optional
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
        dp_model, loss, optimizer, callbacks, metrics = setup(
            model=model,
            loss=loss,
            optimizer=optimizer,
            callbacks=callbacks,
            metrics=metrics,
            gpus=gpus,
        )

        self._execute_callbacks(model, callbacks, execution_times=["on_start"])

        for _ in range(epochs):
            break_flag = self.train_on_epoch(
                dp_model,
                tr_dataset=tr_dataset,
                va_dataset=va_dataset,
                loss=loss,
                optimizer=optimizer,
                gpus=gpus,
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

        self._execute_callbacks(
            model, callbacks=callbacks, execution_times=["on_end"]
        )
