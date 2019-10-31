"""
trainers.py
-------
Base trainer module for all models in `tiki`.
"""

from typing import Iterable, Sequence, Callable, List

from tqdm import tqdm
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from tiki.base.base import BaseTrainTest
from tiki.base.utils import setup
from tiki.callbacks import compile_callbacks, Callback
from tiki.utils.data import get_data_loaders


__author__ = "Frank Odom"
__company__ = "Radiance Technologies, Inc."
__email__ = "frank.odom@radiancetech.com"
__classification__ = "UNCLASSIFIED"
__all__ = ["Tester"]


# TODO:
# * multi-GPU loss
# * multi-GPU performance metrics

# Define batch datatype (used for internal methods).
# Each batch is an iterable (over train, validation sets) of Tensors.
# If the inputs have inconsistent sizes, lists of Tensors are used instead.
Batch = Sequence[Tensor or List[Tensor]]


class Tester(BaseTrainTest):
    """Basic neural network trainer for supervised and unsupervised applications.
    Supports a wide variety of neural network types, including fully-connected,
    CNN, and RNN.
    """

    def __init__(self):
        super().__init__()

    def test_on_batch(
        self,
        model: nn.Module,
        te_batch: Batch = (None,),
        loss: object = None,
        gpus: int or Sequence[int] = (),
        alpha: float = 0.99,
        metrics: Iterable[str or Callable] = (),
        callbacks: Iterable[str or Callback] = (),
    ) -> bool:
        """Performs a single batch of network training.  Returns the training,
        validation losses for the batch.

        Parameters
        ----------
        model: nn.Module
            PyTorch module to train
        te_batch: Iterable[Tensor], optional
            Dataset of inputs (one or more) and labels for validation.  Labels
            should be provided as the *last* argument to the Dataset.
        loss: Callable, optional
            Loss function used for computing training and validation error.
            **If not specified, this function will raise an exception.**
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
        return self.run_on_batch(
            model,
            va_batch=te_batch,
            loss=loss,
            gpus=gpus,
            alpha=alpha,
            metrics=metrics,
            callbacks=callbacks,
        )

    def test(
        self,
        model: nn.Module,
        te_dataset: Dataset or None = None,
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
    ) -> bool:
        """Performs a full testing run over a full dataset.

        Parameters
        ----------
        model: nn.Module
            PyTorch module to test
        te_dataset: Dataset
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

        (te_loader,) = get_data_loaders(
            (te_dataset,),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

        if progress_bar:
            prog_bar = tqdm(total=len(te_loader), desc="Test", dynamic_ncols=True)
        else:
            prog_bar = None

        for te_batch in te_loader:
            break_flag = self.test_on_batch(
                dp_model,
                te_batch=te_batch,
                loss=loss,
                gpus=gpus,
                alpha=alpha,
                metrics=metrics,
                callbacks=compile_callbacks(callbacks, ["on_batch", "on_forward"]),
            )

            if break_flag:
                if progress_bar:
                    prog_bar.close()
                return True

            postfixes = [
                f"{k}: {v:+.3e}" for k, v in self.metrics.items() if "tr" not in k
            ]
            postfix = ", ".join(postfixes).replace("va", "te")

            if progress_bar:
                prog_bar.update()
                prog_bar.set_postfix_str(postfix)

        if progress_bar:
            prog_bar.close()

        return self._execute_callbacks(
            model, callbacks=callbacks, execution_times=["on_epoch"]
        )
