"""
tiki.py
-------
Example script using `tiki` to train a MNIST handwritten digits classifier.
"""

import torch.nn as nn
from tiki import Trainer, Tester

# Torchvision not needed for tiki, but used here for convenience.
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST


class MnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 20),
            nn.Linear(20, 10),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x.view(-1, 784))


if __name__ == "__main__":
    net = MnistNet()
    tr_dataset = MNIST(root="data", train=True, transform=ToTensor(), download=True)
    va_dataset = MNIST(root="data", train=False, transform=ToTensor(), download=True)

    Trainer().train(
        net,
        tr_dataset=tr_dataset,
        va_dataset=va_dataset,
        loss="cross_entropy",
        optimizer="adam",
        epochs=1,
        gpus=[0],
        metrics=["sparse_cat_acc"],
        callbacks=[
            "terminate_on_nan",
            "early_stopping",
            "model_checkpoint",
            "tensorboard"
        ]
    )

    Tester().test(
        net,
        te_dataset=va_dataset,
        loss="cross_entropy",
        gpus=[0],
        metrics=["sparse_cat_acc"],
    )
