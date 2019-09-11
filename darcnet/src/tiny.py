"""
original.py
--------------------
Defines DARCNET models for simultaneous PIR denoising & detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as f

from darcnet.src.network import Network


# noinspection PyPep8Naming
def ConvLayer(in_features: int, mid_features: int, out_features: int):
    return nn.Sequential(
        nn.Conv3d(in_features, mid_features, 3, padding=1),
        nn.LeakyReLU(0.01),
        nn.Conv3d(mid_features, out_features, 1, padding=0),
        nn.Conv3d(out_features, out_features, 1, padding=0),
        nn.LeakyReLU(0.01),
    )


# noinspection PyPep8Naming
def ConvTransposeLayer(in_features: int, mid_features: int, out_features: int):
    return nn.Sequential(
        # nn.ConvTranspose3d(in_features, mid_features, 3, padding=1),
        # nn.LeakyReLU(0.01),
        nn.ConvTranspose3d(in_features, mid_features, 3, padding=1),
        nn.ConvTranspose3d(mid_features, out_features, 1, padding=0),
        nn.LeakyReLU(0.01),
        nn.ConvTranspose3d(out_features, out_features, 1, padding=0),
        nn.LeakyReLU(0.01),
    )


class ResLayer(nn.Module):
    r"""Defines a single residual layer within DARCNET."""

    def __init__(self):
        super(ResLayer, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(2, 4, 3, padding=1),
            nn.Conv3d(4, 4, 1, padding=0),
            nn.LeakyReLU(0.01),
            nn.Conv3d(4, 2, 1, padding=0),
        )

    def forward(self, x):
        r"""Pass input datacube `x` through the Residual Layer."""
        return nn.LeakyReLU(0.1)(x + self.conv_layers(x))


class SummedBox2d(nn.Module):
    r"""PyTorch module to sum inputs over a 2-D box (across rows, cols)."""

    def __init__(self, r=1):
        super(SummedBox2d, self).__init__()
        self.r = r

    def forward(self, x):
        w = int(2 * self.r + 1)
        kernel = torch.ones((1, 1, 1, w, w), device=x.device.type)
        return f.conv3d(x, kernel, padding=(0, self.r, self.r))


class Darcnet(Network):
    def __init__(self):
        super(Darcnet, self).__init__()
        # Build a network of 2 identical residual layers
        self.conv = ConvLayer(1, 4, 2)
        self.res_layers = nn.Sequential(
            ResLayer(),
            # ResLayer(),
        )
        self.output = ConvLayer(2, 4, 1)

    def get_device_type(self):
        r"""Get data type for the network (i.e. CPU or CUDA)."""
        return self.res_layers[-1].conv_layers[0].weight.device.type

    def _forward(self, x):
        """Pass input datacube `x` through DARCNET.  Keep input size to about
        (100, 100, 100) for training, due to large memory footprints.

        :param x: Input values
        :return: Network outputs
        """
        # Push to GPU if necessary
        if self.get_device_type() == "cuda":
            x = x.cuda()

        # Divide by the temporal standard deviation
        x = x / torch.add(x.std(dim=2, keepdim=True), 1e-4)
        # Forward pass through the residual layers
        x = self.conv(x)
        x = self.res_layers(x)
        x = self.output(x)

        return x

    def forward(self, x):
        """Pass input datacube `x` through DARCNET.  Keep input size to about
        (100, 100, 100) for training, due to large memory footprints.

        :param x: Input values
        :return: Network outputs
        """
        # Push to GPU if necessary
        if self.get_device_type() == "cuda":
            x = x.cuda()

        x = SummedBox2d()(x)
        # Divide by the temporal standard deviation
        x = x / torch.add(x.std(dim=2, keepdim=True), 1e-4)
        # Forward pass through the network layers
        x = self.conv(x)
        x = self.res_layers(x)
        x = self.output(x)

        return x

    def get_loss(self, labels, inputs):
        """Modified mean squared error between output and labels inputs.

        :param labels: Ground labels labels
        :param inputs: Training or validation inputs
        :return: Loss tensor
        """
        # Push to GPU if necessary
        if self.get_device_type() == "cuda":
            labels, inputs = labels.cuda(), inputs.cuda()

        outputs = self._forward(inputs)
        mask = labels > 0.25
        loss = nn.MSELoss()(outputs, labels)
        loss += 0.1 * nn.MSELoss()(outputs[mask], labels[mask])

        return loss
