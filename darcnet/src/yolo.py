"""
original.py
--------------------
Defines DARCNET models for simultaneous PIR denoising & detection.
"""

from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as f

from darcnet.src.network import Network


# noinspection PyPep8Naming
def ConvLayer(in_features: int, mid_features: int, out_features: int):
    return nn.Sequential(
        nn.Conv3d(in_features, mid_features, 3, padding=1),
        nn.LeakyReLU(0.1),
        nn.Conv3d(mid_features, out_features, 1, padding=0),
    )


class ResLayer(nn.Module):
    r"""Defines a single residual layer within DARCNET."""

    def __init__(self):
        super(ResLayer, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(4, 8, 3, padding=1),
            nn.Conv3d(8, 4, 1, padding=0),
            nn.LeakyReLU(0.1),
            nn.Conv3d(4, 8, 3, padding=1),
            nn.Conv3d(8, 4, 1, padding=0)
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""Pass input datacube `x` through the Residual Layer."""
        return nn.LeakyReLU(0.1)(x + self.conv_layers(x))


class SummedBox2d(nn.Module):
    r"""PyTorch module to sum inputs over a 2-D box (across rows, cols)."""

    def __init__(self, r: int = 1):
        super(SummedBox2d, self).__init__()
        self.r = r

    def forward(self, x: Tensor) -> Tensor:
        w = int(2 * self.r + 1)
        kernel = torch.ones((1, 1, 1, w, w), device=x.device.type)
        return f.conv3d(x, kernel, padding=(0, self.r, self.r))


class Darcnet(Network):

    def __init__(self):
        super(Darcnet, self).__init__()
        # Build a network of 2 identical residual layers
        self.layers = nn.Sequential(
            ConvLayer(1, 8, 4),
            ResLayer(),
            ResLayer(),
            ConvLayer(4, 8, 3),
        )

    def get_device_type(self) -> str:
        r"""Get data type for the network (i.e. CPU or CUDA)."""
        return self.layers[0][0].weight.device.type

    def forward(self, x: Tensor) -> Tensor:
        """Pass input datacube `x` through DARCNET.  Keep input size to about (100, 100, 100) for training,
        due to large memory footprints.

        :param x: Input values
        :return: Network outputs
        """
        # Push to GPU if necessary
        if self.get_device_type() == 'cuda':
            x = x.cuda()

        x = SummedBox2d()(x)
        # Divide by the temporal standard deviation
        x = x / torch.add(x.std(dim=2, keepdim=True), 1e-4)
        # Forward pass through the residual layers
        x = self.layers(x)
        x[:, 0] = torch.where(x[:, 0] > 0, x[:, 0], torch.zeros_like(x[:, 0]))

        return x

    def batch_forward(
        self,
        x: Tensor,
        chip_size: Tuple[int, int, int] = (100, 256, 256)
    ) -> Tensor:
        """Pass input datacube `x` through DARCNET.  Keep input size to about (100, 100, 100) for training,
        due to large memory footprints.

        :param x: Input values
        :return: Network outputs
        """
        batch, nchan, nframe, nrow, ncol = x.shape
        df, dr, dc = chip_size
        out = torch.zeros(batch, 3, nframe, nrow, ncol, device=x.device)

        with torch.no_grad():
            for f0 in range(0, nframe, df):
                for r0 in range(0, nrow, dr):
                    for c0 in range(0, ncol, dc):
                        f1, r1, c1 = min(f0 + df, nframe), min(r0 + dr, nrow), min(c0 + dc, ncol)
                        out[:, :, f0:f1, r0:r1, c0:c1] = self.forward(x[:, :, f0:f1, r0:r1, c0:c1])

        return out

    @staticmethod
    def _non_max_frame(detections: Tensor, area_threshold: float = 1.0):
        """Performs non-max suppression for detections in a single frame.  This greatly reduces the memory footprint,
        compared to non-max suppression for all detections at once.

        :param detections: Detections in a single frame.  For consistency, maintains the same format as detections
            in _non_max_suppresion below.  Shape: (N, 4).  Columns contain: (confidence, frame, row, col).
        :param area_threshold: Minimum overlapping area before neighboring detections are suppressed.  For PIR
            detections, this should be set to a low value (e.g. area_threshold < 1.0).
        :return: Filtered detections, with non-maximal values removed
        """
        if detections.numel() == 0:
            return torch.empty(0, 3, device=detections.device, dtype=detections.dtype)

        dx = torch.relu(
            torch.sub(3, torch.abs(detections[:, 2].unsqueeze(0) - detections[:, 2].unsqueeze(1)))
        )
        dy = torch.relu(
            torch.sub(3, torch.abs(detections[:, 3].unsqueeze(0) - detections[:, 3].unsqueeze(1)))
        )
        mask = (dx * dy).ge(area_threshold).type(torch.float32)
        retain_idx = torch.argmax(detections[:, 0].unsqueeze(0) * mask, dim=1)

        if retain_idx.numel() > 0:
            retain_idx = torch.unique(retain_idx)
            return detections[retain_idx, 1:]
        else:
            return torch.empty(0, 3, device=detections.device, dtype=detections.dtype)

    def _non_max_suppression(self, detections: Tensor, area_threshold: float = 1.0):
        """Performs non-max suppression for each frame in a datacube.  Non-maximal detections are those that overlap
        another detection (or multiple) in the same frame, but have lower confidence scores.  Non-maximal detections
        are assumed to be duplicate detections of a single object, and they are removed from the detections array.

        :param detections: Detection values.  Shape: (N, 4).  Columns contain: (confidence, frame, row, col).
        :param area_threshold: Minimum overlapping area before neighboring detections are suppressed.  For PIR
            detections, this should be set to a low value (e.g. area_threshold < 1.0).
        :return: Filtered detections, with non-maximal values removed
        """
        detects = []
        frames = torch.unique(detections[:, 1].type(torch.int32))

        for frame in frames:
            _detects = self._non_max_frame(detections[detections[:, 1] == frame.item()], area_threshold=area_threshold)
            if _detects.numel() > 0:
                detects.append(_detects)

        if len(detects) == 0:
            return torch.empty(0, 3, device=detections.device)

        return torch.cat(detects, 0)

    def detect(self, x: Tensor, threshold: float = 0.7, area_threshold: float = 0.5) -> Tensor:
        """Performs a forward pass with Darcnet, extracts detection coordinates, and suppresses non-maximal detections.

        :param x: Raw datacube to be processed
        :param threshold: Intensity threshold for obtaining detections.  Applied after the Darcnet forward pass
        :param area_threshold: Area threshold for determining overlapping detections.  Used for non-max suppression.
        :return: Tensor of detection coordinates
        """
        with torch.no_grad():
            x = self.batch_forward(x, chip_size=(150, 256, 256)).squeeze(0)

        nchan, nframe, nrow, ncol = x.shape
        _frames, _rows, _cols = torch.meshgrid([
            torch.arange(nframe, dtype=torch.float32, device=x.device),
            torch.arange(nrow, dtype=torch.float32, device=x.device),
            torch.arange(ncol, dtype=torch.float32, device=x.device),
        ])

        mask = x[0] > threshold
        dr, dc = x[1], x[2]
        detects = torch.stack(
            (
                torch.masked_select(x[0], mask),
                torch.masked_select(_frames, mask),
                torch.masked_select(_rows + dr, mask),
                torch.masked_select(_cols + dc, mask),
            ), dim=1
        )

        detects = self._non_max_suppression(
            detects,
            area_threshold=area_threshold
        )

        return detects

    def get_loss(self, truth: Tensor, inputs: Tensor) -> Tensor:
        """Modified mean squared error between output and labels inputs.

        :param truth: Ground labels labels
        :param inputs: Training or validation inputs
        :return: Loss tensor
        """
        # Push to GPU if necessary
        if self.get_device_type() == 'cuda':
            truth, inputs = truth.cuda(), inputs.cuda()

        outputs = self.forward(inputs)
        truth[:, 0:1] = SummedBox2d()(truth[:, 0:1]).detach()

        mask = truth[:, 0] > 0.25
        loss = nn.MSELoss()(outputs[:, 0], truth[:, 0].detach())
        loss += 0.1 * nn.MSELoss()(outputs[:, 0][mask], truth[:, 0][mask].detach())
        mask = torch.stack(2 * [truth[:, 1] > 0.5], 1)
        loss += 0.1 * nn.MSELoss()(outputs[:, 1:][mask], (truth[:, 1:][mask]).detach())

        return loss
