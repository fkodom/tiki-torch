"""
format_data.py
----------------------
"""
import os
import itertools
from datetime import datetime

import h5py
import torch
import numpy as np
from scipy.io import loadmat

from datacube_processing.suppression.pcbs import Pcbs
from tqdm import tqdm


def _inject_gaussian(img, center, std=0.5):
    nrow, ncol = img.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = torch.arange(nrow, device=device).unsqueeze(1).type(torch.float32)
    cols = torch.arange(ncol, device=device).unsqueeze(0).type(torch.float32)
    dx = rows - float(center[0])
    dy = cols - float(center[1])
    signal = torch.exp(-(dx ** 2 + dy ** 2).div(2 * std))

    return signal.transpose(0, 1)


def generate_truth_datacube(targets_path, bgs_frames, r=0.85):
    """Reads the Targets data from file, and generates a truth datacube.  The
    truth datacube contains 1 where targets are present, and 0 everywhere else.
    """
    nframe, nrow, ncol = bgs_frames.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"

    f = h5py.File(targets_path, "r")
    targets = f["Targets"]["Frame"]
    truth = torch.zeros((3, nframe, nrow, ncol), device=device).type(torch.float32)

    for track in targets:
        obj = f[track[0]]

        # ASSET labels have (col, row) format, rather than (row, col)
        # TODO: Check if this is intentional, or just a bug
        for i, (row, col) in enumerate(zip(obj["Row"], obj["Column"])):
            if np.isnan(row) or np.isnan(col):
                continue
            elif 0 > row or row >= nrow or 0 > col or col >= ncol:
                continue

            target_signal = _inject_gaussian(truth[0, i], (row - 1, col - 1), std=r)
            truth[0, i] += target_signal
            truth[1, i, int(row) - 1, int(col) - 1] = row % 1
            truth[2, i, int(row) - 1, int(col) - 1] = col % 1

    f.close()

    return truth


def generate_bgs_datacube(frames_path):
    """Reads the Frames data from file, and returns a background-suppressed
    datacube.  Background suppression is performed using Static PCBS.
    """
    try:
        data = loadmat(frames_path)["Frames"].astype(np.float32)
        data = np.moveaxis(data, -1, 0)
    except NotImplementedError:
        f = h5py.File(frames_path)
        data = f["Frames"].__array__().astype(np.float32)
        f.close()

    data = Pcbs().running_pcbs(data)
    data = data.astype(np.float32)

    return data / data.std()


def generate_data_chips(targets_path, frames_path, chip_size=(50, 50, 50)):
    """Generates both truth and background-suppressed datacubes. Those are
    split into smaller data chips, with size given by 'chip_size'.
    """
    inputs = generate_bgs_datacube(frames_path)
    inputs = torch.from_numpy(inputs)
    inputs = inputs.cuda() if torch.cuda.is_available() else inputs.cpu()

    labels = generate_truth_datacube(targets_path, inputs)
    inputs, labels = inputs.detach().cpu().numpy(), labels.detach().cpu().numpy()

    frames = np.arange(0, inputs.shape[0], chip_size[0])
    rows = np.arange(0, inputs.shape[1], chip_size[1])
    cols = np.arange(0, inputs.shape[2], chip_size[2])

    labels = [
        labels[:, f : f + chip_size[0], r : r + chip_size[1], c : c + chip_size[2]]
        for f, r, c in itertools.product(frames, rows, cols)
    ]
    inputs = [
        inputs[f : f + chip_size[0], r : r + chip_size[1], c : c + chip_size[2]]
        for f, r, c in itertools.product(frames, rows, cols)
    ]

    data = [(a, b) for a, b in zip(labels, inputs) if np.any(a > 0.1)]

    labels = np.array([z[0] for z in data], dtype=np.float32)
    inputs = np.array([z[1] for z in data], dtype=np.float32)
    inputs = np.expand_dims(inputs, 1)

    return labels, inputs


def generate_dataset(data_paths, chip_size=(50, 50, 50), desc=""):
    """For each file in `data_paths`, creates a set of image chips with size
    given by `chip_size`, then concatenates them into a single dataset.
    """
    labels, inputs = [], []

    for i, (target_path, frame_path) in tqdm(
        enumerate(data_paths), desc=desc, total=len(data_paths)
    ):
        a, b = generate_data_chips(target_path, frame_path, chip_size)
        labels.append(a)
        inputs.append(b)

    labels = np.concatenate(labels, 0)
    inputs = np.concatenate(inputs, 0)

    return labels, inputs


if __name__ == "__main__":
    data_chip_size = (64, 64, 64)
    validation_split = 0.3
    data_directory = os.path.join("data", "ASSET")

    # Collect the names of Frame, Target files in the data directory
    frame_files = [
        os.path.join(data_directory, f)
        for f in sorted(os.listdir(data_directory))
        if "Frames.mat" in f
    ]
    target_files = [
        os.path.join(data_directory, f)
        for f in sorted(os.listdir(data_directory))
        if "Targets.mat" in f
    ]

    # Generate the data chips
    data_files = list(zip(target_files, frame_files))
    y, x = generate_dataset(data_files, data_chip_size, desc="Generating data")

    # Split into training, validation sets
    num_validation = int(validation_split * len(x))
    x_train, y_train = x[num_validation:], y[num_validation:]
    x_val, y_val = x[:num_validation], y[:num_validation]

    print("Saving to file...")
    with h5py.File(os.path.join("data", "darcnet-training-data.h5"), "w") as f:
        f.create_dataset("Date Created", data=str(datetime.now()))
        f.create_dataset("Data Source", data="(U) ASSET Simulations")
        f.create_dataset("Data Format", data="(num_images, rows, columns, channels)")

        f.create_group("Training")
        f["Training"].create_dataset("Inputs", data=x_train, compression="gzip")
        f["Training"].create_dataset("Labels", data=y_train, compression="gzip")
        f.create_group("Validation")
        f["Validation"].create_dataset("Inputs", data=x_val, compression="gzip")
        f["Validation"].create_dataset("Labels", data=y_val, compression="gzip")
