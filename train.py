"""
train.py
------------------
Generic script for training a DARCNET model.  Alter to fit your needs.
"""
import os
import time
from datetime import datetime

import numpy as np
import torch
import h5py

from darcnet.src.yolo import Darcnet
from utils.callbacks import model_checkpoint
from utils.data import load_data
from datacube_processing.suppression.pcbs import Pcbs
from datacube_animation import animate_datacubes


if __name__ == "__main__":
    # ------------------------------ Runtime Parameters ------------------------------
    # model: str = ''
    model: str = os.path.join("darcnet", "models", "yolo.dict")
    train: bool = False
    cuda: bool = True
    epochs: int = 20
    lr: float = 1e-3
    batch: int = 20
    save_detections: bool = False
    train_data_path: str = os.path.join("data", "darcnet-training-data.h5")
    # --------------------------------------------------------------------------------

    # Declare or load a model, and push to CUDA if needed
    print("Loading model...")
    net = Darcnet()
    if model:
        net.load_state_dict(torch.load(model))
    if cuda:
        net.cuda()

    if train:
        print("Loading training, validation data...")
        (x_train, y_train), (x_val, y_val) = load_data(train_data_path)
    else:
        x_train, y_train, x_val, y_val = [], [], [], []

    if train:
        # Grab a datetime string for model checkpointing
        date = datetime.now().__str__()
        date = date[:16].replace(":", "-").replace(" ", "-")

        net.fit(
            x_train,
            y_train,
            x_val,
            y_val,
            learn_rate=float(lr),
            epochs=int(epochs),
            batch_size=int(batch),
            callbacks=[
                model_checkpoint(
                    os.path.join("darcnet", "models", f"darcnet-{date}.dict")
                )
            ],
        )

    print(r"--------------------------- DIRSIG TEST CASE ---------------------------")
    print("Loading test data...")
    test_path = os.path.join("data", "case1a_SWIR_burdened_ssgm_SNR3_Frames.mat")
    with h5py.File(test_path, "r") as f:
        datacube = f["Frames"].__array__()

    print("Performing background suppression...")
    datacube = Pcbs().running_pcbs(datacube)
    original = datacube.copy()

    datacube = torch.from_numpy(datacube.astype(np.float32))
    datacube = datacube.view(1, 1, *datacube.shape)
    if cuda:
        datacube = datacube.cuda()

    # Feed the datacube forward
    print("Processing with DARCNET...")
    start = time.time()
    detections = net.detect(datacube, threshold=2.5).cpu()
    detections = torch.cat((detections, torch.ones(detections.shape[0], 2).mul(3)), 1)
    print("Feed Forward Time: %.3E" % (time.time() - start))
    animate_datacubes(
        (original,), detections=detections.numpy(), frame_size=(6, 6), verbose=True
    )
