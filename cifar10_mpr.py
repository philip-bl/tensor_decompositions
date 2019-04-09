import logging
import random
import functools
import itertools
import operator
import gc
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as tnnf
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision import transforms

import tensorly as tl

from crab_tensors.mpr_training import train_model, make_plots
from crab_tensors.td_mpr import MPRLowRankCP
from crab_tensors.ml_utils import dataset_to_tensors, StandardScaler

import click
import click_log

from libcrap.core import save_json

logger = logging.getLogger(__name__)

click_log.basic_config()

@click.command()
@click.option(
    "--out-dir", "-o", required=True, envvar="CRAB_TENSORS_OUT_DIR",
    type=click.Path(
        exists=True, dir_okay=True, file_okay=False,
        writable=True, readable=False
    ),
    help="""Path to directory where training log and other data will be written.
Can be passed as CRAB_TENSORS_OUT_DIR environment variable."""
)
@click.option(
    "--cifar10", "cifar10_dir", required=True, envvar="CIFAR10_DIR",
    type=click.Path(
        exists=True, dir_okay=True, file_okay=False,
        writable=False, readable=True
    ),
    help="""Path to cifar10 root directory.
Can be passed as CIFAR10_DIR environment variable."""
)
@click.option(
    "--device", "-d", type=str, required=False,
    envvar="CRAB_CUDA_DEVICE",
    help="""Cuda device string for pytorch. Can also be passed as
CRAB_CUDA_DEVICE environment variable. By default uses cuda:0
if available, otherwise cpu."""
)
@click.option(
    "--train-length", "-t", required=False,
    type=click.IntRange(min=1, max=50000-1),
    default=48000, show_default=True
)
@click.option(
    "--train-batch-size", required=False, type=int,
    default=256, show_default=True
)
@click.option(
    "--loaders-num-workers", "-w", required=False,
    type=click.IntRange(min=0, max=300), default=2, show_default=True
)
@click.option(
    "--polynomial-degree", required=True, type=int
)
@click.option(
    "--rank", "-r", required=True, type=int,
    help="Tensor rank (CP rank) of weights tensor."
)
@click.option(
    "--learning-rate", "-l", required=True, type=float
)
@click.option(
    "--momentum", required=True, type=float
)
@click.option(
    "--num-epochs", "-n", type=int, default=10
)
@click.option(
    "--weight-decay", type=float, default=1e-3, show_default=True
)
@click_log.simple_verbosity_option()
def main(
    out_dir, cifar10_dir, train_length, train_batch_size,
    polynomial_degree, rank, learning_rate, weight_decay,
    loaders_num_workers, device, momentum, num_epochs
):
    tl.set_backend("pytorch")
    device = torch.device(device) if device is not None else torch.device("cpu")

    start_datetime = datetime.now()

    dataset = CIFAR10(
        cifar10_dir, train=True, download=True,
        transform=transforms.ToTensor()
    )
    dataset_train, dataset_val = random_split(
        dataset,
        lengths=(train_length, len(dataset) - train_length)
    )

    X_train, y_train = dataset_to_tensors(dataset_train)
    X_val, y_val = dataset_to_tensors(dataset_val)

    # normalize dataset componentwise
    standard_scaler = StandardScaler()
    X_train = standard_scaler.fit_transform(X_train)
    X_val = standard_scaler.transform(X_val)

    # flatten
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)

    I = X_train.shape[1]
    O = 10
    
    model = MPRLowRankCP(I, O, polynom_order=polynomial_degree, rank=rank, bias=True)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum
    )

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=train_batch_size, shuffle=True,
        num_workers=loaders_num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=len(X_val),
        pin_memory=(device.type == "cuda"),
        num_workers=loaders_num_workers
    )

    file_prefix = f"cp_mpr_{start_datetime.isoformat()}_"

    training_log = train_model(
        train_loader, val_loader,
        model, optimizer,
        criterion=tnnf.cross_entropy, max_num_epochs=num_epochs, device=device,
        training_log_path=os.path.join(out_dir, file_prefix+"training_log.csv"),
        evaluate_every_num_epochs=2
    )
    make_plots(
        training_log, show=False,
        save_path=os.path.join(out_dir, file_prefix+"training_log.png")
    )
    save_json(
        {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "rank": rank,
            "polynomial_degree": polynomial_degree,
            "train_batch_size": train_batch_size,
            "train_length": train_length,
            "num_epochs": num_epochs
        },
        os.path.join(out_dir, file_prefix+"configuration.json")
    )
            


if __name__ == "__main__":
    main()
