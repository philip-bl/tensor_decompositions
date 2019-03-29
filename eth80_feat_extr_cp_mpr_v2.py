import logging
import random
import functools
import itertools
import operator
import gc
import os

import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as tnnf
from torch.utils.data import TensorDataset, DataLoader

import tensorly as tl

from crab_tensors import eth80
from crab_tensors.eth80 import Eth80Dataset, extract_X_y_train_test
from crab_tensors.feature_extraction import TuckerFeatureExtractor
from crab_tensors.mpr_training import train_model, make_plots
from crab_tensors.td_mpr import MPRLowRankCP

import click
import click_log

def make_CP_MPR(
    num_extracted_features, polynom_order, weights_rank,
    optimizer_creator, learning_rate, regularization_coefficient,
    betas=None
):
    model = MPRLowRankCP(
        num_extracted_features, eth80.NUM_CLASSES,
        polynom_order=polynom_order,
        rank=weights_rank,
        bias=True
    )
    
    optimizer_additional_parameters = {}
    if betas is not None:
        optimizer_additional_parameters["betas"] = betas
    
    optimizer = optimizer_creator(
        model.parameters(), lr=learning_rate,
        weight_decay=regularization_coefficient,
        **optimizer_additional_parameters
    )
    return model, optimizer

logger = logging.getLogger("crab_tensors")
click_log.basic_config(logger)

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
    "--eth80", "-e", required=True, envvar="ETH80_DIR",
    type=click.Path(
        exists=True, dir_okay=True, file_okay=False,
        writable=False, readable=True
    ),
    help="""Path to eth80-cropped-close128 directory.
Can be passed as ETH80_DIR environment variable."""
)
@click.option(
    "--device", "-d", type=str, default="cuda:0", show_default=True,
    envvar="CRAB_CUDA_DEVICE",
    help="""Cuda device string for pytorch. Can also be passed as
CRAB_CUDA_DEVICE environment variable."""
)
@click_log.simple_verbosity_option(logger)
def main(out_dir, eth80, device):
    tl.set_backend("pytorch")
    device = torch.device(device)
    eth80_dataset = Eth80Dataset(eth80)
    def evaluate_CP_MPR(save_path):
        extracted_features_shape = [4, 3, 4, 4]
        polynom_order = 3
        weights_rank = 10
        dataset_train, dataset_val = extract_X_y_train_test(
            eth80_dataset, num_test_objects_per_class=2,
            extracted_features_shape=extracted_features_shape,
            return_torch_datasets=True
        )
        train_loader = DataLoader(
            dataset_train,
            batch_size=len(dataset_train) // 2,
            shuffle=True,
            num_workers=1,
            pin_memory=True
        )
        val_loader = DataLoader(
            dataset_val,
            batch_size=len(dataset_val) // 3,
            shuffle=True,
            num_workers=1,
            pin_memory=True
        )
        model, optimizer = make_CP_MPR(
            functools.reduce(operator.mul, extracted_features_shape),
            polynom_order,
            weights_rank,
            torch.optim.SGD, learning_rate=1e-9,
            regularization_coefficient=1e-3
        )
        training_log = train_model(
            train_loader, val_loader,
            model, optimizer,
            criterion=tnnf.cross_entropy, max_num_epochs=40, device=device,
            training_log_path=os.path.join(save_path, "training_log.csv"),
            evaluate_every_num_epochs=17
        )
        return training_log
    training_log = evaluate_CP_MPR(save_path=out_dir)
    make_plots(training_log, show=False, save_path=os.path.join(out_dir, "training_log.png"))


if __name__ == "__main__":
    main()
