import functools
import operator
import os
import csv
from itertools import chain
import logging

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch.nn.functional as tnnf
import torch

from ignite.engine import (
    create_supervised_trainer, Events, create_supervised_evaluator
)
import ignite.metrics
from ignite.handlers import EarlyStopping, ModelCheckpoint

logger = logging.getLogger("crab_tensors")

class CSVTrainingLogWriter(object):
    def __init__(self, path, columns):
        self.path = path
        self.columns = tuple(columns)

    def __enter__(self):
        self._file = open(self.path, "w", encoding="utf-8", newline="")
        self._csv_writer = csv.writer(
            self._file, dialect="unix", quoting=csv.QUOTE_MINIMAL
        )
        self._csv_writer.writerow(self.columns)
        return self

    def write(self, record: dict):
        line_values = (record[column] for column in self.columns)
        self._csv_writer.writerow(line_values)
        
    def __exit__(self, *exception_info):
        del self._csv_writer
        self._file.close()

        return False 
        # indicate that if an exception was raised, it must be propagated


class TrainingLog(object):
    def __init__(self, do_after_append=None):
        self.records = []
        self._do_after_append = tuple(do_after_append) if do_after_append else ()
    
    def append(self, record):
        self.records.append(record)
        for function in self._do_after_append:
            function(record)


def calc_normalized_fro_norm_of_params(model):
    number_of_scalars = 0
    squares_sum = 0
    for param in model.parameters():
        if param.grad is None:
            logger.warn(f"Gradient of {param} is None")
        else:
            squares_sum += torch.sum(param.grad.data**2).item()
            number_of_scalars_in_this_param = functools.reduce(
                operator.mul,
                param.grad.data.shape
            )
            assert number_of_scalars_in_this_param != 0
            number_of_scalars += number_of_scalars_in_this_param
    return (squares_sum / number_of_scalars)**0.5
    

def train_one_epoch(model, optimizer, train_loader, criterion, device):
    """Perform one epoch of training and return accuracy and average loss in training
    mode. Training mode means that we use losses produced by training process,
    we don't evaluate on training dataset separately."""
    model.train()
    epoch_num_samples = 0
    epoch_loss_sum = 0
    epoch_num_correct_predictions = 0
    grad_norms = []
    for (batch_ind, (X_batch, y_batch)) in enumerate(train_loader):
        X_batch = X_batch.to(device=device)
        y_batch = y_batch.to(device=device)
        optimizer.zero_grad()
        epoch_num_samples += len(X_batch)
        outputs = model(X_batch)
        batch_loss = criterion(outputs, y_batch)
        batch_loss.backward()
        epoch_loss_sum += batch_loss.item() * len(X_batch)
        _, y_batch_pred = torch.max(outputs, dim=1)
        epoch_num_correct_predictions += torch.sum(y_batch_pred == y_batch).item()
        grad_norms.append(calc_normalized_fro_norm_of_params(model))
        optimizer.step()
    return {
        "training_average_loss_training_mode": epoch_loss_sum / epoch_num_samples,
        "training_accuracy_training_mode": epoch_num_correct_predictions / epoch_num_samples,
        "training_mean_grad_norm_training_mode": np.mean(grad_norms),
        "training_std_grad_norm_training_mode": np.std(grad_norms)
    }


def train_model(
    train_loader, val_loader,
    model, optimizer, criterion,
    max_num_epochs, device,
    training_log_path,
    evaluate_every_num_epochs
):
    model = model.to(device=device)
    logger.debug(f"Is model on cuda? {model.parameters().__next__().is_cuda}")
    with CSVTrainingLogWriter(
        training_log_path,
        (
            "epoch",
            "training_average_loss_training_mode", "training_accuracy_training_mode",
            "training_mean_grad_norm_training_mode",
            "training_std_grad_norm_training_mode"#,
            #"validation_loss", "validation_accuracy"
        )
    ) as csv_log_writer:
        training_log = TrainingLog(
            do_after_append=(lambda record: csv_log_writer.write(record),)
        )
        for epoch in range(1, max_num_epochs+1):
            log_record = {
                "epoch": epoch,
                **train_one_epoch(model, optimizer, train_loader, criterion, device)
            }
            training_log.append(log_record)
        return training_log.records

def make_plots(training_log, show=True, save_path=None):
    if not show and save_path is None:
        raise ValueError("Must have show==True or save_path is not None")
    num_plots = 3
    df = pd.DataFrame(training_log).set_index("epoch")
    fig, axes = plt.subplots(nrows=3, ncols=1)
    fig.subplots_adjust(hspace=0.3)
    accuracy_axis, loss_axis, grad_norm_axis = axes.flatten()
    
    def plot_columns_with_substring(substring, axis):
        axis.set_title(substring)
        columns = df.columns[df.columns.str.contains(substring)]
        df[columns].plot(ax=axis)

    plot_columns_with_substring("accuracy", axes[0])
    plot_columns_with_substring("loss", axes[1])

    #plot gradient norm
    axes[2].set_title("Normalized norm of gradient")
    df["training_mean_grad_norm_training_mode"].plot(ax=axes[2])
    axes[2].fill_between(
        df.index,
        (df["training_mean_grad_norm_training_mode"] - df["training_std_grad_norm_training_mode"]).values,
        (df["training_mean_grad_norm_training_mode"] + df["training_std_grad_norm_training_mode"]).values,
        color="green",
        alpha=0.3
    )
    if save_path is not None:
        fig.savefig(save_path, format="png")
    if show:
        plt.show()
