import functools
import os

import matplotlib.pyplot as plt

from IPython.display import clear_output

import torch.nn.functional as tnnf
import torch
from torch.utils.data import DataLoader

from ignite.engine import (
    create_supervised_trainer, Events, create_supervised_evaluator
)
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping, ModelCheckpoint

class TrainingHistory(object):
    """History of how {train,test} {loss,accuracy} changes as training goes."""
    def __init__(self):
        self.epochs = []
        self.train_losses = []
        self.train_accuracies = []
        self.validation_losses = []
        self.validation_accuracies = []
    
    def show_plots(self, clear):
        if clear:
            clear_output(wait=True)
        fig, axes = plt.subplots(ncols=2, figsize=(14, 5))
        axes = axes.flatten()
        
        axes[0].set_title("Loss")
        axes[0].plot(self.epochs, self.train_losses, label="train loss")
        axes[0].plot(self.epochs, self.validation_losses, label="validation loss")
        axes[0].legend()
        
        axes[1].set_title(
            "Accuracy. Last val: {}, best val: {}".format(
                self.validation_accuracies[-1],
                max(self.validation_accuracies)
            )
        )
        axes[1].plot(self.epochs, self.train_accuracies, label="train accuracy")
        axes[1].plot(
            self.epochs, self.validation_accuracies,
            label="validation accuracy"
        )
        axes[1].legend()
        plt.show()


def do_every_num_epochs(num_epochs):
    """This must be written after @trainer.on, not before."""
    def decorate(func):
        def decorated(engine, *args, **kwargs):
            if engine.state.epoch % num_epochs == 0:
                return func(engine, *args, **kwargs)
        return functools.update_wrapper(decorated, func)
    return decorate
# TODO: create custom event sometime


def train_and_evaluate(
    model, optimizer,
    train_loader, val_loader,
    criterion, device,
    eval_every_num_epochs, plot_every_num_epochs,
    num_epochs, early_stopping_epochs,
    save_dir, save_prefix
):
    """The main neural network training function. Trains a nn,
    evaluates it as it goes. Saves the best model in save_dir.
    The filename will have prefix save_prefix. Performs training
    for at most num_epochs. If validation accuracy doesn't improve for
    early_stopping_epochs, stops training process. Plots training
    history during training.
    
    eval_every_num_epochs - how often to evaluate.
    
    plot_every_num_epochs - how often to update plots."""
    trainer = create_supervised_trainer(
        model=model, optimizer=optimizer,
        loss_fn=criterion, device=device
    )
    
    evaluator = create_supervised_evaluator(
        model,
        metrics={
            "accuracy": Accuracy(),
            "loss": Loss(criterion)
        },
        device=device
    )
    
    history = TrainingHistory()
    
    def evaluate(loader, loss_log, accuracy_log):
        model.train(False)
        evaluator.run(loader)
        loss_log.append(evaluator.state.metrics["loss"])
        accuracy_log.append(evaluator.state.metrics["accuracy"])

    @trainer.on(Events.EPOCH_COMPLETED)
    @do_every_num_epochs(eval_every_num_epochs)
    def evaluate_on_train_and_test(engine):
        evaluate(train_loader, history.train_losses, history.train_accuracies)
        evaluate(val_loader, history.validation_losses, history.validation_accuracies)
        assert not isinstance(engine.state.epoch, torch.Tensor)
        history.epochs.append(engine.state.epoch)
    
    @trainer.on(Events.EPOCH_COMPLETED)
    @do_every_num_epochs(plot_every_num_epochs)
    def update_plot(*args):
        history.show_plots(clear=True)
    
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        do_every_num_epochs(eval_every_num_epochs)(EarlyStopping(
        
        # wait this many epochs before stopping
        patience=early_stopping_epochs // eval_every_num_epochs + 1,
        
        score_function=lambda engine: history.validation_accuracies[-1],
        trainer=trainer
        ))
    )# TODO ES DOESNT WORK FIX THIS
    # perhaps rewrite all this shit
    # also use https://github.com/pytorch/ignite/pull/455
    
    # Add handler which saves model to disk whenever it achieves
    # new best result
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        do_every_num_epochs(eval_every_num_epochs)(ModelCheckpoint(
            save_dir, save_prefix,
            score_function=lambda engine: history.validation_accuracies[-1],
            n_saved=1, atomic=True, require_empty=False,
            save_as_state_dict=True
        )),
        {"model": model} # what should be saved
    )
        
    trainer.run(train_loader, max_epochs=num_epochs)
    
    # save training history to disk as well
    with open(os.path.join(save_dir, f"{save_prefix}_history.pkl"), "wb") as history_f:
        pickle.dump(history, history_f)
    
    # usually we will not be using return values, because it returns
    # last model, not the best model
    return (
        model,
        history
    )

                     
def train_and_evaluate(
    X_train, y_train, X_test, y_test,
    model, optimizer,
    eval_every_num_epochs, plot_every_num_epochs,
    num_epochs
):    
    trainer = create_supervised_trainer(
        model=model, optimizer=optimizer,
        loss_fn=tnnf.cross_entropy
    )

    evaluations_epochs = []
    train_log = {
        "losses": [],
        "accuracies": []
    }
    test_log = {
        "losses": [],
        "accuracies": []
    }
    
    def evaluate(X, y, log):
        model.train(False)
        logits = model(X)
        loss = tnnf.cross_entropy(logits, y).item()
        predictions = logits.argmax(dim=1)
        accuracy = (y == predictions).sum().item() / len(y)
        log["losses"].append(loss)
        log["accuracies"].append(accuracy)

    @trainer.on(Events.EPOCH_COMPLETED)
    @do_every_num_epochs(eval_every_num_epochs)
    def evaluate_on_train_and_test(engine):
        evaluate(X_train, y_train, train_log)
        evaluate(X_test, y_test, test_log)
        assert not isinstance(engine.state.epoch, torch.Tensor)
        evaluations_epochs.append(engine.state.epoch)
    
    @trainer.on(Events.EPOCH_COMPLETED)
    @do_every_num_epochs(plot_every_num_epochs)
    def update_plot(engine):
        clear_output(wait=True)
        fig, axes = plt.subplots(ncols=2, figsize=(14, 5))
        axes = axes.flatten()
        axes[0].set_title("Loss")
        axes[0].plot(evaluations_epochs, train_log["losses"], label="train loss")
        axes[0].plot(evaluations_epochs, test_log["losses"], label="test loss")
        axes[0].legend()
        axes[1].set_title(f"Accuracy. Test: {test_log['accuracies'][-1]}")
        axes[1].plot(evaluations_epochs, train_log["accuracies"], label="train accuracy")
        axes[1].plot(evaluations_epochs, test_log["accuracies"], label="test accuracy")
        axes[1].legend()
        plt.show()
    
    # Add handler which saves model to disk whenever it achieves
    # new best result
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        ModelCheckpoint(
            save_dir, save_prefix,
            score_function=lambda engine: history.validation_accuracies[-1],
            n_saved=1, atomic=True, require_empty=False,
            save_as_state_dict=True
        ),
        {"model": model} # what should be saved
    )
        
    trainer.run([(X_train, y_train)], max_epochs=num_epochs)
    return model, evaluations_epochs, train_log, test_log
# In[ ]:


def memreport():
    print(f"""
    {torch.cuda.memory_allocated()/1024/1024/1024} Gb allocated
    {torch.cuda.memory_cached()/1024/1024/1024} Gb cached
    """)
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())


def calculate_features_mean_of_dataset(dataset, batch_size=None, loader_num_workers=16):
    loader = DataLoader(
        dataset, batch_size if batch_size is not None else len(dataset),
        num_workers=loader_num_workers
    )
    X_sum = torch.zeros_like(dataset[0][0])
    for (X_batch, y_batch) in loader:
        X_sum += torch.sum(X_batch, dim=0)
    X_mean = X_sum / len(dataset)
    X_sum_squares = torch.zeros_like(X_sum)
    for (X_batch, y_batch) in loader:
        X_sum_squares += torch.sum((X_batch - X_mean)**2, dim=0)
    X_std = torch.sqrt(X_sum_squares / len(dataset))
    return X_mean, X_std


class StandardScaler(object):
    def __init__(self):
        pass

    def fit(self, X):
        self.mean = torch.mean(X, dim=0)
        self.std = torch.std(X, dim=0)
        self.std[self.std == 0] = 1

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        return (X - self.mean) / self.std


def dataset_to_tensors(dataset, num_workers=16):
    return next(iter(DataLoader(
        dataset, batch_size=len(dataset), num_workers=num_workers
    )))
