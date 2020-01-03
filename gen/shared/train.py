import torch
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from . import base


def run_model(library: str, train: bool, outf, **kwargs):
    """Train or evaluate model.
    :library (str): Library of the model.
    :train (bool): Whether it's a train or test run.
    :outf (str): File to output model performance to.
    """
    if library == 'pytorch':
        func = train_pytorch_model if train else evaluate_pytorch_model
        return func(**kwargs)
    else:  # It's sklearn
        if train:
            model = train_sklearn_model(**kwargs)  # Get model
            return model
        else:
            evals = evaluate_sklearn_model(**kwargs)  # Get evaluation
            return evals


def train_sklearn_model(arg1):
    """TODO: Docstring for train_sklearn_model.

    :arg1: TODO
    :returns: TODO

    """
    pass


def train_pytorch_model(model: base.ModelType, epochs: int, batches: base.DataType, loss_func: base.Callable,
                        optimizer: base.Callable, metrics: base.Dict[str, base.Callable],
                        dev_batches: base.DataType = None, display_metric: str = 'accuracy'):
    """Train a machine learning model.
    :model (base.ModelType): Untrained model to be trained.
    :epochs (int): The number of epochs to run.
    :batches (base.DataType): Batched training set.
    :loss_func (base.Callable): Loss function to use.
    :optimizer (bas.Callable): Optimizer function.
    :metrics (base.Dict[str, base.Callable])): Metrics to use.
    :dev_batches (base.DataType, optional): Batched dev set.
    :display_metric (str): Metric to be diplayed in TQDM iterator
    """

    model.train_mode = True

    train_loss = []
    train_scores = defaultdict(list)

    for epoch in tqdm(range(epochs)):  # TODO Get TQDM to show the scores for each epoch

        model.zero_grad()  # Zero out gradients
        epoch_loss = []
        epoch_scores = defaultdict(list)

        for X, y in tqdm(batches):  # TODO Get TQDM to show the scores for each batch
            scores = model(X)
            scores = torch.argmax(scores, 1)

            loss = loss_func(scores, y)
            epoch_loss.append(float(loss))

            # Update steps
            loss.backward()
            optimizer.step()

            for metric, scorer in metrics.items():
                performance = scorer(scores, y)
                epoch_scores[metric].append(performance)

            # batch_performance = epoch_scores[display_metric][-1]  TODO
        # epoch_performance = np.mean(epoch_scores[display_metric])  TODO

        train_loss.append(sum(epoch_loss))

        for metric in metrics:
            train_scores[metric].append(np.mean(epoch_scores[metric]))

        if dev_batches:
            dev_loss, dev_scores = evaluate_pytorch_model(model, dev_batches, loss_func, metrics)
            # dev_performance = dev_performance[display_metric]  TODO

    return train_loss, dev_loss, train_scores, dev_scores


def evaluate_sklearn_model(arg1):
    """TODO: Docstring for evaluate_sklearn_model.

    :arg1: TODO
    :returns: TODO

    """
    pass


def evaluate_pytorch_model(model: base.ModelType, iterator: base.DataType, loss_func: base.Callable,
                           metrics: base.Dict[str, base.Callable]) -> base.List[float]:
    """Evaluate a machine learning model.
    :model (base.ModelType): Untrained model to be trained.
    :iterator (base.DataType): Test set to evaluate on.
    :loss_func (base.Callable): Loss function to use.
    :metrics (base.Dict[str, base.Callable])): Metrics to use.
    """
    model.train_mode = False
    dev_loss = []
    dev_scores = defaultdict(list)

    with torch.no_grad():
        for X, y in iterator:
            scores = model(X)
            scores = torch.argmax(scores, 1)

            loss = loss_func(scores, y)

            for metric, scorer in metrics.items():
                performance = scorer(scores, y)
                dev_scores[metric].append(performance)

            dev_loss.append(loss.item())

    return np.mean(dev_loss), {m: np.mean(vals) for m, vals in dev_scores.items()}
