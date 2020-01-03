import csv
import torch
import numpy as np
from . import base
from tqdm import tqdm
from collections import defaultdict


def write_results(fout: str, scores: dict, header: base.List[str] = None) -> None:
    """TODO: Docstring for write_results.

    :fout (str): Path to file.
    :scores (dict): Dict containing scores.
    """
    with open(fout, 'a', encoding = 'utf-8') as out:
        writer = csv.writer(out, delimiter = '\t')
        if header:
            writer.writerow(header)


def run_model(library: str, train: bool, outwriter: csv.writer, model_info: list, **kwargs):
    """Train or evaluate model.
    :library (str): Library of the model.
    :train (bool): Whether it's a train or test run.
    :outwriter (csv.writer): File to output model performance to.
    :model_info (list): Information about the model to be added to each line of the output.
    """

    if train:
        func = train_pytorch_model if library == 'pytorch' else train_sklearn_model
    else:
        func = evaluate_pytorch_model if library == 'pytorch' else evaluate_sklearn_model

    return func(**kwargs)  # pytorch train: (list, int, dict, dict)


def train_pytorch_model(model: base.ModelType, epochs: int, batches: base.DataType, loss_func: base.Callable,
                        optimizer: base.Callable, metrics: base.Dict[str, base.Callable],
                        dev_batches: base.DataType = None,
                        display_metric: str = 'accuracy') -> base.Union[list, int, dict, dict]:
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
            epoch_loss.append(float(loss.item()))

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
            dev_loss, _, dev_scores, _ = evaluate_pytorch_model(model, dev_batches, loss_func, metrics)
            # dev_performance = dev_performance[display_metric]  TODO

    return train_loss, dev_loss, train_scores, dev_scores


def evaluate_pytorch_model(model: base.ModelType, iterator: base.DataType, loss_func: base.Callable,
                           metrics: base.Dict[str, base.Callable]) -> base.List[float]:
    """Evaluate a machine learning model.
    :model (base.ModelType): Untrained model to be trained.
    :iterator (base.DataType): Test set to evaluate on.
    :loss_func (base.Callable): Loss function to use.
    :metrics (base.Dict[str, base.Callable])): Metrics to use.
    """
    model.train_mode = False
    loss = []
    scores = defaultdict(list)

    with torch.no_grad():
        for X, y in iterator:
            scores = model(X)
            scores = torch.argmax(scores, 1)

            loss_f = loss_func(scores, y)

            for metric, scorer in metrics.items():
                performance = scorer(scores, y)
                scores[metric].append(performance)

            loss.append(loss_f.item())

    return np.mean(loss), None, {m: np.mean(vals) for m, vals in scores.items()}, None


def train_sklearn_model(arg1):
    """TODO: Docstring for train_sklearn_model.

    :arg1: TODO
    :returns: TODO

    """
    pass


def evaluate_sklearn_model(arg1):
    """TODO: Docstring for evaluate_sklearn_model.

    :arg1: TODO
    :returns: TODO

    """
    pass
