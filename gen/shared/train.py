import torch
import numpy as np
from tqdm import tqdm
from . import base


def train_pytorch_model(model: base.ModelType, epochs: int, batches: base.DataType, loss_func: base.Callable,
                        optimizer: base.Callable, metrics: base.List[str], dev_batches: base.DataType = None):
    """Train a machine learning model.
    :model (base.ModelType): Untrained model to be trained.
    :epochs (int): The number of epochs to run.
    :batches (base.DataType): Batched training set.
    :loss_func (base.Callable): Loss function to use.
    :optimizer (bas.Callable): Optimizer function.
    :metrics (base.List[str])): Metrics to use.
    :dev_batches (base.DataType, optional): Batched dev set.
    """
    losses = []
    model.mode = True

    # TODO Get metric functions.
    scorer = None

    for epoch in tqdm(range(epochs)):
        epoch_loss = []
        model.zero_grad()
        for X, y in batches:  # TODO Update to also use dev batches.
            scores = model(X)
            loss = loss_func(scores, y)
            loss.backward()
            optimizer.step()
            epoch_loss.append(float(loss))
            performance = scorer(torch.argmax(scores, 1), y)
            # TODO Get TQDM to provide this info in output

        # Predict on dev
        if dev_batches:
            with torch.zero_grad:
                for devX, devY in dev_batches:
                    dev_scores = model(devX)
                    dev_loss = loss_func(scores, devY)
                    dev_losses.append(dev_loss)

                    # TODO Compute dev metric

            losses.append(np.mean(epoch_loss))

    print("Max loss: {0};Index: {1}\nMin loss: {2}; Index: {3}".format(np.max(losses), np.argmax(losses),
                                                                       np.min(losses), np.argmin(losses)))


def evaluate_pytorch_model(model: base.ModelType, iterator: base.DataType, loss_func: base.Callable,
                           metrics: base.List[base.Callable]) -> base.List[float]:
    """Evaluate a machine learning model.
    :model (base.ModelType): Untrained model to be trained.
    :iterator (base.DataType): Test set to evaluate on.
    :loss_func (base.Callable): Loss function to use.
    :metrics (base.List[str])): Metrics to use.
    """
    epoch_loss = []
    epoch_eval = []
    model.train_mode = False

    with torch.no_grad():
        for X, y in iterator:
            scores = model(X)
            loss = loss_func(scores, y)
            scores = torch.argmax(scores, 1)
            m = metrics(scores, y)

            epoch_loss.append(loss.item())
            epoch_eval.append(m)
    print(sum(epoch_eval) / len(epoch_eval), sum(epoch_loss) / len(epoch_loss))
    return sum(epoch_eval) / len(epoch_eval), sum(epoch_loss) / len(epoch_loss)
