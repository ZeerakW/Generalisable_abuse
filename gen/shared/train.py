import torch
import numpy as np
from tqdm import tqdm
from ..shared import custom_types as t


def train_model(model: t.ModelType, epochs, batches, dev_batches, loss_func, optimizer, metrics):
    losses = []
    model.mode = True
    for epoch in tqdm(range(epochs)):
        epoch_loss = []
        model.zero_grad()
        for X, y in batches:  # TODO Update to also use dev batches.
            scores = model(X)
            loss = loss_func(scores, y)
            loss.backward()
            optimizer.step()
            epoch_loss.append(float(loss))

        # Predict on dev
        with torch.zero_grad:
            for devX, devY in dev_batches:
                dev_scores = model(devX)
                dev_loss = loss_func(scores, devY)
                dev_losses.append(dev_loss)

                # TODO Compute dev metric

        losses.append(np.mean(epoch_loss))

    print("Max loss: {0};Index: {1}\nMin loss: {2}; Index: {3}".format(np.max(losses), np.argmax(losses),
                                                                       np.min(losses), np.argmin(losses)))


def evaluate_model(model, iterator, loss_func, metric_func, metric_str):
    epoch_loss = []
    epoch_eval = []
    model.train_mode = False

    with torch.no_grad():
        for X, y in iterator:
            scores = model(X)
            loss = loss_func(scores, y)
            scores = torch.argmax(scores, 1)
            m = metric_func(scores, y)

            epoch_loss.append(loss.item())
            epoch_eval.append(m)
    print(sum(epoch_eval) / len(epoch_eval), sum(epoch_loss) / len(epoch_loss))
    return sum(epoch_eval) / len(epoch_eval), sum(epoch_loss) / len(epoch_loss)
