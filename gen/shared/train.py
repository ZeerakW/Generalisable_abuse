import torch
import numpy as np
from tqdm import tqdm


def train_model(model, epochs, batches, loss_func, optimizer, text_field):
    losses = []
    for epoch in tqdm(range(epochs)):
        epoch_loss = []
        model.zero_grad()
        for X, y in batches:
            scores = model(X)
            loss = loss_func(scores, y)
            loss.backward()
            optimizer.step()
            epoch_loss.append(float(loss))
        losses.append(np.mean(epoch_loss))

    print("Max loss: {0};Index: {1}\nMin loss: {2}; Index: {3}".format(np.max(losses), np.argmax(losses),
                                                                       np.min(losses), np.argmin(losses)))


def evaluate_model(model, iterator, loss_func, metric_func, metric_str):
    epoch_loss = []
    epoch_eval = []

    with torch.no_grad():
        for X, y in iterator:
            scores = model(X)
            loss = loss_func(scores, y)
            scores = torch.argmax(scores, 1)
            m = metric_func(scores, y)

            epoch_loss.append(loss.item())
            epoch_eval.append(m)
    return sum(epoch_eval) / len(epoch_eval), sum(epoch_loss) / len(epoch_loss)
