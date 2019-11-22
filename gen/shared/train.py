import os
import pdb
import sys
import torch
import numpy as np
from tqdm import tqdm
from torchtext.data import TabularDataset, BucketIterator, Iterator
sys.path.extend(['/Users/zeerakw/Documents/PhD/projects/active/Generalisable_abuse'])

import gen.shared.types as t
from gen.shared.data import OnehotBatchGenerator, BatchGenerator


def prepare_data(data_dir: str, fields: t.List[t.Tuple[str, t.FieldType]],
                 data_field: t.Tuple[str, t.FieldType],
                 label_field: t.Tuple[str, t.FieldType], device: str, onehot: bool,
                 file_format: str, batch_size: int,
                 train_file: str, test_file: bool, dev_file: str = None, **kwargs) -> tuple:
    """
    :param data_dir (str): Directory of the data files.
    :param fields (t.List[t.Tuple[str, t.FieldType]]): The fields.
    :param data_field (t.Tuple[str, t.FieldType]): The data field name and the data field.
    :param label_field (t.Tuple[str, t.FieldType]): The label field name and the label field.
    :param device (str): The device to send it to.
    :param onehot (bool): Generate onehot encodings.
    :param file_format (str): The fileformat of the data files.
    :param batch_sizes (tuple): Tuple of ints providing batch sizes.
    :param train_file (str): Filename of test set.
    :param test_file (bool): True if a separate test file exists.
    :param dev_file (str, optional): Filename of dev set.
    :return train_batch, dev_batch: Train batched train and dev set.
    """

    train_path = os.path.join(data_dir, train_file)
    data_name, data_field = data_field
    label_name, label_field = label_field
    batch_sizes = (batch_size, batch_size)

    if not test_file:
        if not dev_file:
            input_data = TabularDataset(train_path, format = file_format, fields = fields, skip_header = True)
            train, dev, test = input_data.split(split_ratio = [0.8, 0.1, 0.1], stratified = True)
        else:
            train, dev = TabularDataset.splits(data_dir, format = file_format, fields = fields, skip_header = True,
                                               train = train_file, validation = dev_file)
    else:
        if not dev_file:
            input_data = TabularDataset.splits(data_dir, format = file_format, fields = fields, skip_header = True,
                                               train = train_file, test = test_file)
        else:
            train, dev = TabularDataset.splits(data_dir, format = file_format, fields = fields, skip_header = True,
                                               train = train_file, validation = dev_file, test = test_file)


    if not test_file and not dev_file:
    elif dev_file and not test_file:
        dev_path = os.path.join(data_dir, dev_file)
        dev = TabularDataset(dev_path, format = file_format, fields = fields, skip_header = True)
    elif test_file:
        train, dev = input_data.split(split_ratio = 0.8, stratified = True)
    else:

    data_field.build_vocab(input_data)
    VOCAB_SIZE = len(data_field.vocab)
    res = [VOCAB_SIZE]

    print("Vocab Size", len(data_field.vocab))

    if test_file:
        train_batch, dev_batch = BucketIterator.splits((train, dev), batch_sizes = batch_sizes,
                                                       sort_key = lambda x: len(x.text), device = device,
                                                       shuffle = True, repeat = False)
    else:
        train_batch, dev_batch = BucketIterator.splits((train, dev), batch_sizes = batch_sizes,
                                                       sort_key = lambda x: len(x.text), device = device,
                                                       shuffle = True, repeat = False)
        test_batch = Iterator(test, batch_size = batch_size, device = device, sort = False,
                              sort_within_batch = False, repeat = False)

    if onehot:
        train_batch = OnehotBatchGenerator(train_batch, data_name, label_name, VOCAB_SIZE)
        dev_batch = OnehotBatchGenerator(dev_batch, data_name, label_name, VOCAB_SIZE)
        if not test_file:
            test_batch = OnehotBatchGenerator(test_batch, data_name, label_name, VOCAB_SIZE)
    else:
        train_batch = BatchGenerator(train_batch, data_name, label_name)
        dev_batch = BatchGenerator(dev_batch, data_name, label_name)
        if not test_file:
            test_batch = BatchGenerator(test_batch, data_name, label_name, VOCAB_SIZE)

    if test_file:
        res.extend([train_batch, dev_batch])
    else:
        res.extend([train_batch, dev_batch, test_batch])

    return res


def prepare_test_data(data_dir: str, test_file: str, fields: t.List[t.FieldType], data_field: t.Tuple[str, t.FieldType],
                      label_field: t.Tuple[str, t.FieldType], device: str, onehot: bool, file_format: str,
                      batch_size: int, vocab_size: int, **kwargs):

    test_path = os.path.join(data_dir, test_file)
    data_name, data_field = data_field
    label_name, label_field = label_field

    test = TabularDataset(test_path, format = file_format, fields = fields, skip_header = True)
    test_batch = Iterator(test, batch_size = batch_size, device = device, sort = False, sort_within_batch = False,
                          repeat = False)

    if onehot:
        test_batch = OnehotBatchGenerator(test_batch, data_name, label_name, vocab_size)
    else:
        test_batch = BatchGenerator(test_batch, data_name, label_name)

    return test_batch


def train_model(model, epochs, batches, loss_func, optimizer, text_field):
    losses = []
    for epoch in tqdm(range(epochs)):
        epoch_loss = []
        model.zero_grad()
        for X, y in batches:  # TODO Update to also use dev batches.
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
    print(sum(epoch_eval) / len(epoch_eval), sum(epoch_loss) / len(epoch_loss))
    return sum(epoch_eval) / len(epoch_eval), sum(epoch_loss) / len(epoch_loss)
