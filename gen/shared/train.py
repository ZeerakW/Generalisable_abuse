import re
import pdb
import torch
import numpy as np
from tqdm import tqdm
import gen.shared.types as t
from string import punctuation


def read_liwc() -> dict:
    with open('/Users/zeerakw/Documents/PhD/projects/Generalisable_abuse/data/liwc-2015.csv', 'r') as liwc_f:
        liwc_dict = {}
        for line in liwc_f:
            k, v = line.strip('\n').split(',')
            if k in liwc_dict:
                liwc_dict[k] += [v]
            else:
                liwc_dict.update({k: [v]})

    return liwc_dict


global liwc_dict
liwc_dict = read_liwc()


def compute_unigram_liwc(doc: t.DocType):
    """Compute LIWC for each document document.
    :param doc (t.DocType): Document to operate on.
    :return liwc_doc (t.DocType): Document represented as LIWC categories.
    """
    # TODO modify
    liwc_doc = []
    kleene_star = [k[:-1] for k in liwc_dict if k[-1] == '*']

    if isinstance(doc, str):
        doc = [w if w[0] not in punctuation and w[-1] not in punctuation else w.strip(punctuation) for w in doc.split()]

    for w in doc:
        if w in liwc_dict:
            liwc_doc.append(liwc_dict[w][0])
        else:
            # This because re.findall is slow.
            candidates = [r for r in kleene_star if r in w]  # Find all candidates
            num_cands = len(candidates)
            try:
                if num_cands == 0:
                    term = 'NUM' if re.findall(r'[0-9]+', w) else 'UNK'
                elif num_cands == 1:
                    term = candidates[0] + '*'
                elif num_cands > 1:
                    sorted_cands = sorted(candidates, key=len, reverse = True)  # Longest first
                    term = sorted_cands[0] + '*'
                if term == 'UNK':
                    liwc_doc.append(term)
                else:
                    liwc_term = liwc_dict[term]
                    if isinstance(liwc_term, list):
                        term = "_".join(liwc_term)
                    else:
                        term = liwc_term
                    liwc_doc.append(term)
            except Exception as e:
                pdb.set_trace()
    try:
        assert(len(liwc_doc) == len(doc))
    except AssertionError:
        pdb.set_trace()

    return " ".join(liwc_doc)


def train(model, epochs, batches, loss_func, optimizer, text_field):
    losses = []
    for epoch in tqdm(range(epochs)):
        epoch_loss = []
        model.zero_grad()
        for X, y in batches:
            # print(y.shape)
            scores = model(X)
            loss = loss_func(scores, y)
            loss.backward()
            optimizer.step()
            epoch_loss.append(float(loss))
        losses.append(np.mean(epoch_loss))

    print("Max loss: {0};Index: {1}\nMin loss: {2}; Index: {3}".format(np.max(losses), np.argmax(losses),
                                                                       np.min(losses), np.argmin(losses)))


def evaluate(model, iterator, loss_func, metric_func, metric_str):

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
    return sum(epoch_eval) / len(epoch_eval),
