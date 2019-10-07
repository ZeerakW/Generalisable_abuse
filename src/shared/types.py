import torch
import spacy
import numpy as np
from torch.nn import Module
from sklearn.base import ClassifierMixin, TransformerMixin
from typing import Union


NPData = Union[list, np.ndarray, torch.LongTensor]
ModelType = Union[ClassifierMixin, Module]
VectType = TransformerMixin
DocType = Union[str, list, spacy.tokens.doc.Doc]
