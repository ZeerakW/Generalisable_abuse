import torch
import spacy
import numpy as np
from typing import *
from torch.nn import Module
from ..shared.data import Field, Datapoint
from sklearn.base import ClassifierMixin, TransformerMixin


# Data types
FieldType = Field
DataType = Union[list, np.ndarray, torch.LongTensor, List[Datapoint]]
DocType = Union[str, List[str], spacy.tokens.doc.Doc, Datapoint]
ModelType = Union[Module]

# Model/Vectorizer Type
ModelType = Union[ClassifierMixin, Module]
VectType = TransformerMixin


# Set up label types for data processing
text_label = Field(sequential = True,
                   include_lengths = False,
                   use_vocab = True,
                   pad_token = "<PAD>",
                   unk_token = "<UNK>")

int_label = Field(sequential = False,
                  include_lengths = False,
                  use_vocab = True,
                  pad_token = None,
                  unk_token = None)

text_data = Field(sequential = True,
                  include_lengths=True,
                  use_vocab=True)
