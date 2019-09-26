import torch
import numpy as np
from torch.nn import Module
from sklearn.base import ClassifierMixin, TransformerMixin
from typing import Union, List, Tuple, Dict


NPData = Union[list, np.ndarray]
ModelType = Union[ClassifierMixin, Module]
VectType = TransformerMixin

