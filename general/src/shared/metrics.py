from typing import List, Callable
from sklearn.metrics import accuracy_score, precision_score, recall_score, auc_score, confusion_matrix


def select_metrics(metrics: List[str]) -> Dict[str, Callable]:
    """Select metrics for computation based on a list of metric names.
    :param metrics: List of metric names.
    :return out: Dictionary containing name and methods.
    """
    out = {}
    for m in metrics:
        m = m.lower()
        if 'accuracy' in m and 'accuracy' not in out:
            out['accuracy'] = accuracy_score
        elif 'precision' in m and 'precision' not in out:
            out['precision'] = precision_score
        elif 'recall' in m and 'recall' not in out:
            out['recall'] = recall_score
        elif 'auc' in m and 'auc' not in out:
            out['auc'] = auc_score
        elif 'confusion' in m and 'confusion' not in out:
            out['confusion'] = confusion_matrix

    return output



