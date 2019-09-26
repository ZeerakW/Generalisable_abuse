from typing import List, Tuple, Callable
from src.shared import features as feats
from src.shared.metrics import select_metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from src.shared.types import NPData, ModelType, VectType
from slearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer


def select_vectorizer(vectorizer: str):
    """Identify vectorizer used and return it to be used.
    :param vectorizer:

    """

    if not any(vec in vectorizer for vec in ['dict', 'count', 'hash', 'tfidf']):
        print("You need to select from the options: dict, count, hash, tfidf. Defaulting to Dict.")
        return DictVectorizer

    vect = vectorizer.lower()
    if 'dict' in vect:
        v = DictVectorizer
    elif 'tfidf' in vect:
        v = TfidfVectorizer
    elif 'hash' in vect:
        v = HashingVectorizer
    elif 'count' in vect:
        v = CountVectorizer
    return v


def train(dataX: NPData, dataY: NPData,
          testX: NPData, testY: NPData,
          featurizer: Callable,
          vectorizer: str,
          devX: NPData = [], devY: NPData = []) -> Tuple[ModelType, VectType, VectType]:
    """Train a model and return the fitted model, vectorizer, and labelencoder.
    :param dataX: Training data.
    :param dataY: Training labels.
    :param devX: [Optional] Dev data.
    :param devY: [Optional] Dev labels.
    :param featurizer: function to transform documents into features.
    :return out_tuple: Fitted classifier, vectorizer, and labelencoder.
    """
    # Initialise things
    clf = LogisticRegression()
    le = LabelEncoder()
    vect = select_vectorizer(vectorizer)

    # Featurise data
    train_feat = featurizer(dataX)

    # Fit things
    le.fit(dataY)
    vect.fit(train_feat)

    # Transform and featurise
    trainY = le.transform(dataY)
    trainX = vect.transform(train_feat)

    # Fit model
    clf.fit(trainX, trainY)

    return clf, vect, le


def evaluate_model(model: ModelType, label_encoder: VectType, vect: VectType,
                   metrics: List[str], featurizer: Callable,
                   dataX: NPData, dataY: NPData) -> dict:
    """Evaluate model on the data.
    :param model: Fitted model.
    :param label_encoder: Fitted labelencoder.
    :param vect: Fitted vectorizer.
    :param metric: Metric to use to evaluate model.
    :param featurizer: Function to transform data to featurised.
    :param dataX: Data to predict on.
    :param dataY: Labels to evaluate on.
    :return performance: Dictionary containing evaluations.
    """

    performance = {}

    # Get metric functions and generate features
    dataX_feats = featurizer(dataX)
    eval_metrics = select_metrics(metrics)

    # Do transformations
    X = vect.transform(dataX_feats)
    Y = label_encoder.transform(dataY)

    # Get predictions
    preds = model.predict(X)

    for m in eval_metrics.keys():
        performance[m] = eval_metrics[m](Y, preds)

    return performance
