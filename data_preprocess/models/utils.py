import sklearn as sk 
import sys
import copy
import logging
import numpy as np
from algorithm import Algorithm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import sklearn.metrics as metrics 
import tensorflow as tf
import sklearn as sk
from sklearn.metrics import confusion_matrix


def recall_th_99(predictions, labels):
    """
    Returns the threshold necessary to obtain 99% recall with the provided predictions.
    """
    # Compute recall scores for a range of thresholds
    thresholds = np.arange(0, 1, 0.01)
    recalls = [metrics.recall_score(labels, predictions >= t) for t in thresholds]

    # Find the threshold that gives 99% recall
    idx = np.argmin(np.abs(np.array(recalls) - 0.99))

    return thresholds[idx]



def precision_th_99(predictions, labels):
    """
    Returns the threshold necessary to obtain 99% recall with the provided predictions.
    """
    # Compute recall scores for a range of thresholds
    thresholds = np.arange(0, 1, 0.01)
    recalls = [metrics.precision_score(labels, predictions >= t) for t in thresholds]

    # Find the threshold that gives 99% recall
    idx = np.argmin(np.abs(np.array(recalls) - 0.99))

    return thresholds[idx]