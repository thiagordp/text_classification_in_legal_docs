"""

@authors Thiago Raulino Dal Pont
"""
from sklearn import metrics


def evaluate_mse(y_true, y_pred):
    try:
        return metrics.precision_score(y_true, y_pred)
    except:
        return 0.0


def evaluate_mae(y_true, y_pred):
    return 0
