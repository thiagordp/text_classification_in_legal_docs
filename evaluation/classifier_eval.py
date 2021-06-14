"""

@author Thiago R. Dal Pont
"""

# TODO: Implement classifier evalution methods
from sklearn import metrics
from sklearn.metrics import classification_report


def evaluate_balanced_accuracy(y_true, y_pred):
    try:
        return metrics.balanced_accuracy_score(y_true, y_pred)
    except:
        return 0.0


def evaluate_accuracy(y_true, y_pred):
    try:
        return metrics.accuracy_score(y_true, y_pred)
    except:
        return 0.0


def evaluate_f_score(y_true, y_pred):
    try:
        return [
            metrics.f1_score(y_true, y_pred, average="micro"),
            metrics.f1_score(y_true, y_pred, average="macro"),
            metrics.f1_score(y_true, y_pred, average="weighted")
        ]
    except:
        return [0, 0, 0]


def confusion_matrix(y_true, y_pred):
    # try:
    return metrics.confusion_matrix(y_true, y_pred)
    # except:
    #   return 0.0


def evaluate_precision(y_true, y_pred):
    # try:
    return [
        metrics.precision_score(y_true, y_pred, average="micro"),
        metrics.precision_score(y_true, y_pred, average="macro"),
        metrics.precision_score(y_true, y_pred, average="weighted")
    ]
    # except:
    #    return 0.0


def evaluate_recall(y_true, y_pred):
    # try:
    return [
        metrics.recall_score(y_true, y_pred, average="micro"),
        metrics.recall_score(y_true, y_pred, average="macro"),
        metrics.recall_score(y_true, y_pred, average="weighted")
    ]
    # except:
    #    return 0.0


def evaluate_roc_auc(y_true, y_pred):
    # try:
    return 0
    # except:
    #    return 0


def full_evaluation(y_true, y_pred):
    print("================================")
    print("CLASSIFIER EVALUATION")

    print("Accuracy:         ", evaluate_accuracy(y_true, y_pred))
    print("Balanced Acc:     ", evaluate_balanced_accuracy(y_true, y_pred))
    print("F1-Score:         ", evaluate_f_score(y_true, y_pred))
    print("Precision:        ", evaluate_precision(y_true, y_pred))
    print("Recall:           ", evaluate_recall(y_true, y_pred))
    print("ROC AUC:          ", evaluate_roc_auc(y_true, y_pred))
    print("Confusion Matrix\n", confusion_matrix(y_true, y_pred))
    print("Full report:    ")

    full_report = classification_report(y_true, y_pred)

    print(full_report)


def evaluate_classifier(y_true, y_pred, results_dict):
    acc = evaluate_accuracy(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average="macro")
    conf_mat = confusion_matrix(y_true, y_pred)

    results_dict["acc"].append(acc)
    results_dict["f1"].append(f1)
    results_dict["conf_mat"].append(conf_mat.tolist())
