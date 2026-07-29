import torch 
from typing import Dict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

from utils import config_utils


def show_accuracy(predictions: torch.Tensor, targets: torch.Tensor):
    accuracy = accuracy_score(targets, predictions)
    print("Accuracy:", accuracy)


def analyse_outputs(probabilities: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
    assert predictions.keys() == targets.keys()
    for k in predictions.keys():
        print('Accuracy for', k)
        show_accuracy(predictions[k], targets[k])
        print()

def get_metrics(probabilities: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
    accuracy = accuracy_score(targets, predictions)
    weighted_precision = precision_score(targets, predictions, average="weighted")
    macro_precision = precision_score(targets, predictions, average="macro")
    micro_precision = precision_score(targets, predictions, average="micro")
    weighted_recall = recall_score(targets, predictions, average="weighted")
    macro_recall = recall_score(targets, predictions, average="macro")
    micro_recall = recall_score(targets, predictions, average="micro")
    f1 = f1_score(targets, predictions, average="weighted")
    weighted_f1 = f1_score(targets, predictions, average="weighted")
    micro_f1 = f1_score(targets, predictions, average="micro")
    macro_f1 = f1_score(targets, predictions, average="macro")
    confusion_mat = confusion_matrix(targets, predictions)
    return {
        "accuracy": accuracy,
        "weighted_precision": weighted_precision,
        "macro_precision": macro_precision,
        "micro_precision": micro_precision,
        "weighted_recall": weighted_recall,
        "macro_recall": macro_recall,
        "micro_recall": micro_recall,
        "f1": f1,
        "weighted_f1": weighted_f1,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "confusion_mat": confusion_mat
    }