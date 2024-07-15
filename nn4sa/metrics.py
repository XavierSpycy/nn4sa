import torch

def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return (y_true == y_pred).sum().item() / len(y_true)

def precision(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    return tp / (tp + fp)

def recall(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()
    return tp / (tp + fn)