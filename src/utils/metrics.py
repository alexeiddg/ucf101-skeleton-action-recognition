import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def compute_accuracy(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    return accuracy_score(y_true, y_pred)


def compute_precision(y_true, y_pred, average='macro'):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def compute_recall(y_true, y_pred, average='macro'):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def compute_f1(y_true, y_pred, average='macro'):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def compute_confusion_matrix(y_true, y_pred, normalize=None):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize is not None:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    return cm


def compute_metrics(y_true, y_pred, average='macro'):
    return {
        'accuracy': compute_accuracy(y_true, y_pred),
        'precision': compute_precision(y_true, y_pred, average=average),
        'recall': compute_recall(y_true, y_pred, average=average),
        'f1': compute_f1(y_true, y_pred, average=average)
    }