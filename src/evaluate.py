"""
Model Evaluation and Diagnostic Reporting Module
Computes classification metrics, confusion matrices, ROC curves,
and PR curves for imbalanced predictive maintenance.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

def evaluate_model(model, X_test, y_test, model_name=None, threshold=None):
    """
    Evaluates a trained ANN model on test data using standard and imbalanced metrics.
    """
    name = model_name or getattr(model, 'name', 'ANN Model')
    thresh = threshold if threshold is not None else getattr(model, 'best_threshold_', 0.5)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= thresh).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_test, probs)
    except Exception:
        roc_auc = 0.0

    try:
        pr_auc = average_precision_score(y_test, probs)
    except Exception:
        pr_auc = 0.0

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    n_features = len(model.selected_indices_) if getattr(model, 'selected_indices_', None) is not None else X_test.shape[1]

    metrics = {
        'Model': name,
        'Features Used': n_features,
        'Threshold': round(thresh, 3),
        'Accuracy': round(acc, 4),
        'F1-Score': round(f1, 4),
        'Recall (Sensitivity)': round(rec, 4),
        'Precision': round(prec, 4),
        'Specificity': round(specificity, 4),
        'ROC-AUC': round(roc_auc, 4),
        'PR-AUC (Avg Precision)': round(pr_auc, 4),
        'TP': int(tp),
        'FP': int(fp),
        'TN': int(tn),
        'FN': int(fn),
        'Confusion Matrix': cm.tolist()
    }
    return metrics

def compare_models(metrics_list):
    """
    Formats a list of evaluation metric dictionaries into a clean comparison DataFrame.
    """
    df = pd.DataFrame(metrics_list)
    cols_display = [
        'Model', 'Features Used', 'Threshold', 'F1-Score',
        'Recall (Sensitivity)', 'Precision', 'ROC-AUC', 'PR-AUC (Avg Precision)', 'Accuracy'
    ]
    available_cols = [c for c in cols_display if c in df.columns]
    return df[available_cols]

def get_roc_curve_data(model, X_test, y_test):
    """Returns FPR, TPR, and AUC for plotting ROC curves."""
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_val = roc_auc_score(y_test, probs)
    return fpr, tpr, auc_val

def get_pr_curve_data(model, X_test, y_test):
    """Returns Precision, Recall, and PR-AUC for plotting PR curves."""
    probs = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc_val = average_precision_score(y_test, probs)
    return recall, precision, pr_auc_val
