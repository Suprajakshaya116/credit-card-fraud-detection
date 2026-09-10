"""
Model Training, Evaluation, and Ensembling Utilities for Credit Card Fraud Detection
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

def evaluate_model_performance(y_true, y_pred, y_prob=None, model_name="Model"):
    """
    Computes standard evaluation metrics for imbalanced classification.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    roc_auc = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    pr_auc = average_precision_score(y_true, y_prob) if y_prob is not None else np.nan
    
    return {
        "Model": model_name,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4) if not np.isnan(roc_auc) else None,
        "PR-AUC": round(pr_auc, 4) if not np.isnan(pr_auc) else None,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn)
    }

def tune_classification_threshold(y_true, y_prob, start=0.05, stop=0.99, step=0.01):
    """
    Scans a range of thresholds and computes Precision, Recall, and F1 at each.
    """
    thresholds = np.arange(start, stop + step / 2, step)
    results = []
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        results.append({
            "Threshold": round(float(thresh), 3),
            "Precision": round(float(p), 4),
            "Recall": round(float(r), 4),
            "F1-Score": round(float(f), 4),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn)
        })
        
    df_results = pd.DataFrame(results)
    best_f1_idx = df_results["F1-Score"].idxmax()
    best_row = df_results.loc[best_f1_idx]
    
    return df_results, best_row

class SoftVotingEnsemble:
    """
    Simple probability-based weighted ensemble for binary classification.
    """
    def __init__(self, models, weights=None):
        self.models = models  # List of tuples: [('name', model), ...]
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]
            
    def predict_proba(self, X):
        """
        Returns weighted probability matrix (N, 2).
        """
        ensemble_prob = np.zeros((X.shape[0], 2))
        for (name, model), w in zip(self.models, self.weights):
            prob = model.predict_proba(X)
            ensemble_prob += w * prob
        return ensemble_prob
        
    def predict(self, X, threshold=0.5):
        prob_pos = self.predict_proba(X)[:, 1]
        return (prob_pos >= threshold).astype(int)
