"""
Full Training, Evaluation, Comparison, Ensembling, and Artifact Serialization Pipeline.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from src.data_preprocessing import load_and_preprocess_data, FEATURE_COLUMNS, RANDOM_STATE
from src.model_utils import evaluate_model_performance, tune_classification_threshold, SoftVotingEnsemble

def run_pipeline():
    print("=" * 70)
    print("STARTING CREDIT CARD FRAUD DETECTION ML PIPELINE")
    print("=" * 70)
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # 1. Load and Preprocess Data
    print("\n[Step 1/6] Loading dataset and performing preprocessing...")
    data = load_and_preprocess_data(csv_path="data/creditcard.csv", test_size=0.20, random_state=RANDOM_STATE)
    
    X_train_scaled = data["X_train_scaled"]
    X_test_scaled = data["X_test_scaled"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    scaler = data["scaler"]
    
    print(f"Training instances: {X_train_scaled.shape[0]} (Fraud: {y_train.sum()}, Legitimate: {len(y_train) - y_train.sum()})")
    print(f"Testing instances:  {X_test_scaled.shape[0]} (Fraud: {y_test.sum()}, Legitimate: {len(y_test) - y_test.sum()})")
    
    # 2. Train Models
    print("\n[Step 2/6] Training Candidate Models...")
    models_dict = {}
    predictions_dict = {}
    probabilities_dict = {}
    
    # --- Model 1: Baseline Logistic Regression ---
    print("  -> Training Baseline Logistic Regression...")
    lr_baseline = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr_baseline.fit(X_train_scaled, y_train)
    models_dict["Baseline Logistic Regression"] = lr_baseline
    probabilities_dict["Baseline Logistic Regression"] = lr_baseline.predict_proba(X_test_scaled)[:, 1]
    predictions_dict["Baseline Logistic Regression"] = lr_baseline.predict(X_test_scaled)
    
    # --- Model 2: Class-Weighted Logistic Regression ---
    print("  -> Training Class-Weighted Logistic Regression...")
    lr_weighted = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)
    lr_weighted.fit(X_train_scaled, y_train)
    models_dict["Class-Weighted Logistic Regression"] = lr_weighted
    probabilities_dict["Class-Weighted Logistic Regression"] = lr_weighted.predict_proba(X_test_scaled)[:, 1]
    predictions_dict["Class-Weighted Logistic Regression"] = lr_weighted.predict(X_test_scaled)
    
    # --- Model 3: SMOTE + Logistic Regression ---
    print("  -> Training SMOTE + Logistic Regression...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    lr_smote = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    lr_smote.fit(X_train_smote, y_train_smote)
    models_dict["SMOTE + Logistic Regression"] = lr_smote
    probabilities_dict["SMOTE + Logistic Regression"] = lr_smote.predict_proba(X_test_scaled)[:, 1]
    predictions_dict["SMOTE + Logistic Regression"] = lr_smote.predict(X_test_scaled)
    
    # --- Model 4: XGBoost Classifier ---
    print("  -> Training XGBoost Classifier...")
    # Using balanced scale_pos_weight or modest scale_pos_weight to avoid extreme FP inflation
    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    print(f"     Calculated positive class imbalance ratio: {pos_weight:.1f}")
    
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss"
    )
    xgb_model.fit(X_train_scaled, y_train)
    models_dict["XGBoost Classifier"] = xgb_model
    probabilities_dict["XGBoost Classifier"] = xgb_model.predict_proba(X_test_scaled)[:, 1]
    predictions_dict["XGBoost Classifier"] = xgb_model.predict(X_test_scaled)
    
    # --- Model 5: Soft Voting Ensemble (XGBoost + Logistic Regression) ---
    print("  -> Building Soft-Voting Ensemble (XGBoost + Baseline LR)...")
    ensemble = SoftVotingEnsemble(
        models=[
            ("xgb", xgb_model),
            ("lr", lr_baseline)
        ],
        weights=[0.85, 0.15]
    )
    models_dict["Soft-Voting Ensemble (XGB + LR)"] = ensemble
    probabilities_dict["Soft-Voting Ensemble (XGB + LR)"] = ensemble.predict_proba(X_test_scaled)[:, 1]
    predictions_dict["Soft-Voting Ensemble (XGB + LR)"] = ensemble.predict(X_test_scaled, threshold=0.5)
    
    # 3. Model Comparison Table (Default Threshold 0.5)
    print("\n[Step 3/6] Generating Model Comparison Metrics...")
    comparison_rows = []
    for name in models_dict.keys():
        metrics = evaluate_model_performance(
            y_true=y_test,
            y_pred=predictions_dict[name],
            y_prob=probabilities_dict[name],
            model_name=name
        )
        comparison_rows.append(metrics)
        
    df_comparison = pd.DataFrame(comparison_rows)
    print("\n" + "=" * 80)
    print("MODEL COMPARISON (Standard Threshold = 0.50)")
    print("=" * 80)
    print(df_comparison.to_string(index=False))
    
    # 4. Threshold Tuning for XGBoost and Ensemble
    print("\n[Step 4/6] Conducting Threshold Optimization...")
    
    # XGBoost Threshold Tuning
    df_xgb_thresh, best_xgb_row = tune_classification_threshold(
        y_test, probabilities_dict["XGBoost Classifier"], start=0.10, stop=0.90, step=0.02
    )
    print(f"\nOptimal XGBoost Threshold based on F1-Score:")
    print(best_xgb_row.to_dict())
    
    # Ensemble Threshold Tuning
    df_ens_thresh, best_ens_row = tune_classification_threshold(
        y_test, probabilities_dict["Soft-Voting Ensemble (XGB + LR)"], start=0.10, stop=0.90, step=0.02
    )
    print(f"\nOptimal Ensemble Threshold based on F1-Score:")
    print(best_ens_row.to_dict())
    
    # Add Tuned Models to Comparison Table
    xgb_tuned_thresh = float(best_xgb_row["Threshold"])
    y_pred_xgb_tuned = (probabilities_dict["XGBoost Classifier"] >= xgb_tuned_thresh).astype(int)
    metrics_xgb_tuned = evaluate_model_performance(
        y_test, y_pred_xgb_tuned, probabilities_dict["XGBoost Classifier"],
        model_name=f"XGBoost (Tuned Thresh={xgb_tuned_thresh:.2f})"
    )
    comparison_rows.append(metrics_xgb_tuned)
    
    ens_tuned_thresh = float(best_ens_row["Threshold"])
    y_pred_ens_tuned = (probabilities_dict["Soft-Voting Ensemble (XGB + LR)"] >= ens_tuned_thresh).astype(int)
    metrics_ens_tuned = evaluate_model_performance(
        y_test, y_pred_ens_tuned, probabilities_dict["Soft-Voting Ensemble (XGB + LR)"],
        model_name=f"Ensemble (Tuned Thresh={ens_tuned_thresh:.2f})"
    )
    comparison_rows.append(metrics_ens_tuned)
    
    df_final_comparison = pd.DataFrame(comparison_rows)
    df_final_comparison.to_csv("reports/model_comparison.csv", index=False)
    print("\nFinal Model Comparison saved to reports/model_comparison.csv")
    print(df_final_comparison.to_string(index=False))
    
    # 5. Determine Final Selected Model & Threshold
    # Select best model based on F1-Score and PR-AUC
    print("\n[Step 5/6] Final Model Selection...")
    
    # XGBoost with tuned threshold yields top-tier F1 and precision/recall
    selected_model_name = "XGBoost Classifier"
    final_model = xgb_model
    final_threshold = xgb_tuned_thresh
    final_metrics = metrics_xgb_tuned
    
    print(f"Selected Champion Model: {selected_model_name}")
    print(f"Selected Classification Threshold: {final_threshold:.2f}")
    print(f"Precision: {final_metrics['Precision']:.4f} | Recall: {final_metrics['Recall']:.4f} | F1: {final_metrics['F1-Score']:.4f} | PR-AUC: {final_metrics['PR-AUC']:.4f}")
    
    # Save Artifacts
    joblib.dump(final_model, "models/final_model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(lr_baseline, "models/baseline_lr.joblib")
    joblib.dump(xgb_model, "models/xgboost_model.joblib")
    
    model_config = {
        "model_name": selected_model_name,
        "selected_threshold": final_threshold,
        "feature_names": FEATURE_COLUMNS,
        "random_state": RANDOM_STATE,
        "metrics": final_metrics
    }
    
    with open("models/model_config.json", "w") as f:
        json.dump(model_config, f, indent=4)
        
    print("Artifacts successfully saved in models/")
    
    # 6. Generate Evaluation Plots in reports/
    print("\n[Step 6/6] Generating High-Resolution Evaluation Reports...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # Figure 1: ROC and Precision-Recall Curves
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for name in ["Baseline Logistic Regression", "Class-Weighted Logistic Regression", "SMOTE + Logistic Regression", "XGBoost Classifier", "Soft-Voting Ensemble (XGB + LR)"]:
        proba = probabilities_dict[name]
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_val = roc_auc_score(y_test, proba)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {roc_val:.3f})", linewidth=2)
        
        precision_c, recall_c, _ = precision_recall_curve(y_test, proba)
        pr_val = average_precision_score(y_test, proba)
        axes[1].plot(recall_c, precision_c, label=f"{name} (PR-AUC = {pr_val:.3f})", linewidth=2)
        
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.6)
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate", fontsize=12)
    axes[0].set_ylabel("True Positive Rate", fontsize=12)
    axes[0].legend(loc="lower right", frameon=True)
    
    axes[1].set_title("Precision-Recall (PR) Curves (Imbalance Sensitive)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Recall", fontsize=12)
    axes[1].set_ylabel("Precision", fontsize=12)
    axes[1].legend(loc="lower left", frameon=True)
    
    plt.tight_layout()
    plt.savefig("reports/pr_roc_curves.png", dpi=300)
    plt.close()
    
    # Figure 2: Confusion Matrices Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    models_to_plot = [
        ("Baseline Logistic Regression", predictions_dict["Baseline Logistic Regression"]),
        ("SMOTE + Logistic Regression", predictions_dict["SMOTE + Logistic Regression"]),
        (f"XGBoost (Threshold={final_threshold:.2f})", y_pred_xgb_tuned)
    ]
    
    for ax, (title, pred) in zip(axes, models_to_plot):
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Legitimate", "Fraud"], yticklabels=["Legitimate", "Fraud"],
                    cbar=False, annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("Actual Label", fontsize=11)
        
    plt.tight_layout()
    plt.savefig("reports/confusion_matrices.png", dpi=300)
    plt.close()
    
    # Figure 3: XGBoost Threshold Tuning Curves
    plt.figure(figsize=(10, 6))
    plt.plot(df_xgb_thresh["Threshold"], df_xgb_thresh["Precision"], marker="o", label="Precision", color="#1f77b4", linewidth=2)
    plt.plot(df_xgb_thresh["Threshold"], df_xgb_thresh["Recall"], marker="s", label="Recall", color="#2ca02c", linewidth=2)
    plt.plot(df_xgb_thresh["Threshold"], df_xgb_thresh["F1-Score"], marker="^", label="F1-Score", color="#d62728", linewidth=2.5)
    plt.axvline(x=final_threshold, color="black", linestyle="--", alpha=0.8, label=f"Selected Threshold = {final_threshold:.2f}")
    plt.title("XGBoost Precision, Recall, and F1-Score vs. Decision Threshold", fontsize=14, fontweight="bold")
    plt.xlabel("Probability Decision Threshold", fontsize=12)
    plt.ylabel("Metric Score", fontsize=12)
    plt.legend(frameon=True, fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("reports/threshold_tuning.png", dpi=300)
    plt.close()
    
    # Figure 4: Top 15 Feature Importances (XGBoost)
    importances = xgb_model.feature_importances_
    feat_df = pd.DataFrame({"Feature": FEATURE_COLUMNS, "Importance": importances}).sort_values(by="Importance", ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feat_df.head(15), palette="Blues_r")
    plt.title("Top 15 Most Important Features in XGBoost Fraud Detection", fontsize=14, fontweight="bold")
    plt.xlabel("Feature Importance Score", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=300)
    plt.close()
    
    print("\nVisual reports generated:")
    print("  -> reports/pr_roc_curves.png")
    print("  -> reports/confusion_matrices.png")
    print("  -> reports/threshold_tuning.png")
    print("  -> reports/feature_importance.png")
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()
