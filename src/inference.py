"""
Inference Module for Credit Card Fraud Detection
Loads saved model, scaler, and threshold configuration to generate predictions.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
    'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
    'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Hour'
]

class FraudDetector:
    """
    Fraud Detection Inference Service.
    Loads serialized model artifacts and performs fraud scoring.
    """
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "final_model.joblib")
        self.scaler_path = os.path.join(model_dir, "scaler.joblib")
        self.config_path = os.path.join(model_dir, "model_config.json")
        
        # Load artifacts
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Model artifacts not found in {model_dir}. Please run training first.")
            
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        
        with open(self.config_path, "r") as f:
            self.config = json.load(f)
            
        self.threshold = float(self.config.get("selected_threshold", 0.5))
        self.feature_names = self.config.get("feature_names", FEATURE_COLUMNS)
        self.model_name = self.config.get("model_name", "XGBoost Classifier")
        
    def preprocess_dataframe(self, df):
        """
        Preprocesses a raw DataFrame (adds Hour if missing, orders columns, scales features).
        """
        df_proc = df.copy()
        
        # Calculate Hour from Time if not already provided
        if "Hour" not in df_proc.columns and "Time" in df_proc.columns:
            df_proc["Hour"] = ((df_proc["Time"] // 3600) % 24).astype(int)
            
        # Ensure all features exist
        for col in self.feature_names:
            if col not in df_proc.columns:
                df_proc[col] = 0.0
                
        # Align columns
        X = df_proc[self.feature_names]
        
        # Scale
        X_scaled = self.scaler.transform(X)
        return X_scaled
        
    def predict_proba(self, data):
        """
        Calculates fraud probability.
        Supports dict, Series, and DataFrame input.
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.Series):
            df = data.to_frame().T
        else:
            df = data
            
        X_scaled = self.preprocess_dataframe(df)
        probas = self.model.predict_proba(X_scaled)[:, 1]
        return probas
        
    def predict(self, data, custom_threshold=None):
        """
        Predicts fraud labels and probabilities.
        Supports:
        - dict: single transaction
        - pandas Series: single transaction
        - pandas DataFrame: batch transactions
        """
        threshold = (
            custom_threshold
            if custom_threshold is not None
            else self.threshold
        )

        # Convert Series to DataFrame
        is_single = isinstance(data, (dict, pd.Series))

        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.Series):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        # Get probabilities
        probas = self.predict_proba(df)

        predictions = (probas >= threshold).astype(int)

        # Single transaction
        if is_single:
            prob = float(probas[0])
            pred = int(predictions[0])

            return {
                "fraud_probability": round(prob, 4),
                "prediction": pred,
                "label": "Fraud" if pred == 1 else "Legitimate",
                "risk_level": (
                    "High Risk (Fraud)"
                    if pred == 1
                    else "Low Risk (Legitimate)"
                ),
                "threshold_applied": threshold
            }

        # Batch transactions
        results_df = data.copy()
        results_df["Fraud_Probability"] = np.round(probas, 4)
        results_df["Prediction"] = predictions
        results_df["Label"] = np.where(
            predictions == 1,
            "Fraud",
            "Legitimate"
        )

        return results_df
