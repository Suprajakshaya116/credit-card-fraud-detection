"""
Data Preprocessing Module for Credit Card Fraud Detection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
    'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
    'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Hour'
]

RANDOM_STATE = 42

def load_and_preprocess_data(csv_path="data/creditcard.csv", test_size=0.20, random_state=RANDOM_STATE):
    """
    Loads raw credit card dataset, cleans duplicates, creates engineered features,
    and returns stratified train/test splits and fitted scaler.
    """
    df = pd.read_csv(csv_path)
    
    # 1. Remove duplicate records
    df = df.drop_duplicates().reset_index(drop=True)
    
    
    # 2. Feature Engineering: extract Hour of day from Time (seconds)
    df["Hour"] = ((df["Time"] // 3600) % 24).astype(int)
    
    # 3. Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]
    
    # Ensure exact column order
    X = X[FEATURE_COLUMNS]
    
    # 4. Stratified Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # 5. Fit Scaler strictly on X_train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "feature_names": FEATURE_COLUMNS
    }

def prepare_single_sample(sample_dict, scaler):
    """
    Prepares a single dictionary or series for model inference.
    """
    df_sample = pd.DataFrame([sample_dict])
    
    # If Hour is not provided, calculate from Time
    if "Hour" not in df_sample.columns and "Time" in df_sample.columns:
        df_sample["Hour"] = ((df_sample["Time"] // 3600) % 24).astype(int)
    
    # Ensure all feature columns exist and are ordered
    df_sample = df_sample[FEATURE_COLUMNS]
    
    # Transform using fitted scaler
    scaled_sample = scaler.transform(df_sample)
    return scaled_sample
