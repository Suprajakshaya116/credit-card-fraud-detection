# Credit Card Fraud Detection - Machine Learning Project

End-to-End Machine Learning Project for Credit Card Fraud Detection with interactive Streamlit deployment.

---

## 📌 Problem Statement & Project Overview

Credit card fraud detection is a critical application of machine learning in financial cybersecurity. In transaction datasets, fraudulent activity accounts for a tiny fraction of total activity (less than 0.2%). This extreme class imbalance creates a fundamental challenge: standard algorithms optimizing for overall accuracy often achieve high scores simply by classifying every transaction as legitimate, failing to detect actual fraud cases (False Negatives).

This project implements a complete, structured machine learning workflow:
- **Exploratory Data Analysis (EDA):** Feature distribution, correlation, and fraud patterns.
- **Data Cleaning & Preprocessing:** Duplicate removal, feature engineering (`Hour`), and data leakage prevention.
- **Handling Class Imbalance:** Comparing baseline models, class weighting, SMOTE oversampling, and gradient boosting.
- **Model Evaluation & Comparison:** Evaluating candidate models using imbalanced classification metrics (PR-AUC, ROC-AUC, Precision, Recall, F1-Score).
- **Threshold Optimization:** Tuning decision thresholds to balance precision and recall.
- **Interactive Web Application:** Multi-page Streamlit application with left sidebar navigation for live transaction screening and model demonstration.

---

## 📊 Dataset Description

The project utilizes the European cardholder credit card fraud dataset:
- **Total Transactions (Raw):** 284,807 transactions.
- **Duplicate Records Removed:** 1,081 duplicates.
- **Cleaned Dataset:** 283,726 unique transactions.
- **Class Distribution:**
  - **Legitimate (`Class = 0`):** 283,253 transactions (~99.83%).
  - **Fraudulent (`Class = 1`):** 473 transactions (~0.17%).
- **Imbalance Ratio:** Approximately $1 : 577$.
- **Features (31 input columns):**
  - `Time`: Elapsed seconds from the first transaction in the dataset.
  - `Hour`: Feature-engineered hour of the day ($0\text{--}23$).
  - `Amount`: Monetary transaction amount.
  - `V1` to `V28`: Principal Component Analysis (PCA) transformed numerical features (anonymized for confidentiality).

---

## ⚙️ Preprocessing & Data Pipeline

1. **Deduplication:** Removed 1,081 exact duplicate records to prevent data leakage and inflated evaluation scores.
2. **Feature Engineering:** Extracted `Hour = ((Time // 3600) % 24)` to capture diurnal transaction patterns.
3. **Stratified Splitting (80/20):** Stratified split preserving the exact class proportion:
   - **Training Set:** 226,980 transactions (378 fraud, 226,602 legitimate).
   - **Test Set:** 56,746 transactions (95 fraud, 56,651 legitimate).
4. **Zero-Leakage Scaling:** `StandardScaler` was fitted **strictly on the training partition** and then applied to transform the test set and new inference inputs.

---

## 🧪 Candidate Models Evaluated

Five modeling approaches were evaluated on the identical test partition:

1. **Baseline Logistic Regression:** Standard linear classifier without class weighting.
2. **Class-Weighted Logistic Regression:** Linear classifier with inverse-frequency class weights (`class_weight='balanced'`).
3. **SMOTE + Logistic Regression:** Synthetic Minority Over-sampling Technique applied to the training set followed by logistic regression.
4. **XGBoost Classifier:** Regularized gradient boosted decision trees (`n_estimators=200`, `max_depth=5`, `learning_rate=0.08`, `subsample=0.8`, `colsample_bytree=0.8`).
5. **Soft-Voting Ensemble:** Weighted probability combination of XGBoost (0.85) and Baseline Logistic Regression (0.15).

---

## 📈 Model Performance & Comparison

All models were evaluated on the **untouched test set** (56,746 transactions with 95 actual fraud cases):

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | TP | FP | TN | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Logistic Regression** | 84.85% | 58.95% | 69.57% | 0.9562 | 0.6887 | 56 | 10 | 56,641 | 39 |
| **Class-Weighted Logistic Regression** | 5.55% | 87.37% | 10.43% | 0.9685 | 0.6683 | 83 | 1,413 | 55,238 | 12 |
| **SMOTE + Logistic Regression** | 5.14% | 87.37% | 9.71% | 0.9663 | 0.6785 | 83 | 1,532 | 55,119 | 12 |
| **XGBoost Classifier (Threshold = 0.50)** | 97.30% | 75.79% | 85.21% | **0.9745** | **0.8346** | 72 | **2** | 56,649 | 23 |
| **XGBoost Classifier (Tuned Thresh = 0.22)** | **97.33%** | 76.84% | **85.88%** | **0.9745** | **0.8346** | **73** | **2** | 56,649 | **22** |
| **Soft-Voting Ensemble (Tuned Thresh = 0.22)** | **97.33%** | 76.84% | **85.88%** | 0.9716 | 0.8246 | **73** | **2** | 56,649 | **22** |

### Why Accuracy Alone is Insufficient
In a dataset where 99.83% of samples are legitimate, a trivial model predicting "Legitimate" for every transaction achieves 99.83% accuracy while catching 0% of fraud. Therefore, the primary evaluation criteria are:
- **Precision:** Minimizes false alarms (False Positives) that inconvenience legitimate customers.
- **Recall:** Maximizes fraud detection rate (True Positives) to minimize financial losses.
- **PR-AUC (Precision-Recall Area Under Curve):** The most reliable metric for extreme class imbalance.
- **F1-Score:** Harmonic mean balancing precision and recall.

### Selected Champion Model
The **XGBoost Classifier at threshold 0.22** was selected as the champion model because it delivers:
- **97.33% Precision:** Only 2 false alarms out of 56,651 legitimate test transactions.
- **76.84% Recall:** Successfully catches 73 out of 95 fraud cases on the test set.
- **0.8588 F1-Score** and **0.8346 PR-AUC**.

---

## 🚀 Streamlit Web Application

The interactive web application (`app.py`) provides a clean, multi-page user interface with a **Left Sidebar Navigation**:

```
────────────────────────────
💳 Fraud Detection
Machine Learning Project
────────────────────────────
(o) 🏠 Home
( ) 🔍 Fraud Detection
( ) ⚙️ Model Information
( ) 🏗️ Project Architecture
( ) 📊 Model Performance
────────────────────────────
```

### Application Pages:
1. **🏠 Home:** Project title, overview, problem description, and summary of task, model, dataset, and features.
2. **🔍 Fraud Detection:** Live prediction interface:
   - Sample transaction selector (pre-loaded with verified legitimate and fraudulent cases).
   - Time and Amount input fields.
   - Expandable PCA features section (`V1–V28`) for viewing or editing.
   - **Predict Transaction** button.
   - Clear output display showing **Fraud Probability**, **Decision Threshold**, **Risk Level**, and **Prediction Badge** (`✅ LEGITIMATE` or `🚨 FRAUDULENT`).
3. **⚙️ Model Information:** Displays dynamic metadata (model type, binary task, dataset, feature list, scaling method, and dynamically loaded threshold `0.22`).
4. **🏗️ Project Architecture:** Visual pipeline flow and concise repository structure diagram.
5. **📊 Model Performance:** Displays the 5 verified evaluation metrics (`ROC-AUC`, `PR-AUC`, `Precision`, `Recall`, `F1-Score`) with metric cards and technical notes.

---

## 📁 Project Directory Structure

```text
credit_card/
├── data/
│   ├── .gitkeep                       # Keeps data directory in Git
│   └── creditcard.csv                 # Raw dataset (download from Kaggle)
├── notebooks/
│   ├── 01_EDA.ipynb                   # Exploratory Data Analysis & visual insights
│   ├── 02_Preprocessing_and_Modeling.ipynb # Preprocessing, baseline LR & SMOTE
│   └── 03_Advanced_Models.ipynb       # XGBoost, Threshold Tuning, Ensembling
├── models/
│   ├── final_model.joblib             # Serialized XGBoost champion model
│   ├── scaler.joblib                  # Fitted StandardScaler
│   ├── baseline_lr.joblib             # Baseline Logistic Regression
│   ├── xgboost_model.joblib           # XGBoost model artifact
│   └── model_config.json              # Model configuration & verified metrics
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py          # Ingestion, deduplication, and scaling
│   ├── model_utils.py                 # Metric evaluation, threshold tuning, ensemble
│   ├── train_pipeline.py              # Full end-to-end training script
│   └── inference.py                   # FraudDetector inference service
├── reports/
│   ├── model_comparison.csv           # Model comparison table
│   ├── pr_roc_curves.png              # ROC and PR curves
│   ├── confusion_matrices.png         # Confusion matrices comparison
│   ├── threshold_tuning.png           # Threshold vs metrics curves
│   └── feature_importance.png         # Top feature importances
├── .gitignore                         # Git ignore rules (ignores virtualenv, cache, large CSV)
├── app.py                             # Streamlit web application with sidebar navigation
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
```

---

## 💻 How to Run the Project

### 1. Installation & Environment Setup
Clone the repository and install required dependencies:
```bash
git clone <repository-url>
cd credit_card
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application
Run the Streamlit app locally:
```bash
streamlit run app.py
```
The application will automatically open in your default browser at `http://localhost:8501`.

### 3. (Optional) Run the Training Pipeline
To retrain models and regenerate evaluation artifacts and reports:
1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data/` folder.
2. Run the pipeline script:
```bash
python src/train_pipeline.py
```

---

## 🌐 Streamlit Cloud Deployment

This repository is pre-configured for one-click deployment on **Streamlit Community Cloud**:
1. Push the repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Select your repository, set the branch to `main`, and main file path to `app.py`.
4. Click **Deploy**.

> **Note on Model Artifacts:** The lightweight serialized model artifacts (`final_model.joblib`, `scaler.joblib`, `model_config.json`) total less than 400 KB and are included in the repository, allowing full inference on Streamlit Cloud without needing the 144 MB raw CSV file.
