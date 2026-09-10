"""
Credit Card Fraud Detection - Streamlit Web Application
A clean, multi-page sidebar navigation interface for student ML project demonstration.
"""

import os
import sys
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference import FraudDetector

# -------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# Model Loader
# -------------------------------------------------------------
@st.cache_resource
def load_detector():
    """Loads the serialized FraudDetector inference pipeline."""
    try:
        return FraudDetector(model_dir=os.path.join(PROJECT_ROOT, "models"))
    except Exception as e:
        return None

detector = load_detector()

# -------------------------------------------------------------
# Real Sample Presets (from dataset)
# -------------------------------------------------------------
SAMPLE_TRANSACTIONS = {
    "Legitimate Transaction (Sample 1 - $149.62)": {
        "Time": 0.0, "Amount": 149.62,
        "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
        "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
        "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
        "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
        "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
        "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
        "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053
    },
    "Legitimate Transaction (Sample 2 - $2.69)": {
        "Time": 0.0, "Amount": 2.69,
        "V1": 1.191857, "V2": 0.266151, "V3": 0.166480, "V4": 0.448154,
        "V5": 0.060018, "V6": -0.082361, "V7": -0.078803, "V8": 0.085102,
        "V9": -0.255425, "V10": -0.166974, "V11": 1.612727, "V12": 1.065235,
        "V13": 0.489095, "V14": -0.143772, "V15": 0.635558, "V16": 0.463917,
        "V17": -0.114805, "V18": -0.183361, "V19": -0.145783, "V20": -0.069083,
        "V21": -0.225775, "V22": -0.638672, "V23": 0.101288, "V24": -0.339846,
        "V25": 0.167170, "V26": 0.125895, "V27": -0.008983, "V28": 0.014724
    },
    "Fraudulent Transaction (Sample 1 - Account Verification $0.00)": {
        "Time": 406.0, "Amount": 0.00,
        "V1": -2.312227, "V2": 1.951992, "V3": -1.609851, "V4": 3.997906,
        "V5": -0.522188, "V6": -1.426545, "V7": -2.537387, "V8": 1.391657,
        "V9": -2.770089, "V10": -2.772272, "V11": 3.202033, "V12": -2.899907,
        "V13": -0.595222, "V14": -4.289254, "V15": 0.389724, "V16": -1.140747,
        "V17": -2.830056, "V18": -0.016822, "V19": 0.416956, "V20": 0.126911,
        "V21": 0.517232, "V22": -0.035049, "V23": -0.465211, "V24": 0.320198,
        "V25": 0.044519, "V26": 0.177840, "V27": 0.261145, "V28": -0.143276
    },
    "Fraudulent Transaction (Sample 2 - High Value Fraud $529.00)": {
        "Time": 472.0, "Amount": 529.00,
        "V1": -3.043541, "V2": -3.157307, "V3": 1.088463, "V4": 2.288644,
        "V5": 1.359805, "V6": -1.064823, "V7": 0.325574, "V8": -0.067794,
        "V9": -0.270953, "V10": -0.838587, "V11": -0.414575, "V12": -0.503141,
        "V13": 0.676502, "V14": -1.692029, "V15": 2.000635, "V16": 0.666780,
        "V17": 0.599717, "V18": 1.725321, "V19": 0.283345, "V20": 2.102339,
        "V21": 0.661696, "V22": 0.435477, "V23": 1.375966, "V24": -0.293803,
        "V25": 0.279798, "V26": -0.145362, "V27": -0.252773, "V28": 0.035764
    }
}

# -------------------------------------------------------------
# Check Model Status
# -------------------------------------------------------------
if detector is None:
    st.error("⚠️ Failed to load model artifacts. Please ensure model files exist in the `models/` directory.")
    st.stop()

# -------------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------------
st.sidebar.markdown("## 💳 Fraud Detection")
st.sidebar.caption("Machine Learning Project")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    options=[
        "🏠 Home",
        "🔍 Fraud Detection",
        "⚙️ Model Information",
        "🏗️ Project Architecture",
        "📊 Model Performance"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption(f"**Model:** {detector.model_name}")
st.sidebar.caption(f"**Threshold:** {detector.threshold:.2f}")

# -------------------------------------------------------------
# Page 1: Home
# -------------------------------------------------------------
if page == "🏠 Home":
    st.title("💳 Credit Card Fraud Detection")
    st.write(
        "This project uses machine learning to detect fraudulent credit card transactions by analyzing "
        "transaction-level features and predicting whether a transaction is legitimate or fraudulent."
    )
    
    st.markdown("---")
    st.subheader("📌 Project Overview")
    st.markdown(f"• **Task:** Binary Classification")
    st.markdown(f"• **Model:** {detector.model_name}")
    st.markdown(f"• **Dataset:** Credit Card Fraud Detection")
    st.markdown(f"• **Features:** Time, Amount, Hour, V1–V28")

# -------------------------------------------------------------
# Page 2: Fraud Detection (Prediction Interface)
# -------------------------------------------------------------
elif page == "🔍 Fraud Detection":
    st.title("🔍 Fraud Detection")
    st.write("Select a sample transaction or enter custom details to predict whether it is legitimate or fraudulent.")
    
    # 1. Transaction Sample Selector
    st.subheader("Transaction Sample")
    sample_choice = st.selectbox(
        "Choose a sample transaction:",
        options=list(SAMPLE_TRANSACTIONS.keys()),
        index=0
    )
    selected_sample = SAMPLE_TRANSACTIONS[sample_choice]
    
    # 2. Transaction Details (Time & Amount)
    st.subheader("Transaction Details")
    col_time, col_amount = st.columns(2)
    with col_time:
        time_val = st.number_input(
            "Time (seconds)",
            value=float(selected_sample["Time"]),
            step=1.0,
            format="%.1f",
            key=f"{sample_choice}_time"
        )
    with col_amount:
        amount_val = st.number_input(
            "Amount ($)",
            value=float(selected_sample["Amount"]),
            min_value=0.0,
            step=1.0,
            format="%.2f",
            key=f"{sample_choice}_amount"
        )
        
    # 3. PCA Features Expander (V1 - V28)
    with st.expander("🔬 View / Edit PCA Features (V1–V28)", expanded=False):
        st.caption("Principal component features (V1–V28) obtained from PCA dimensionality reduction.")
        v_features = {}
        cols = st.columns(4)
        for i in range(1, 29):
            feat_name = f"V{i}"
            with cols[(i - 1) % 4]:
                v_features[feat_name] = st.number_input(
                    feat_name,
                    value=float(selected_sample[feat_name]),
                    format="%.6f",
                    key=f"{sample_choice}_{feat_name}"
                )
                
    # 4. Prediction Trigger & Results
    st.markdown("")
    if st.button("🔍 Predict Transaction", type="primary", use_container_width=True):
        # Prepare input payload for inference
        input_data = {
            "Time": time_val,
            "Amount": amount_val,
            **v_features
        }
        
        # Run prediction through existing inference service
        result = detector.predict(input_data)
        
        st.markdown("---")
        st.subheader("Prediction Result")
        
        prob_pct = result["fraud_probability"] * 100
        thresh_pct = result["threshold_applied"] * 100
        
        if result["prediction"] == 1:
            st.error("### 🚨 FRAUDULENT TRANSACTION")
        else:
            st.success("### ✅ LEGITIMATE TRANSACTION")
            
        col1, col2, col3 = st.columns(3)
        col1.metric("Fraud Probability", f"{prob_pct:.2f}%")
        col2.metric("Decision Threshold", f"{thresh_pct:.2f}%")
        col3.metric("Risk Level", result["risk_level"])

# -------------------------------------------------------------
# Page 3: Model Information
# -------------------------------------------------------------
elif page == "⚙️ Model Information":
    st.title("⚙️ Model Information")
    
    st.markdown(f"• **Model:** {detector.model_name}")
    st.markdown("• **Task:** Binary Classification")
    st.markdown("• **Dataset:** Credit Card Fraud Detection")
    st.markdown("• **Features:** Time, Amount, Hour, V1–V28")
    st.markdown("• **Preprocessing:** StandardScaler")
    st.markdown(f"• **Decision Threshold:** `{detector.threshold:.2f}`")
    
    st.markdown("---")
    st.write(
        "The model produces a fraud probability for each transaction. The probability is compared "
        "with the selected decision threshold to classify the transaction as fraudulent or legitimate."
    )

# -------------------------------------------------------------
# Page 4: Project Architecture
# -------------------------------------------------------------
elif page == "🏗️ Project Architecture":
    st.title("🏗️ Project Architecture")
    
    st.subheader("Pipeline Flow")
    st.code("""
Transaction Data
       ↓
Preprocessing
       ↓
Hour Feature Engineering
       ↓
StandardScaler
       ↓
XGBoost Classifier
       ↓
Fraud Probability
       ↓
Decision Threshold
       ↓
Fraud / Legitimate
""", language="text")
    
    st.subheader("Project Structure")
    st.code("""
credit_card/
├── app.py
├── src/
│   ├── inference.py
│   ├── data_preprocessing.py
│   ├── model_utils.py
│   └── train_pipeline.py
├── models/
│   ├── final_model.joblib
│   ├── scaler.joblib
│   └── model_config.json
├── notebooks/
└── reports/
""", language="text")

# -------------------------------------------------------------
# Page 5: Model Performance
# -------------------------------------------------------------
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    
    metrics = detector.config.get("metrics", {})
    roc_auc = float(metrics.get("ROC-AUC", 0.9745))
    pr_auc = float(metrics.get("PR-AUC", 0.8346))
    precision = float(metrics.get("Precision", 0.9733)) * 100
    recall = float(metrics.get("Recall", 0.7684)) * 100
    f1 = float(metrics.get("F1-Score", 0.8588)) * 100
    
    # 2 rows to avoid cramped layout
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    row1_col1.metric("ROC-AUC", f"{roc_auc:.4f}")
    row1_col2.metric("PR-AUC", f"{pr_auc:.4f}")
    row1_col3.metric("F1-Score", f"{f1:.2f}%")
    
    st.markdown("")
    row2_col1, row2_col2 = st.columns(2)
    row2_col1.metric("Precision", f"{precision:.2f}%")
    row2_col2.metric("Recall", f"{recall:.2f}%")
    
    st.markdown("---")
    st.write(
        "Fraud detection is a highly imbalanced classification problem, so precision, recall, "
        "PR-AUC and F1-score provide more useful information about model performance than accuracy alone."
    )
