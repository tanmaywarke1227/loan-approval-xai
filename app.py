# =============================================================================
#  app.py  —  Streamlit Web App: Explainable Loan Approval System
#
#  Usage (in VS Code terminal — AFTER running train_model.py):
#      streamlit run app.py
#
#  Opens automatically at: http://localhost:8501
# =============================================================================

import os
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — required for Streamlit
import matplotlib.pyplot as plt
import shap
import streamlit as st

warnings.filterwarnings("ignore")

# Import shared paths and helper functions from utils.py
from utils import (
    fill_missing,
    engineer_features,
    encode_categoricals,
    scale_numerics,
    generate_explanation,
    MODEL_PATH,
    SCALER_PATH,
    ENCODER_PATH,
    FEATURES_PATH,
    CATEGORICAL_COLS,
    SCALE_COLS,
)


# =============================================================================
#  PAGE CONFIGURATION  (must be the first Streamlit call)
# =============================================================================
st.set_page_config(
    page_title="Loan Approval System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
#  LOAD SAVED MODEL ARTIFACTS  (cached so they load only once)
# =============================================================================
@st.cache_resource
def load_artifacts():
    """Train model if not present, then load artifacts."""
    if not os.path.exists(MODEL_PATH):
        import subprocess
        subprocess.run(["python", "train_model.py"], check=True)
    model    = joblib.load(MODEL_PATH)
    scaler   = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODER_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, scaler, encoders, features


model, scaler, encoders, feature_names = load_artifacts()


# =============================================================================
#  HEADER
# =============================================================================
st.title("🏦 Explainable Loan Approval System")
st.markdown(
    "Enter applicant details in the **sidebar** and click **Predict** "
    "to see the loan decision with an AI-powered explanation."
)
st.divider()


# =============================================================================
#  CHECK IF MODEL EXISTS
# =============================================================================
if model is None:
    st.error(
        "⚠️  Model not found. "
        "Please run `python train_model.py` in your VS Code terminal first."
    )
    st.stop()


# =============================================================================
#  SIDEBAR — APPLICANT INPUT FORM
# =============================================================================
st.sidebar.header("📋 Applicant Details")
st.sidebar.markdown("Fill in all fields, then click **Predict**.")

# ── Categorical inputs ────────────────────────────────────────────────────────
gender     = st.sidebar.selectbox("Gender",        ["Male", "Female"])
married    = st.sidebar.selectbox("Married",       ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents",    ["0", "1", "2", "3+"])
education  = st.sidebar.selectbox("Education",     ["Graduate", "Not Graduate"])
self_emp   = st.sidebar.selectbox("Self Employed", ["No", "Yes"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
# ── Numeric inputs ────────────────────────────────────────────────────────────
app_income   = st.sidebar.number_input(
    "Applicant Monthly Income (₹)",
    min_value=0, max_value=200_000, value=5_000, step=500
)
coapp_income = st.sidebar.number_input(
    "Co-applicant Monthly Income (₹)",
    min_value=0, max_value=100_000, value=0, step=500
)
loan_amount  = st.sidebar.slider(
    "Loan Amount (₹ thousands)",
    min_value=10, max_value=700, value=150, step=5
)
loan_term    = st.sidebar.selectbox(
    "Loan Term (months)",
    [60, 120, 180, 240, 360, 480],
    index=4   # default 360 months
)
credit_hist  = st.sidebar.radio(
    "Credit History",
    ["Good (1) — meets guidelines", "Poor (0) — does not meet guidelines"]
)

# Parse credit history to binary
credit_history_val = 1 if credit_hist.startswith("Good") else 0

st.sidebar.divider()
predict_btn = st.sidebar.button("🔍 Predict Loan Eligibility", use_container_width=True)


# =============================================================================
#  PREPROCESSING FUNCTION (for a single applicant)
# =============================================================================
def preprocess_single_applicant():
    """
    Build a single-row DataFrame from sidebar inputs,
    apply the same preprocessing pipeline used in training,
    and return the processed row ready for prediction.
    """
    # Step 1 — Build raw single-row DataFrame
    raw = pd.DataFrame([{
        "Gender":           gender,
        "Married":          married,
        "Dependents":       dependents,
        "Education":        education,
        "Self_Employed":    self_emp,
        "Property_Area":    property_area,
        "ApplicantIncome":  app_income,
        "CoapplicantIncome": coapp_income,
        "LoanAmount":       loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History":   credit_history_val,
    }])

    # Step 2 — Engineer TotalIncome (mirrors training pipeline)
    raw["TotalIncome"] = raw["ApplicantIncome"] + raw["CoapplicantIncome"]
    raw = raw.drop(columns=["ApplicantIncome", "CoapplicantIncome"])

    # Step 3 — Encode categoricals using saved encoders (fit=False)
    raw, _ = encode_categoricals(raw, fit=False, encoders=encoders)

    # Step 4 — Scale numerics using saved scaler (fit=False)
    raw, _ = scale_numerics(raw, fit=False, scaler=scaler)

    # Step 5 — Reorder columns to exactly match training order
    return raw[feature_names]


# =============================================================================
#  PREDICTION + EXPLANATION  (runs when Predict button is clicked)
# =============================================================================
if predict_btn:
    with st.spinner("Analyzing application..."):

        # ── Preprocess ──────────────────────────────────────────────────────
        input_df    = preprocess_single_applicant()
        prediction  = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])  # P(Approved)

        # ── Compute SHAP values ─────────────────────────────────────────────
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(input_df)
        # shap_vals is a list: [class_0_values, class_1_values]
        shap_approved = shap_vals[1][0]    # shape: (n_features,)
        base_value    = explainer.expected_value[1]

        # ── Text explanation ─────────────────────────────────────────────────
        explanation_text = generate_explanation(
            shap_approved, feature_names, prediction
        )

    # =========================================================================
    #  LAYOUT: Two-column result display
    # =========================================================================
    left_col, right_col = st.columns([1, 1.6], gap="large")

    # ── LEFT COLUMN: Decision + Key Metrics ──────────────────────────────────
    with left_col:
        st.subheader("Decision")

        if prediction == 1:
            st.success("## ✅  APPROVED")
        else:
            st.error("## ❌  REJECTED")

        # Confidence metrics
        m1, m2 = st.columns(2)
        m1.metric("Approval Probability", f"{probability * 100:.1f}%")
        m2.metric("Risk Score",           f"{(1 - probability) * 100:.1f}%")

        st.divider()

        # Plain-language explanation
        st.subheader("Why this decision?")
        if prediction == 1:
            st.success(explanation_text)
        else:
            st.error(explanation_text)

        st.divider()

        # Applicant summary table
        st.subheader("Applicant Summary")
        summary = {
            "Gender":        gender,
            "Married":       married,
            "Dependents":    dependents,
            "Education":     education,
            "Self Employed": self_emp,
            "Applicant Income": f"₹{app_income:,}",
            "Co-app Income":    f"₹{coapp_income:,}",
            "Total Income":     f"₹{app_income + coapp_income:,}",
            "Loan Amount":      f"₹{loan_amount * 1000:,}",
            "Loan Term":        f"{loan_term} months",
            "Credit History":   "Good" if credit_history_val == 1 else "Poor",
        }
        st.table(pd.DataFrame.from_dict(summary, orient="index", columns=["Value"]))

    # ── RIGHT COLUMN: SHAP Visualizations ────────────────────────────────────
    with right_col:
        st.subheader("SHAP Explanation — Feature Contributions")
        st.markdown(
            "Each bar shows how much a feature **pushed** the prediction toward "
            "Approved (blue/positive) or Rejected (red/negative)."
        )

        # ── SHAP Waterfall Plot ──────────────────────────────────────────────
        shap_explanation = shap.Explanation(
            values=shap_approved,
            base_values=base_value,
            data=input_df.values[0],
            feature_names=feature_names,
        )

        fig_wf, ax_wf = plt.subplots(figsize=(8, 4.5))
        shap.waterfall_plot(shap_explanation, show=False, max_display=10)
        plt.title("Individual Prediction — SHAP Waterfall", fontsize=12, pad=10)
        plt.tight_layout()
        st.pyplot(fig_wf)
        plt.close(fig_wf)

        st.divider()

        # ── SHAP Bar Chart (feature importances for this prediction) ─────────
        st.subheader("Feature Impact (this prediction)")

        feat_shap_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": shap_approved,
        }).sort_values("SHAP Value", key=abs, ascending=True)

        colors = [
            "#2196F3" if v > 0 else "#F44336"
            for v in feat_shap_df["SHAP Value"]
        ]

        fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
        ax_bar.barh(
            feat_shap_df["Feature"],
            feat_shap_df["SHAP Value"],
            color=colors
        )
        ax_bar.axvline(0, color="black", linewidth=0.8)
        ax_bar.set_xlabel("SHAP value  (positive = pushes toward Approved)")
        ax_bar.set_title("Feature Contributions to This Prediction")
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)


# =============================================================================
#  GLOBAL FEATURE IMPORTANCE  (always visible below — from saved PNG)
# =============================================================================
st.divider()
st.subheader("📊 Global Feature Importance (across all training data)")
st.markdown(
    "This chart shows which features matter most **on average** "
    "for all loan applications in the dataset."
)

fi_path = os.path.join("models", "feature_importance.png")
if os.path.exists(fi_path):
    st.image(fi_path, use_column_width=True)
else:
    st.info("Run `python train_model.py` to generate the feature importance chart.")

# Footer
st.divider()
st.caption(
    "Explainable Loan Approval System · Random Forest + SHAP · "
    "Built with Streamlit · Micro-Project"
)
