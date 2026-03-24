# =============================================================================
#  utils.py  —  Helper functions shared by train_model.py and app.py
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH     = os.path.join("data", "loan_data.csv")
MODEL_DIR     = "models"
MODEL_PATH    = os.path.join(MODEL_DIR, "loan_model.pkl")
SCALER_PATH   = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH  = os.path.join(MODEL_DIR, "encoders.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")

# ── Column definitions ────────────────────────────────────────────────────────
CATEGORICAL_COLS = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
NUMERIC_COLS     = ["LoanAmount", "Loan_Amount_Term", "Credit_History"]
SCALE_COLS       = ["LoanAmount", "Loan_Amount_Term", "TotalIncome"]
TARGET_COL       = "Loan_Status"


def load_raw_data():
    """Load the raw CSV dataset."""
    df = pd.read_csv(DATA_PATH)
    print(f"[utils] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def fill_missing(df):
    """
    Fill missing values:
      - Numeric columns  → median
      - Categorical cols → mode
    """
    for col in NUMERIC_COLS:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    for col in CATEGORICAL_COLS:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)

    return df


def engineer_features(df):
    """
    Create TotalIncome feature and drop original income columns.
    Also drops Loan_ID if present (not useful for prediction).
    """
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])

    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df = df.drop(columns=["ApplicantIncome", "CoapplicantIncome"])
    return df


def encode_categoricals(df, fit=True, encoders=None):
    """
    Encode all categorical columns using LabelEncoder.

    Parameters
    ----------
    df       : DataFrame to encode
    fit      : If True, fit new encoders (training). If False, use existing (inference).
    encoders : dict of {col: LabelEncoder} — required when fit=False

    Returns
    -------
    df, encoders
    """
    if fit:
        encoders = {}
        all_cat_cols = CATEGORICAL_COLS + [TARGET_COL]
        for col in all_cat_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                encoders[col] = le
    else:
        # Inference mode — use saved encoders, only for input features
        for col in CATEGORICAL_COLS:
            if col in df.columns and col in encoders:
                df[col] = encoders[col].transform(df[col])

    return df, encoders


def scale_numerics(df, fit=True, scaler=None):
    """
    Apply StandardScaler to numeric columns.

    Parameters
    ----------
    df     : DataFrame
    fit    : If True, fit new scaler (training). If False, use existing (inference).
    scaler : Fitted StandardScaler — required when fit=False

    Returns
    -------
    df, scaler
    """
    if fit:
        scaler = StandardScaler()
        df[SCALE_COLS] = scaler.fit_transform(df[SCALE_COLS])
    else:
        df[SCALE_COLS] = scaler.transform(df[SCALE_COLS])

    return df, scaler


def generate_explanation(shap_vals, feature_names, prediction):
    """
    Produce a plain-English sentence explaining the loan decision.

    Parameters
    ----------
    shap_vals     : 1-D array of SHAP values for a single prediction
    feature_names : list of feature names
    prediction    : int, 0 = Rejected, 1 = Approved

    Returns
    -------
    str — human-readable explanation
    """
    # Sort features by absolute SHAP value (most impactful first)
    pairs = sorted(
        zip(feature_names, shap_vals),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    # Separate positive (approval) and negative (rejection) contributors
    positive = [f.replace("_", " ").lower() for f, v in pairs if v > 0][:2]
    negative = [f.replace("_", " ").lower() for f, v in pairs if v < 0][:2]

    if prediction == 0:  # Rejected
        reasons = ", ".join(negative) if negative else "multiple weak factors"
        return (
            f"Loan REJECTED. "
            f"The main factors working against this application: {reasons}. "
            f"Improving credit history and increasing income may help re-qualify."
        )
    else:  # Approved
        reasons = ", ".join(positive) if positive else "strong overall profile"
        return (
            f"Loan APPROVED. "
            f"Key strengths of this application: {reasons}."
        )
