import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import warnings
warnings.filterwarnings("ignore")

from utils import (
    load_raw_data,
    fill_missing,
    engineer_features,
    encode_categoricals,
    scale_numerics,
    MODEL_DIR,
    MODEL_PATH,
    SCALER_PATH,
    ENCODER_PATH,
    FEATURES_PATH,
    TARGET_COL,
)



os.makedirs(MODEL_DIR, exist_ok=True)



print("=" * 60)
print("STEP 1 — Loading dataset")
print("=" * 60)
df = load_raw_data()
print(df.head())
print(f"\nMissing values:\n{df.isnull().sum()}\n")



print("=" * 60)
print("STEP 2 — Filling missing values")
print("=" * 60)
df = fill_missing(df)
print(f"Missing values after fill:\n{df.isnull().sum()}\n")



print("=" * 60)
print("STEP 3 — Feature engineering  (TotalIncome)")
print("=" * 60)
df = engineer_features(df)
print(f"Columns after engineering: {df.columns.tolist()}\n")



print("=" * 60)
print("STEP 4 — Encoding categorical features")
print("=" * 60)
df, encoders = encode_categoricals(df, fit=True)
joblib.dump(encoders, ENCODER_PATH)
print(f"Encoded columns: {list(encoders.keys())}")
print(f"Encoders saved to {ENCODER_PATH}\n")



print("=" * 60)
print("STEP 5 — Scaling numeric features")
print("=" * 60)
df, scaler = scale_numerics(df, fit=True)
joblib.dump(scaler, SCALER_PATH)
print(f"Scaler saved to {SCALER_PATH}\n")



print("=" * 60)
print("STEP 6 — Preparing X (features) and y (target)")
print("=" * 60)
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]


joblib.dump(X.columns.tolist(), FEATURES_PATH)
print(f"Feature names saved to {FEATURES_PATH}")
print(f"Feature list: {X.columns.tolist()}")
print(f"Target distribution:\n{y.value_counts()}\n")


print("=" * 60)
print("STEP 7 — Train / test split  (80 / 20, stratified)")
print("=" * 60)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y       
)
print(f"Training samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}\n")



print("=" * 60)
print("STEP 8 — Training Random Forest Classifier")
print("=" * 60)
model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=8,           
    min_samples_split=5,   
    min_samples_leaf=2,     
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)
print("Training complete.\n")



cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f}  ±  {cv_scores.std():.4f}\n")


print("=" * 60)
print("STEP 9 — Evaluation metrics")
print("=" * 60)
y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
print(f"Test Accuracy : {acc:.4f}  ({acc*100:.1f}%)\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))



cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Rejected", "Approved"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix — Test Set")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print("Confusion matrix saved to models/confusion_matrix.png\n")



importances = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(7, 4))
importances.plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Random Forest — Feature Importances (Gini)")
ax.set_xlabel("Importance score")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "feature_importance.png"), dpi=150)
plt.close()
print("Feature importance chart saved to models/feature_importance.png\n")



joblib.dump(model, MODEL_PATH)
print("=" * 60)
print(f"Model saved to {MODEL_PATH}")
print("All artifacts saved. You can now run:  streamlit run app.py")
print("=" * 60)
