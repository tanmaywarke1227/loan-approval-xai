# Explainable Loan Approval System
### Machine Learning + SHAP Explainable AI | Python + Streamlit

---

## Project Structure

```
loan_approval_xai/
│
├── data/
│   └── loan_data.csv          ← Download from Kaggle (link below)
│
├── models/                    ← Auto-created when you run train_model.py
│   ├── loan_model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   ├── feature_names.pkl
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── utils.py                   ← Shared helper functions
├── train_model.py             ← Run ONCE to train and save the model
├── app.py                     ← Streamlit web app (run after training)
├── requirements.txt           ← All dependencies
└── README.md                  ← This file
```

---

## Setup in VS Code (Step by Step)

### Step 1 — Install Python
Make sure Python 3.9 or higher is installed.
Check by opening VS Code terminal (Ctrl + `) and typing:
```
python --version
```

### Step 2 — Open the project folder in VS Code
```
File → Open Folder → select loan_approval_xai/
```

### Step 3 — Create a virtual environment
In the VS Code terminal:
```bash
python -m venv venv
```

Activate it:
- Windows:    `venv\Scripts\activate`
- Mac/Linux:  `source venv/bin/activate`

You should see `(venv)` at the start of the terminal prompt.

### Step 4 — Install all libraries
```bash
pip install -r requirements.txt
```
This installs: pandas, numpy, scikit-learn, shap, streamlit, matplotlib, joblib.

### Step 5 — Download the dataset
1. Go to: https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset
2. Download `train.csv`
3. Rename it to `loan_data.csv`
4. Place it inside the `data/` folder

### Step 6 — Train the model (run ONCE)
```bash
python train_model.py
```
This will:
- Load and preprocess the dataset
- Train a Random Forest Classifier
- Print accuracy, classification report
- Save model artifacts to `models/`

### Step 7 — Launch the web app
```bash
streamlit run app.py
```
Your browser will open automatically at: http://localhost:8501

---

## Libraries Used

| Library       | Version | Purpose                              |
|---------------|---------|--------------------------------------|
| pandas        | 2.0.3   | Data loading and manipulation        |
| numpy         | 1.24.3  | Numerical operations                 |
| scikit-learn  | 1.3.0   | Random Forest, preprocessing, metrics|
| shap          | 0.44.0  | Explainable AI (SHAP values)         |
| streamlit     | 1.28.0  | Web app dashboard                    |
| matplotlib    | 3.7.2   | Charts and SHAP plots                |
| joblib        | 1.3.2   | Save/load model artifacts            |

---

## How It Works

1. Fill in applicant details in the sidebar
2. Click "Predict Loan Eligibility"
3. The app shows:
   - APPROVED / REJECTED decision
   - Approval probability %
   - Plain-English reason (e.g., "Credit history is the main rejection factor")
   - SHAP waterfall chart showing each feature's exact contribution
   - SHAP bar chart for visual breakdown

---

## Dataset
- Source: Kaggle — Loan Prediction Problem Dataset
- Rows: 614 applicants
- Target: Loan_Status (Y = Approved, N = Rejected)
- Features: Gender, Married, Dependents, Education, Self_Employed,
  ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
  Credit_History
