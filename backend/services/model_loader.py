import joblib
import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models")
)

_model = None
_scaler = None
_feature_order = None
_numerical_cols = None

def load_artifacts():
    global _model, _scaler, _feature_order, _numerical_cols

    model_path  = os.path.join(MODEL_DIR, "hiring_model.pkl")   # ← correct filename
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    _model  = joblib.load(model_path)
    _scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    with open(os.path.join(MODEL_DIR, "feature_order.json")) as f:
        _feature_order = json.load(f)

    with open(os.path.join(MODEL_DIR, "numerical_cols.json")) as f:
        _numerical_cols = json.load(f)

    return _model, _scaler, _feature_order, _numerical_cols

def get_model():
    if _model is None:
        load_artifacts()
    return _model, _scaler, _feature_order, _numerical_cols

def preprocess_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    _, scaler, feature_order, numerical_cols = get_model()

    df = df.copy()
    df.fillna(0, inplace=True)

    # =====================================
    # 🔥 STEP 1: MAP YOUR DATASET → MODEL SCHEMA
    # =====================================

    # Required base columns
    if 'hours_per_week' not in df.columns:
        df['hours_per_week'] = df.get('years_experience', 0) * 2

    if 'occupation' not in df.columns:
        df['occupation'] = "Tech"   # default

    if 'race' not in df.columns:
        df['race'] = "General"

    if 'caste_proxy' not in df.columns:
        df['caste_proxy'] = "General"

    if 'gender' not in df.columns:
        df['gender'] = "Male"

    if 'education_num' not in df.columns:
        df['education_num'] = 2


    df['employment_years'] = (df['age'] - 18) * 0.7
    df['dependents'] = 2

    df['income_score'] = df['education_num'] * 10 + df['hours_per_week'] * 0.5
    df['dependents_per_income'] = df['dependents'] / (df['income_score'] + 1e-6)

    occ_map = {'Tech':3, 'Management':4, 'Sales':2, 'Admin':2, 'Other':1}
    df['occupation_seniority'] = df['occupation'].map(occ_map).fillna(1)

    # Polynomial features
    df['age_squared'] = df['age'] ** 2
    df['education_squared'] = df['education_num'] ** 2
    df['hours_squared'] = df['hours_per_week'] ** 2

    df['age_education_interaction'] = df['age'] * df['education_num']
    df['age_hours_interaction'] = df['age'] * df['hours_per_week']
    df['education_hours_interaction'] = df['education_num'] * df['hours_per_week']

    df = pd.get_dummies(df)
    
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_order]

    if scaler is not None and numerical_cols:
        cols_present = [c for c in numerical_cols if c in df.columns]
        if cols_present:
            df[cols_present] = scaler.transform(df[cols_present])

    return df