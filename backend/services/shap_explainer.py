import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def compute_shap(df: pd.DataFrame, target_col: str = "approved") -> dict:
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available: {list(df.columns)}"
        )

    from services.model_loader import get_model, preprocess_for_inference

    model  = None
    X      = None

    # ── Attempt 1: Use pre-trained model + preprocessed features ─────────
    try:
        model, _, _, _ = get_model()
        X = preprocess_for_inference(df.drop(columns=[target_col]))

        # Test that SHAP can actually load this model — XGBoost + old SHAP
        # versions fail here with "could not convert string to float: '[8.22E-1]'"
        shap.TreeExplainer(model)

    except Exception as primary_err:
        print(f"[shap_explainer] Pre-trained model SHAP failed ({primary_err}). "
              f"Falling back to RandomForest for feature attribution.")
        model = None   # signal to use fallback below

    # ── Attempt 2 (fallback): Train a local RF purely for SHAP ───────────
    if model is None:
        try:
            X = df.drop(columns=[target_col]).copy()
            y_temp = df[target_col].copy()

            for col in X.select_dtypes(include=["object", "category"]).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            X = X.select_dtypes(include="number")

            if y_temp.dtype == object or str(y_temp.dtype) == "category":
                y_temp = LabelEncoder().fit_transform(y_temp.astype(str))
            else:
                y_temp = y_temp.values

            if len(X) > 2000:
                rng    = np.random.RandomState(42)
                idx    = rng.choice(len(X), 2000, replace=False)
                X      = X.iloc[idx].reset_index(drop=True)
                y_temp = y_temp[idx]

            model = RandomForestClassifier(
                n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
            )
            model.fit(X, y_temp)

        except Exception as fallback_err:
            raise RuntimeError(
                f"Both SHAP strategies failed. "
                f"Fallback RF error: {fallback_err}"
            ) from fallback_err

    if X is None or X.empty:
        raise ValueError("No numeric feature columns available for SHAP analysis.")

    # Sample for speed (max 2000 rows)
    if len(X) > 2000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), 2000, replace=False)
        X   = X.iloc[idx].reset_index(drop=True)

    # ── SHAP values ───────────────────────────────────────────────────────
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Handle all SHAP output shapes:
    #   - list of 2 arrays  → binary classification, use class 1
    #   - list of N arrays  → multi-class, average |SHAP| across all classes
    #   - single 3-D array  → newer SHAP (samples, features, classes)
    #   - single 2-D array  → regression / single output
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            sv = shap_values[1]
        else:
            sv = np.mean(np.abs(np.stack(shap_values, axis=0)), axis=0)
    else:
        sv = np.array(shap_values)
        if sv.ndim == 3:
            sv = np.mean(np.abs(sv), axis=2)

    mean_abs = np.abs(sv).mean(axis=0)
    features  = X.columns.tolist()
    mean_abs  = np.array(mean_abs, dtype=float).ravel()

    paired = sorted(zip(features, mean_abs.tolist()), key=lambda x: x[1], reverse=True)
    feature_importance = {f: round(float(v), 6) for f, v in paired}

    sorted_values = [v for _, v in paired]
    top_feature   = paired[0][0] if paired else "unknown"
    shap_max      = round(float(sorted_values[0]),  6) if sorted_values else 0.0
    shap_min      = round(float(sorted_values[-1]), 6) if sorted_values else 0.0

    top5 = sorted_values[:5]
    if len(top5) > 1 and np.mean(top5) > 0:
        stability = float(1.0 - (np.std(top5) / np.mean(top5)))
        stability = round(max(0.0, min(1.0, stability)), 4)
    else:
        stability = 1.0

    return {
        "topFeature":         top_feature,
        "shapMax":            shap_max,
        "shapMin":            shap_min,
        "featureStability":   stability,
        "feature_importance": feature_importance,
    }