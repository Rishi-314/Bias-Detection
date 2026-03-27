"""
lime_explainer.py
Local instance explanations using per-instance SHAP values.

The key insight: TreeExplainer gives EXACT per-instance feature contributions.
Each instance gets different values — this is real local explanation, not averages.

Fixed bug: was using original_idx to index sv (SHAP output) instead of i.
sv has shape (n_samples, n_features), indexed 0..n_samples-1, NOT by df row index.
"""

import numpy as np
import pandas as pd
import traceback


def compute_lime(df, target_col, n_samples=5):
    """Try explanation strategies in order, return first that works."""

    # Strategy 1: Official LIME package
    try:
        return _lime_official(df, target_col, n_samples)
    except ImportError:
        print("[lime] lime package not installed, trying per-instance SHAP")
    except Exception as e:
        print(f"[lime] Official LIME failed: {e}")

    # Strategy 2: Per-instance SHAP (best fallback for tree models)
    try:
        result = _per_instance_shap(df, target_col, n_samples)
        print(f"[lime] Per-instance SHAP succeeded: {len(result['instances'])} instances")
        return result
    except Exception as e:
        print(f"[lime] Per-instance SHAP failed: {e}")
        traceback.print_exc()

    # Strategy 3: Train a local RF just for SHAP (model-agnostic fallback)
    try:
        result = _local_rf_shap(df, target_col, n_samples)
        print(f"[lime] Local RF SHAP succeeded")
        return result
    except Exception as e:
        print(f"[lime] Local RF SHAP failed: {e}")

    return {
        "method": "unavailable",
        "instances": [],
        "global_summary": "Local explanations unavailable — check backend logs.",
    }


# ── Strategy 1: Official LIME ────────────────────────────────────────────────

def _lime_official(df, target_col, n_samples):
    import lime.lime_tabular
    from services.model_loader import get_model, preprocess_for_inference

    model, _, _, _ = get_model()
    X = preprocess_for_inference(df.drop(columns=[target_col], errors="ignore"))

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X.values,
        feature_names=X.columns.tolist(),
        class_names=["rejected", "approved"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    idx_sample = _diverse_sample(model, X, n_samples)
    instances = []

    for original_idx in idx_sample:
        exp = explainer.explain_instance(
            X.iloc[original_idx].values,
            model.predict_proba,
            num_features=6,
            num_samples=500,
        )
        pred_prob = float(model.predict_proba(X.iloc[[original_idx]])[0][1])
        top_features = [
            {"feature": f, "weight": round(float(w), 4),
             "direction": "increases approval" if w > 0 else "decreases approval"}
            for f, w in exp.as_list()[:6]
        ]
        instances.append({
            "index": int(original_idx), "prediction": round(pred_prob, 4),
            "outcome": "APPROVED" if pred_prob >= 0.5 else "REJECTED",
            "top_features": top_features,
            "plain_english": _plain_english(pred_prob, top_features),
        })

    return {
        "method": "lime",
        "instances": instances,
        "global_summary": f"Official LIME explanations for {len(instances)} decisions.",
    }


# ── Strategy 2: Per-instance SHAP ───────────────────────────────────────────

def _per_instance_shap(df, target_col, n_samples):
    """
    Uses TreeExplainer on individual rows to get the EXACT contribution of
    each feature to each specific prediction.

    CRITICAL fix: sv is indexed 0..n_samples-1 by POSITION, not by the
    original dataframe row number. Must use the loop counter i, not original_idx.
    """
    import shap
    from services.model_loader import get_model, preprocess_for_inference

    model, _, _, _ = get_model()
    X = preprocess_for_inference(df.drop(columns=[target_col], errors="ignore"))
    feature_names = X.columns.tolist()

    # Select n diverse rows spread across the prediction range
    idx_sample = _diverse_sample(model, X, n_samples)

    # Build a sub-dataframe from just these rows, re-indexed 0..n-1
    X_sample = X.iloc[idx_sample].reset_index(drop=True)

    # SHAP values for all selected rows at once
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_sample)

    # Normalise to 2D: (n_samples, n_features) for the positive class
    sv = _extract_class1_shap(raw)

    assert sv.shape[0] == len(idx_sample), (
        f"Shape mismatch: sv has {sv.shape[0]} rows, idx_sample has {len(idx_sample)}"
    )

    instances = []
    for i, original_idx in enumerate(idx_sample):
        pred_prob = float(model.predict_proba(X.iloc[[original_idx]])[0][1])

        # ── KEY FIX: use i (position in X_sample), NOT original_idx ──────
        row_shap = np.array(sv[i], dtype=float).ravel()

        pairs = sorted(zip(feature_names, row_shap.tolist()),
                       key=lambda x: abs(x[1]), reverse=True)

        meaningful = [(f, v) for f, v in pairs if abs(v) > 1e-8]
        if not meaningful:
            meaningful = pairs[:6]

        top_features = [
            {"feature": f, "weight": round(float(v), 4),
             "direction": "increases approval" if v > 0 else "decreases approval"}
            for f, v in meaningful[:6]
        ]

        instances.append({
            "index": int(original_idx),
            "prediction": round(pred_prob, 4),
            "outcome": "APPROVED" if pred_prob >= 0.5 else "REJECTED",
            "top_features": top_features,
            "plain_english": _plain_english(pred_prob, top_features),
        })

    return {
        "method": "per_instance_shap",
        "instances": instances,
        "global_summary": (
            f"Per-instance SHAP values for {len(instances)} individual decisions. "
            "Each bar shows the exact feature contribution to that specific decision "
            "(green = pushed toward approval, red = pushed toward rejection). "
            "Values differ per record — this is not a global average."
        ),
    }


# ── Strategy 3: Local RF SHAP (model-agnostic fallback) ────────────────────

def _local_rf_shap(df, target_col, n_samples):
    """Train a local RandomForest on the same data, then use its SHAP values."""
    import shap
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    X = df.drop(columns=[target_col], errors="ignore").copy()
    y = df[target_col].copy()

    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.select_dtypes(include="number").fillna(0)

    if str(y.dtype) in ["object", "category"]:
        y = LabelEncoder().fit_transform(y.astype(str))
    else:
        y = y.values.astype(int)

    if len(X) > 2000:
        pick = np.random.choice(len(X), 2000, replace=False)
        X = X.iloc[pick].reset_index(drop=True)
        y = y[pick]

    rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    idx_sample = np.linspace(0, len(X) - 1, min(n_samples, len(X)), dtype=int).tolist()
    X_sample = X.iloc[idx_sample].reset_index(drop=True)

    explainer = shap.TreeExplainer(rf)
    raw = explainer.shap_values(X_sample)
    sv = _extract_class1_shap(raw)

    feature_names = X.columns.tolist()
    instances = []
    for i, original_idx in enumerate(idx_sample):
        pred_prob = float(rf.predict_proba(X.iloc[[original_idx]])[0][1])
        row_shap = np.array(sv[i], dtype=float).ravel()

        pairs = sorted(zip(feature_names, row_shap.tolist()),
                       key=lambda x: abs(x[1]), reverse=True)
        meaningful = [(f, v) for f, v in pairs if abs(v) > 1e-8] or pairs[:6]

        top_features = [
            {"feature": f, "weight": round(float(v), 4),
             "direction": "increases approval" if v > 0 else "decreases approval"}
            for f, v in meaningful[:6]
        ]
        instances.append({
            "index": int(original_idx),
            "prediction": round(pred_prob, 4),
            "outcome": "APPROVED" if pred_prob >= 0.5 else "REJECTED",
            "top_features": top_features,
            "plain_english": _plain_english(pred_prob, top_features),
        })

    return {
        "method": "per_instance_shap",
        "instances": instances,
        "global_summary": (
            f"Per-instance SHAP via local RandomForest for {len(instances)} decisions."
        ),
    }


# ── Shared helpers ───────────────────────────────────────────────────────────

def _extract_class1_shap(raw):
    """
    Normalise TreeExplainer output to 2D (n_samples, n_features) for class 1.

    TreeExplainer can return:
      list of 2 arrays  → sklearn RF binary: [class0, class1]
      list of N arrays  → multiclass
      2D ndarray        → XGBoost binary
      3D ndarray        → some XGBoost versions (n_samples, n_features, n_classes)
    """
    if isinstance(raw, list):
        if len(raw) == 2:
            return np.array(raw[1], dtype=float)   # binary: take class 1
        # multiclass: average absolute values across classes
        return np.mean(np.abs(np.stack(raw, axis=0)), axis=0)

    arr = np.array(raw, dtype=float)
    if arr.ndim == 3:
        return arr[:, :, 1]   # (samples, features, classes) → class 1
    return arr                 # already (samples, features)


def _diverse_sample(model, X, n):
    """
    Select n rows spread evenly across the predicted probability range.
    This ensures we show a variety of outcomes (rejected, borderline, approved).
    """
    total = len(X)
    if total <= n:
        return list(range(total))
    try:
        probs = model.predict_proba(X)[:, 1]
        sorted_idx = np.argsort(probs)
        positions = np.linspace(0, len(sorted_idx) - 1, n, dtype=int)
        return [int(sorted_idx[p]) for p in positions]
    except Exception:
        return np.random.choice(total, n, replace=False).tolist()


def _plain_english(pred: float, top_features: list) -> str:
    outcome = "approved" if pred >= 0.5 else "rejected"
    conf = round(pred * 100, 1) if pred >= 0.5 else round((1 - pred) * 100, 1)

    if not top_features:
        return f"This application was {outcome} with {conf}% confidence."

    top = max(top_features, key=lambda f: abs(f.get("weight", 0)))
    name = top["feature"].replace("_", " ").split("<=")[0].split(">")[0].strip()
    w = top.get("weight", 0)

    # Match protected attributes as whole words / column names, not substrings
    # e.g. "age" should NOT match "age_education_interaction"
    feature_lower = top["feature"].lower()
    sensitive_exact = ["gender", "sex", "race", "caste", "religion", "ethnicity", "nationality"]
    # "age" only if the column IS "age" or starts with "age" as a standalone word
    is_sensitive = (
        any(feature_lower == k or feature_lower.startswith(k + "_") or feature_lower.endswith("_" + k)
            for k in sensitive_exact)
        or feature_lower in ["age", "gender_female", "gender_male", "race_obc", "race_sc", "race_st",
                              "race_general", "caste_proxy_sc", "caste_proxy_st", "caste_proxy_obc",
                              "caste_proxy_general"]
    )
    warn = (
        f" ⚠️ Note: '{name}' is a protected attribute — this influence should be investigated."
        if is_sensitive else ""
    )

    second = ""
    for f in top_features[1:]:
        if abs(f.get("weight", 0)) > abs(w) * 0.25:
            n2 = f["feature"].replace("_", " ").split("<=")[0].strip()
            second = f" '{n2}' also contributed ({f['direction']})."
            break

    return (
        f"This application was {outcome} with {conf}% confidence. "
        f"The biggest factor was '{name}' (SHAP: {w:+.4f}), which {top['direction']}.{second}{warn}"
    )