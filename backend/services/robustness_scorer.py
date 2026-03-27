"""
robustness_scorer.py
Post-hoc robustness analysis using only predictions + dataset.
Does NOT touch or modify the model.

Measures:
- Prediction stability near decision boundary
- Class balance in predictions
- Outlier sensitivity
- Edge case coverage
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def compute_robustness_score(
    df: pd.DataFrame,
    preds: pd.DataFrame,
    target_col: str,
    sensitive_attrs: list,
) -> Dict[str, Any]:
    """
    Post-hoc robustness analysis.

    Returns:
        robustness_score        float 0-1 (higher = more robust)
        boundary_instability    float 0-1 (fraction near decision boundary)
        class_imbalance         float 0-1 (prediction balance ratio)
        outlier_sensitivity     float 0-1 (how much outliers affect predictions)
        edge_case_coverage      float 0-1 (how well edge cases are handled)
        findings                list of finding dicts
        recommendations         list of recommendation dicts
    """
    findings = []
    recommendations = []
    penalty = 0.0

    columns = list(df.columns)

    # Resolve prediction column
    pred_col = None
    for c in preds.columns:
        if preds[c].between(0, 1).all():
            pred_col = c
            break
    if pred_col is None:
        numeric = preds.select_dtypes(include='number').columns.tolist()
        if numeric:
            pred_col = numeric[0]

    if pred_col is None:
        return {
            "robustness_score": 0.5,
            "boundary_instability": 0.5,
            "class_imbalance": 0.5,
            "outlier_sensitivity": 0.5,
            "edge_case_coverage": 0.5,
            "findings": [{"severity": "INFO", "category": "Robustness", "title": "No prediction column found", "detail": "Cannot compute robustness.", "columns": []}],
            "recommendations": [],
        }

    probs = preds[pred_col].dropna().values
    n = len(probs)

    # ── 1. Decision Boundary Instability ─────────────────────────────────────
    # Records near 0.5 threshold — model is uncertain about these
    boundary_mask = (probs > 0.40) & (probs < 0.60)
    boundary_ratio = float(boundary_mask.mean())

    if boundary_ratio > 0.30:
        penalty += 0.20
        findings.append({
            "severity": "HIGH",
            "category": "Decision Boundary Instability",
            "title": f"{round(boundary_ratio*100,1)}% of predictions near decision boundary (0.4–0.6)",
            "detail": (
                "A high fraction of predictions fall in the uncertain zone near the 0.5 threshold. "
                "Small perturbations in input data could flip these decisions, making the model unstable."
            ),
            "value": round(boundary_ratio, 4),
        })
        recommendations.append({
            "priority": "HIGH",
            "action": "Review training data quality — model is uncertain for too many records",
            "code": "# Investigate records where 0.4 < pred_score < 0.6\nboundary_cases = df[boundary_mask]",
            "effort": "High",
        })
    elif boundary_ratio > 0.15:
        penalty += 0.08
        findings.append({
            "severity": "MEDIUM",
            "category": "Decision Boundary Instability",
            "title": f"{round(boundary_ratio*100,1)}% of predictions in uncertain zone",
            "detail": "Some predictions are near the decision threshold — monitor for stability.",
            "value": round(boundary_ratio, 4),
        })

    # ── 2. Prediction Class Balance ───────────────────────────────────────────
    pos_rate = float((probs >= 0.5).mean())
    class_imbalance_ratio = min(pos_rate, 1 - pos_rate) / max(pos_rate, 1 - pos_rate) if max(pos_rate, 1 - pos_rate) > 0 else 1.0

    if pos_rate < 0.05 or pos_rate > 0.95:
        penalty += 0.15
        findings.append({
            "severity": "HIGH",
            "category": "Prediction Class Imbalance",
            "title": f"Extreme prediction imbalance: {round(pos_rate*100,1)}% positive predictions",
            "detail": (
                "The model overwhelmingly predicts one class. This may indicate the model learned "
                "a trivial rule from imbalanced training data, reducing its utility for the minority class."
            ),
            "value": round(pos_rate, 4),
        })
        recommendations.append({
            "priority": "HIGH",
            "action": "Investigate class imbalance in training data — consider SMOTE or class weights",
            "code": "# from sklearn.utils import class_weight\n# weights = class_weight.compute_class_weight('balanced', classes=[0,1], y=y_train)",
            "effort": "Medium",
        })
    elif pos_rate < 0.10 or pos_rate > 0.90:
        penalty += 0.08
        findings.append({
            "severity": "MEDIUM",
            "category": "Prediction Class Imbalance",
            "title": f"Notable prediction imbalance: {round(pos_rate*100,1)}% positive",
            "detail": "Model prediction rates are significantly skewed. Review for fairness implications.",
            "value": round(pos_rate, 4),
        })

    # ── 3. Outlier Sensitivity ────────────────────────────────────────────────
    # Detect numeric outliers in dataset and check if predictions cluster differently
    outlier_sensitivity = 0.0
    try:
        numeric_cols = df.select_dtypes(include='number').drop(
            columns=[target_col], errors='ignore'
        ).columns.tolist()

        if numeric_cols and len(probs) == len(df):
            outlier_flags = pd.Series(False, index=df.index)
            for col in numeric_cols[:5]:  # check top 5 numeric cols
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    is_outlier = (df[col] < q1 - 3*iqr) | (df[col] > q3 + 3*iqr)
                    outlier_flags = outlier_flags | is_outlier

            outlier_count = outlier_flags.sum()
            if outlier_count > 0:
                outlier_pred_mean   = float(probs[outlier_flags.values].mean()) if outlier_count > 0 else 0.5
                inlier_pred_mean    = float(probs[~outlier_flags.values].mean())
                outlier_sensitivity = abs(outlier_pred_mean - inlier_pred_mean)

                if outlier_sensitivity > 0.20:
                    penalty += 0.12
                    findings.append({
                        "severity": "MEDIUM",
                        "category": "Outlier Sensitivity",
                        "title": f"Model predictions shift {round(outlier_sensitivity*100,1)}% for outlier records",
                        "detail": (
                            f"{int(outlier_count)} outlier records ({round(outlier_count/len(df)*100,1)}% of data) "
                            "receive systematically different predictions. Model may be sensitive to data quality issues."
                        ),
                        "value": round(outlier_sensitivity, 4),
                    })
    except Exception:
        outlier_sensitivity = 0.0

    # ── 4. Edge Case Coverage ─────────────────────────────────────────────────
    # Check if sensitive attribute groups have sufficient representation
    edge_case_issues = []
    try:
        valid_attrs = [a for a in (sensitive_attrs or []) if a in df.columns]
        for attr in valid_attrs:
            group_counts = df[attr].value_counts()
            rare_groups  = group_counts[group_counts < max(10, len(df) * 0.01)]
            if len(rare_groups) > 0:
                edge_case_issues.append({
                    "attribute": attr,
                    "rare_groups": rare_groups.to_dict(),
                })

        if edge_case_issues:
            penalty += min(0.15, len(edge_case_issues) * 0.05)
            findings.append({
                "severity": "MEDIUM",
                "category": "Edge Case Coverage",
                "title": f"Underrepresented groups in {len(edge_case_issues)} attribute(s)",
                "detail": (
                    "Some protected groups have very few samples (<1% of dataset). "
                    "Model behavior for these groups is unreliable — fairness metrics may not be accurate."
                ),
                "value": len(edge_case_issues),
                "details": edge_case_issues,
            })
            recommendations.append({
                "priority": "MEDIUM",
                "action": "Collect more data for underrepresented groups before deployment",
                "code": "# Target at least 100 samples per group for reliable fairness measurement",
                "effort": "High",
            })
    except Exception:
        pass

    # ── 5. Prediction Entropy / Confidence Distribution ───────────────────────
    # Good model: well-spread confidence. Bad: all predictions near 0 or 1 (overconfident)
    overconfident_ratio = float(((probs > 0.95) | (probs < 0.05)).mean())
    if overconfident_ratio > 0.80:
        penalty += 0.10
        findings.append({
            "severity": "MEDIUM",
            "category": "Overconfidence",
            "title": f"{round(overconfident_ratio*100,1)}% of predictions are near-certain",
            "detail": (
                "Extremely high model confidence across most records suggests possible overfitting. "
                "Well-calibrated models should show a spread of confidence values."
            ),
            "value": round(overconfident_ratio, 4),
        })

    # ── Final Score ───────────────────────────────────────────────────────────
    robustness_score = round(max(0.0, min(1.0, 1.0 - penalty)), 4)
    edge_case_score  = round(max(0.0, 1.0 - min(0.5, len(edge_case_issues) * 0.15)), 4)

    if not findings:
        findings.append({
            "severity": "INFO",
            "category": "Robustness",
            "title": "No major robustness issues detected",
            "detail": "Prediction distribution appears stable with no extreme boundary instability.",
            "value": None,
        })

    return {
        "robustness_score":     robustness_score,
        "boundary_instability": round(boundary_ratio, 4),
        "class_imbalance":      round(1 - class_imbalance_ratio, 4),
        "outlier_sensitivity":  round(outlier_sensitivity, 4),
        "edge_case_coverage":   edge_case_score,
        "positive_rate":        round(pos_rate, 4),
        "overconfident_ratio":  round(overconfident_ratio, 4),
        "findings":             findings,
        "recommendations":      recommendations,
        "penalty":              round(penalty, 4),
    }