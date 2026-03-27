"""
privacy_scorer.py
Post-hoc privacy analysis of the dataset and predictions.
Does NOT touch the model.

Returns a privacy_score (0-1, higher = more private / safer) plus findings.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, Any

# ── PII keyword patterns ──────────────────────────────────────────────────────
PII_COLUMN_PATTERNS = [
    r'\bname\b', r'\bfirst.?name\b', r'\blast.?name\b', r'\bfull.?name\b',
    r'\bemail\b', r'\be.?mail\b',
    r'\bphone\b', r'\bmobile\b', r'\bcell\b', r'\btelephone\b',
    r'\bssn\b', r'\bsocial.?security\b', r'\bnational.?id\b', r'\baadhar\b', r'\bpassport\b',
    r'\baddress\b', r'\bstreet\b', r'\bzip\b', r'\bpostcode\b', r'\bpincode\b',
    r'\bip.?address\b', r'\bmac.?address\b',
    r'\bdate.?of.?birth\b', r'\bdob\b', r'\bbirthday\b',
    r'\bcredit.?card\b', r'\bcard.?number\b', r'\biban\b', r'\baccount.?number\b',
    r'\bmedical\b', r'\bdiagnosis\b', r'\bhealth\b',
    r'\bbiometric\b', r'\bfingerprint\b', r'\bface.?id\b',
]

QUASI_IDENTIFIER_PATTERNS = [
    r'\bage\b', r'\bgender\b', r'\bsex\b', r'\brace\b', r'\bethnicity\b',
    r'\boccupation\b', r'\bzipcode\b', r'\bpostcode\b', r'\bregion\b',
    r'\bnationality\b', r'\breligion\b', r'\bcaste\b',
]

SENSITIVE_ATTR_PATTERNS = [
    r'\bincome\b', r'\bsalary\b', r'\bwage\b',
    r'\bcredit.?score\b', r'\bcredit\b',
    r'\bcriminal\b', r'\bconviction\b',
    r'\bimmigration\b', r'\bvisa\b',
    r'\bpolitical\b', r'\bvote\b',
]


def _check_column(col: str, patterns: list) -> bool:
    col_lower = col.lower().replace('_', ' ').replace('-', ' ')
    return any(re.search(p, col_lower) for p in patterns)


def compute_privacy_score(
    df: pd.DataFrame,
    preds: pd.DataFrame,
    target_col: str,
    sensitive_attrs: list,
) -> Dict[str, Any]:
    """
    Analyse the dataset for privacy risks without touching the model.

    Returns:
        privacy_score       float 0-1 (higher = safer)
        pii_columns         list of detected PII column names
        quasi_identifiers   list of quasi-identifier columns
        sensitive_columns   list of sensitive attribute columns
        data_leakage_risk   float 0-1 (higher = more risk)
        k_anonymity_estimate int  (lower = worse privacy)
        findings            list of finding dicts
        recommendations     list of recommendation dicts
    """
    columns = list(df.columns)
    findings = []
    recommendations = []
    penalty = 0.0  # accumulated deduction from 1.0

    # ── 1. PII Detection ─────────────────────────────────────────────────────
    pii_cols = [c for c in columns if _check_column(c, PII_COLUMN_PATTERNS)]
    quasi_cols = [c for c in columns if _check_column(c, QUASI_IDENTIFIER_PATTERNS)]
    sensitive_cols = [c for c in columns if _check_column(c, SENSITIVE_ATTR_PATTERNS)]

    if pii_cols:
        penalty += min(0.35, len(pii_cols) * 0.07)
        findings.append({
            "severity": "HIGH",
            "category": "PII Exposure",
            "title": f"Direct PII columns detected: {', '.join(pii_cols[:5])}",
            "detail": (
                f"{len(pii_cols)} column(s) appear to contain personally identifiable information. "
                "Including PII in model training data risks memorization and re-identification attacks."
            ),
            "columns": pii_cols,
        })
        recommendations.append({
            "priority": "HIGH",
            "action": "Remove or pseudonymize PII columns before training",
            "code": f"df.drop(columns={pii_cols[:3]}, inplace=True)  # or use hashing",
            "effort": "Low",
        })

    if quasi_cols:
        penalty += min(0.20, len(quasi_cols) * 0.04)
        findings.append({
            "severity": "MEDIUM",
            "category": "Quasi-Identifier Risk",
            "title": f"Quasi-identifiers present: {', '.join(quasi_cols[:5])}",
            "detail": (
                "Combinations of quasi-identifiers (age, gender, zip, etc.) can re-identify individuals "
                "even without direct PII. Consider generalization or suppression."
            ),
            "columns": quasi_cols,
        })

    # ── 2. Data Leakage Risk ──────────────────────────────────────────────────
    # Proxy: if a non-target column is near-perfectly correlated with target
    leakage_risk = 0.0
    leakage_cols = []
    try:
        y = df[target_col].astype(float)
        for col in columns:
            if col == target_col:
                continue
            if df[col].dtype in [np.float64, np.int64, float, int]:
                try:
                    corr = abs(float(df[col].astype(float).corr(y)))
                    if corr > 0.95:
                        leakage_cols.append((col, round(corr, 4)))
                        leakage_risk = max(leakage_risk, corr)
                except Exception:
                    pass
    except Exception:
        pass

    if leakage_cols:
        penalty += 0.25
        findings.append({
            "severity": "HIGH",
            "category": "Data Leakage",
            "title": f"Potential target leakage: {', '.join(c for c, _ in leakage_cols[:3])}",
            "detail": (
                f"Column(s) with correlation > 0.95 to the target variable found. "
                "This may indicate label leakage — inflating model performance artificially."
            ),
            "columns": [c for c, _ in leakage_cols],
            "correlations": {c: v for c, v in leakage_cols},
        })
        recommendations.append({
            "priority": "HIGH",
            "action": "Remove or investigate high-correlation columns",
            "code": f"# Columns with near-perfect target correlation:\n# {[c for c,_ in leakage_cols[:3]]}",
            "effort": "Medium",
        })
    else:
        leakage_risk = 0.02  # baseline tiny risk

    # ── 3. Prediction Confidence Analysis (memorization proxy) ───────────────
    # If a very high fraction of predictions are near 0 or 1, model may be overfit
    overconfidence_ratio = 0.0
    boundary_ratio = 0.0
    try:
        pred_col = [c for c in preds.columns if preds[c].between(0, 1).all()]
        if pred_col:
            probs = preds[pred_col[0]].dropna()
            overconfident = ((probs > 0.97) | (probs < 0.03)).mean()
            boundary = ((probs > 0.4) & (probs < 0.6)).mean()
            overconfidence_ratio = float(overconfident)
            boundary_ratio = float(boundary)

            if overconfident > 0.70:
                penalty += 0.10
                findings.append({
                    "severity": "MEDIUM",
                    "category": "Overconfidence / Memorization Risk",
                    "title": f"{round(overconfident*100,1)}% of predictions are near-certain (>97% or <3%)",
                    "detail": (
                        "Extremely high-confidence predictions across most of the dataset may indicate "
                        "overfitting or memorization of training examples, which is a privacy risk."
                    ),
                    "columns": [],
                })
    except Exception:
        pass

    # ── 4. k-Anonymity Estimate ───────────────────────────────────────────────
    k_anon = None
    try:
        qi_in_df = [c for c in quasi_cols if c in df.columns]
        if qi_in_df:
            group_sizes = df.groupby(qi_in_df).size()
            k_anon = int(group_sizes.min())
            if k_anon < 3:
                penalty += 0.15
                findings.append({
                    "severity": "HIGH",
                    "category": "k-Anonymity",
                    "title": f"k-anonymity = {k_anon} (below threshold of 3)",
                    "detail": (
                        f"The smallest group formed by quasi-identifiers has only {k_anon} member(s). "
                        "This means individuals in rare combinations can be re-identified."
                    ),
                    "columns": qi_in_df,
                })
            elif k_anon < 5:
                penalty += 0.07
                findings.append({
                    "severity": "MEDIUM",
                    "category": "k-Anonymity",
                    "title": f"k-anonymity = {k_anon} (borderline — threshold is 5)",
                    "detail": "Some quasi-identifier combinations have few members, posing re-identification risk.",
                    "columns": qi_in_df,
                })
    except Exception:
        k_anon = None

    # ── 5. Sensitive Attribute in Training Data ───────────────────────────────
    declared_sensitive = set(sensitive_attrs or [])
    undeclared_sensitive = [
        c for c in sensitive_cols
        if c in columns and c not in declared_sensitive and c != target_col
    ]
    if undeclared_sensitive:
        penalty += min(0.15, len(undeclared_sensitive) * 0.05)
        findings.append({
            "severity": "MEDIUM",
            "category": "Undeclared Sensitive Attributes",
            "title": f"Sensitive columns not declared as protected: {', '.join(undeclared_sensitive[:4])}",
            "detail": (
                "These columns appear sensitive but were not declared as protected attributes. "
                "They may be used as proxy variables for discrimination."
            ),
            "columns": undeclared_sensitive,
        })
        recommendations.append({
            "priority": "MEDIUM",
            "action": "Declare all sensitive attributes explicitly in evaluation config",
            "code": f'sensitive_attributes = {list(declared_sensitive) + undeclared_sensitive[:3]}',
            "effort": "Low",
        })

    # ── Final Score ───────────────────────────────────────────────────────────
    privacy_score = round(max(0.0, min(1.0, 1.0 - penalty)), 4)

    if not findings:
        findings.append({
            "severity": "INFO",
            "category": "Privacy",
            "title": "No major privacy issues detected",
            "detail": "No direct PII, data leakage, or k-anonymity violations found in the dataset.",
            "columns": [],
        })

    return {
        "privacy_score":       privacy_score,
        "pii_columns":         pii_cols,
        "quasi_identifiers":   quasi_cols,
        "sensitive_columns":   sensitive_cols,
        "undeclared_sensitive": undeclared_sensitive,
        "data_leakage_risk":   round(leakage_risk, 4),
        "overconfidence_ratio": round(overconfidence_ratio, 4),
        "boundary_ratio":      round(boundary_ratio, 4),
        "k_anonymity":         k_anon,
        "findings":            findings,
        "recommendations":     recommendations,
        "penalty":             round(penalty, 4),
    }