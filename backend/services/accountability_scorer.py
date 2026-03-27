"""
accountability_scorer.py
Post-hoc accountability and transparency scoring.
Does NOT touch the model.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List


def compute_accountability_score(
    df: pd.DataFrame,
    sensitive_attrs: List[str],
    target_col: str,
    fairness_results: dict,
    shap_results: dict,
    eval_metadata: dict,
) -> Dict[str, Any]:
    """
    Score accountability based on:
    - Were sensitive attributes declared?
    - Is the target variable documented?
    - Is the audit trail complete?
    - Were all 7 fairness dimensions evaluated?
    - Is explainability available (SHAP)?
    """
    findings = []
    recommendations = []
    score = 1.0
    checks = []

    # ── Check 1: Sensitive attributes declared ────────────────────────────────
    n_attrs = len([a for a in (sensitive_attrs or []) if a in df.columns])
    if n_attrs == 0:
        score -= 0.20
        checks.append({"name": "Sensitive attributes declared", "passed": False,
                        "detail": "No protected attributes were specified for evaluation."})
        recommendations.append({
            "priority": "HIGH",
            "action": "Always declare sensitive attributes (gender, race, age, caste) before evaluation",
            "code": 'sensitive_attributes = ["gender", "race", "age"]',
            "effort": "Low",
        })
    elif n_attrs < 2:
        score -= 0.07
        checks.append({"name": "Sensitive attributes declared", "passed": True,
                        "detail": f"Only {n_attrs} protected attribute(s) declared — consider more."})
    else:
        checks.append({"name": "Sensitive attributes declared", "passed": True,
                        "detail": f"{n_attrs} protected attribute(s) properly declared."})

    # ── Check 2: Target variable documented ───────────────────────────────────
    if target_col and target_col in df.columns:
        checks.append({"name": "Target variable documented", "passed": True,
                        "detail": f"Target variable '{target_col}' is identified and present."})
    else:
        score -= 0.15
        checks.append({"name": "Target variable documented", "passed": False,
                        "detail": "Target variable not explicitly documented."})

    # ── Check 3: All 7 fairness dimensions evaluated ──────────────────────────
    required_dims = [
        "individualFairness", "groupFairness", "demographicParity",
        "disparateImpact", "calibrationError", "counterfactual", "intersectional"
    ]
    evaluated_dims = [d for d in required_dims if fairness_results.get(d) is not None]
    coverage = len(evaluated_dims) / len(required_dims)
    if coverage < 1.0:
        score -= (1.0 - coverage) * 0.20
        checks.append({"name": "All fairness dimensions evaluated", "passed": False,
                        "detail": f"Only {len(evaluated_dims)}/7 fairness dimensions were computed."})
    else:
        checks.append({"name": "All fairness dimensions evaluated", "passed": True,
                        "detail": "All 7 fairness dimensions successfully evaluated."})

    # ── Check 4: Explainability available ────────────────────────────────────
    has_shap = bool(shap_results and shap_results.get("topFeature"))
    if has_shap:
        checks.append({"name": "Model explainability (SHAP)", "passed": True,
                        "detail": f"SHAP explanations available. Top feature: {shap_results.get('topFeature')}."})
    else:
        score -= 0.15
        checks.append({"name": "Model explainability (SHAP)", "passed": False,
                        "detail": "No SHAP explanation available. Model decisions are unexplainable."})
        recommendations.append({
            "priority": "HIGH",
            "action": "Enable SHAP explainability for all model evaluations",
            "code": "import shap; explainer = shap.TreeExplainer(model); shap_values = explainer.shap_values(X)",
            "effort": "Low",
        })

    # ── Check 5: Per-attribute breakdown ─────────────────────────────────────
    per_attr = fairness_results.get("per_attribute", {})
    if per_attr and n_attrs > 0:
        checks.append({"name": "Per-attribute fairness breakdown", "passed": True,
                        "detail": f"Fairness breakdown available for {len(per_attr)} attribute(s)."})
    else:
        score -= 0.10
        checks.append({"name": "Per-attribute fairness breakdown", "passed": False,
                        "detail": "No per-attribute fairness breakdown available."})

    # ── Check 6: Decision logging (audit trail) ───────────────────────────────
    has_records = eval_metadata.get("records", 0) > 0
    if has_records:
        checks.append({"name": "Decision records logged", "passed": True,
                        "detail": f"{eval_metadata.get('records', 0):,} decisions evaluated and logged."})
    else:
        score -= 0.10
        checks.append({"name": "Decision records logged", "passed": False,
                        "detail": "No decision records found in evaluation."})

    # ── Check 7: Report generated ─────────────────────────────────────────────
    has_report = bool(eval_metadata.get("report_type"))
    if has_report:
        checks.append({"name": "Compliance report generated", "passed": True,
                        "detail": f"Report type: {eval_metadata.get('report_type')}."})
    else:
        score -= 0.05
        checks.append({"name": "Compliance report generated", "passed": False,
                        "detail": "No compliance report generated for this evaluation."})

    accountability_score = round(max(0.0, min(1.0, score)), 4)
    passed = sum(1 for c in checks if c["passed"])

    return {
        "accountability_score": accountability_score,
        "checks_passed":        passed,
        "checks_total":         len(checks),
        "checks":               checks,
        "findings":             findings,
        "recommendations":      recommendations,
    }


def compute_transparency_score(
    shap_results: dict,
    fairness_results: dict,
    df: pd.DataFrame,
    target_col: str,
) -> Dict[str, Any]:
    """
    Score transparency based on:
    - SHAP feature stability
    - Feature dominance (is one feature dominating everything?)
    - Number of meaningful features
    - Interpretability of top features
    """
    findings = []
    recommendations = []
    score = 1.0

    if not shap_results or not shap_results.get("topFeature"):
        return {
            "transparency_score": 0.4,
            "feature_stability": None,
            "feature_dominance": None,
            "interpretable_feature_count": 0,
            "top_feature": None,
            "findings": [{"severity": "HIGH", "category": "Transparency",
                          "title": "No SHAP data available",
                          "detail": "Model cannot be explained without SHAP values."}],
            "recommendations": [{"priority": "HIGH", "action": "Enable SHAP explainability",
                                  "code": "shap.TreeExplainer(model)", "effort": "Low"}],
        }

    # ── Feature Stability ─────────────────────────────────────────────────────
    stability = shap_results.get("featureStability", 1.0)
    if stability < 0.5:
        score -= 0.20
        findings.append({
            "severity": "HIGH",
            "category": "Feature Instability",
            "title": f"SHAP feature stability = {stability:.2f} (below 0.50)",
            "detail": "The model's important features change significantly across samples, making it hard to explain.",
        })
    elif stability < 0.7:
        score -= 0.10
        findings.append({
            "severity": "MEDIUM",
            "category": "Feature Instability",
            "title": f"SHAP feature stability = {stability:.2f}",
            "detail": "Model explanations are somewhat unstable. Top features vary across predictions.",
        })

    # ── Feature Dominance ─────────────────────────────────────────────────────
    fi = shap_results.get("feature_importance", {})
    feature_dominance = 0.0
    if fi and isinstance(fi, dict) and len(fi) > 1:
        vals = sorted(fi.values(), reverse=True)
        total = sum(vals)
        if total > 0:
            top_share = vals[0] / total
            feature_dominance = top_share
            if top_share > 0.60:
                score -= 0.15
                findings.append({
                    "severity": "HIGH",
                    "category": "Feature Dominance",
                    "title": f"Top feature '{shap_results.get('topFeature')}' accounts for {round(top_share*100,1)}% of model decisions",
                    "detail": (
                        "A single feature dominates all predictions. This reduces model transparency "
                        "and may indicate proxy discrimination if the feature correlates with protected attributes."
                    ),
                })
                recommendations.append({
                    "priority": "HIGH",
                    "action": f"Investigate '{shap_results.get('topFeature')}' for proxy discrimination",
                    "code": f"# Check: df['{shap_results.get('topFeature')}'].corr(df[sensitive_attr])",
                    "effort": "Low",
                })
            elif top_share > 0.40:
                score -= 0.07
                findings.append({
                    "severity": "MEDIUM",
                    "category": "Feature Dominance",
                    "title": f"Top feature dominates {round(top_share*100,1)}% of decisions",
                    "detail": "One feature is highly influential. Verify it is a legitimate, non-discriminatory factor.",
                })

    # ── Interpretable Feature Count ───────────────────────────────────────────
    n_features = len(fi) if fi else 0
    if n_features < 3:
        score -= 0.10
        findings.append({
            "severity": "MEDIUM",
            "category": "Feature Breadth",
            "title": f"Only {n_features} feature(s) have non-zero SHAP values",
            "detail": "Very few features drive decisions, limiting the interpretability of individual predictions.",
        })

    transparency_score = round(max(0.0, min(1.0, score)), 4)

    if not findings:
        findings.append({
            "severity": "INFO",
            "category": "Transparency",
            "title": "Model transparency looks good",
            "detail": f"SHAP stability = {stability:.2f}. No single feature dominates decisions.",
        })

    return {
        "transparency_score":         transparency_score,
        "feature_stability":          round(float(stability), 4),
        "feature_dominance":          round(feature_dominance, 4),
        "interpretable_feature_count": n_features,
        "top_feature":                shap_results.get("topFeature"),
        "shap_max":                   shap_results.get("shapMax"),
        "shap_min":                   shap_results.get("shapMin"),
        "findings":                   findings,
        "recommendations":            recommendations,
    }