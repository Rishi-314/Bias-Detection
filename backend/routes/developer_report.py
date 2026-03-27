from flask import Blueprint, jsonify
from routes.evaluation import evaluations   # shared in-memory store

developer_report_routes = Blueprint("developer_report_routes", __name__)


def _badge(score: float, threshold: float = 0.7) -> str:
    if score >= 0.8:
        return "PASS"
    elif score >= 0.6:
        return "WARNING"
    return "FAIL"


def _dimension_style(score: float) -> str:
    return "success" if score >= 0.80 else "warning"


# ── Dimension builders ────────────────────────────────────────────────────────
def normalize_gap(gap: float, threshold: float = 0.05) -> float:
    """
    Converts fairness gap into score.

    threshold = acceptable gap (5%)
    beyond this → strong penalty
    """
    if gap is None:
        return 0.5

    if gap <= threshold:
        return round(1 - (gap / threshold) * 0.5, 4)  # small penalty

    # heavy penalty beyond threshold
    return round(max(0.0, 0.5 * (threshold / gap)), 4)

def _build_individual_fairness(fairness: dict) -> dict:
    score = fairness.get("individualFairness", 0.0)
    return {
        "id":          1,
        "icon":        "👥",
        "title":       "Individual Fairness",
        "score":       round(score, 4),
        "status":      _badge(score),
        "style":       _dimension_style(score),
        "methodology": "KNN consistency check (k=5 nearest neighbours)",
        "violation_rate": round((1 - score) * 100, 2),
        "detail": {
            "description": (
                "Measures whether similar individuals receive similar predictions. "
                "Uses k-NN to find near-identical records and checks prediction consistency."
            ),
            "consistency_pct": round(score * 100, 1),
        },
    }


def _build_group_fairness(fairness: dict) -> dict:
    per_attr = fairness.get("per_attribute", {})

    worst_eo = 0.0
    groups = []

    for attr, metrics in per_attr.items():
        if "error" in metrics:
            continue

        eo = abs(metrics.get("equalized_odds_difference", 0.0))
        worst_eo = max(worst_eo, eo)

        groups.append({
            "attribute": attr,
            "equalized_odds_diff": round(eo, 4),
        })

    score = normalize_gap(worst_eo)

    return {
        "id": 2,
        "icon": "👪",
        "title": "Group Fairness",
        "score": score,
        "status": _badge(score, threshold=0.7),
        "style": _dimension_style(score),
        "detail": {
            "worst_equalized_odds_diff": round(worst_eo, 4),
            "per_attribute": groups,
        },
    }


def _build_demographic_bias(fairness: dict) -> dict:
    per_attr = fairness.get("per_attribute", {})

    worst_gap = 0.0
    disparities = []

    for attr, metrics in per_attr.items():
        if "error" in metrics:
            continue

        gap = abs(metrics.get("demographic_parity_difference", 0.0))
        worst_gap = max(worst_gap, gap)

        disparities.append({
            "attribute": attr,
            "gap_pct": round(gap * 100, 2),
            "raw_gap": round(gap, 4),
        })

    score = normalize_gap(worst_gap)

    return {
        "id": 3,
        "icon": "🌍",
        "title": "Demographic Bias",
        "score": score,
        "status": _badge(score, threshold=0.7),
        "style": _dimension_style(score),
        "detail": {
            "worst_gap": round(worst_gap, 4),
            "disparities": disparities,
            "interpretation": (
                f"Worst group gap is {round(worst_gap*100,2)}%. "
                "Above 5% indicates potential bias."
            ),
        },
    }

def _build_calibration(fairness: dict) -> dict:
    cal_err = fairness.get("calibrationError", 0.0)

    # Brier score threshold ~0.1 good
    score = normalize_gap(cal_err, threshold=0.1)

    return {
        "id": 4,
        "icon": "⚖️",
        "title": "Calibration",
        "score": score,
        "status": _badge(score, threshold=0.7),
        "style": _dimension_style(score),
        "detail": {
            "brier_score": round(cal_err, 4),
            "interpretation": (
                "Closer to 0 is better. >0.1 indicates poor calibration."
            ),
        },
    }


def _build_disparate_impact(fairness: dict) -> dict:
    di = fairness.get("disparateImpact", 0.0)

    # ideal = 1.0
    gap = abs(1 - di)
    score = normalize_gap(gap, threshold=0.2)

    return {
        "id": 5,
        "icon": "📊",
        "title": "Disparate Impact",
        "score": score,
        "status": _badge(score, threshold=0.7),
        "style": _dimension_style(score),
        "detail": {
            "impact_ratio": round(di, 4),
            "ideal": 1.0,
        },
    }


def _build_counterfactual(fairness: dict) -> dict:
    score_raw = fairness.get("counterfactual", 1.0)

    # convert to violation
    violation = 1 - score_raw

    score = normalize_gap(violation, threshold=0.05)

    return {
        "id": 6,
        "icon": "🔄",
        "title": "Counterfactual Fairness",
        "score": score,
        "status": _badge(score, threshold=0.7),
        "style": _dimension_style(score),
        "detail": {
            "violation_rate": round(violation * 100, 2),
            "interpretation": (
                "Measures how often changing sensitive attributes changes outcome."
            ),
        },
    }


def _build_intersectional(fairness: dict) -> dict:
    score = fairness.get("intersectional", 0.0)
    return {
        "id":     7,
        "icon":   "🔀",
        "title":  "Intersectional Fairness",
        "score":  round(score, 4),
        "status": _badge(score),
        "style":  _dimension_style(score),
        "detail": {
            "description": (
                "Evaluates worst-case demographic parity gap across all pairwise combinations "
                "of sensitive attributes (e.g. gender × age)."
            ),
            "worst_case_gap": round(max(0.0, 1.0 - score), 4),
            "interpretation": (
                "A score of 1.0 means no intersectional disparity. "
                f"Current worst-case gap: {round(max(0.0, 1.0 - score) * 100, 1)}%"
            ),
        },
    }


def _build_shap_section(shap: dict) -> dict:
    """
    Shape the SHAPExplanation data for the developer report.
    Returns schema fields + sorted feature_importance list for chart rendering.
    """
    if not shap:
        return {}

    fi = shap.get("feature_importance", {})
    # Sort descending and convert to list of {feature, shap_value}
    sorted_features = [
        {"feature": f, "shap_value": v}
        for f, v in sorted(fi.items(), key=lambda x: x[1], reverse=True)
    ]

    return {
        "top_feature":        shap.get("topFeature"),
        "shap_max":           shap.get("shapMax"),
        "shap_min":           shap.get("shapMin"),
        "feature_stability":  shap.get("featureStability"),
        "feature_importance": sorted_features,   # for beeswarm / bar chart
        "feature_names":      [f["feature"] for f in sorted_features],
        "shap_values":        [f["shap_value"] for f in sorted_features],
    }


def _build_model_performance(ev: dict) -> dict:
    """
    Build performance section from model_metrics (accuracy, f1, roc_auc).
    Confusion matrix values are not available server-side without ground truth,
    so we expose whatever the evaluation computed.
    """
    mm = ev.get("model_metrics", {})
    return {
        "accuracy": mm.get("accuracy"),
        "f1_score": mm.get("f1_score"),
        "roc_auc":  mm.get("roc_auc"),
        # Confusion matrix requires raw counts — expose as null until
        # evaluation route stores them; frontend should handle null gracefully.
        "confusion_matrix": {
            "true_positives":  None,
            "false_positives": None,
            "false_negatives": None,
            "true_negatives":  None,
        },
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@developer_report_routes.route("/report/developer/<eval_id>", methods=["GET"])
def get_developer_report(eval_id: str):
    """
    GET /report/developer/<eval_id>

    Full developer report payload for page5.html.

    Response shape:
    {
        evaluation_id:   str,
        status:          str,
        ethical_score:   float,
        report_type:     str,
        records:         int,
        model_id:        str,

        shap: {
            top_feature, shap_max, shap_min, feature_stability,
            feature_importance: [{ feature, shap_value }, ...],
            feature_names:      [str, ...],
            shap_values:        [float, ...],
        },

        dimensions: [
            {
                id, icon, title, score, status, style,
                detail: { description, ... dimension-specific fields ... }
            },
            ...   (7 total)
        ],

        model_performance: {
            accuracy, f1_score, roc_auc,
            confusion_matrix: { tp, fp, fn, tn }
        },

        sensitive_attributes: [{ name }, ...],
        fairness_weights:     [{ dimension, weight }, ...],
        per_attribute:        { attr: { dp_diff, eo_diff, fairness_score } },
    }
    """
    ev = evaluations.get(eval_id)
    if ev is None:
        return jsonify({"error": f"Evaluation '{eval_id}' not found"}), 404

    status = ev.get("status", "queued")

    if status in ("queued", "running"):
        return jsonify({
            "evaluation_id": eval_id,
            "status":        status,
            "current_step":  ev.get("current_step", 0),
        }), 200

    if status == "error":
        return jsonify({
            "evaluation_id": eval_id,
            "status":        "error",
            "error":         ev.get("error", "Unknown error"),
        }), 200

    fairness = ev.get("fairness", {})
    shap     = ev.get("shap", {})
    
    # 🔥 Detect real bias using counterfactual
    counterfactual_score = fairness.get("counterfactual", 1.0)
    bias_detected = any([
        fairness.get("counterfactual", 1.0) < 0.95,
        fairness.get("demographicParity", 0.0) > 0.05,
        fairness.get("groupFairness", 1.0) < 0.9
    ])

    dimensions = [
        _build_individual_fairness(fairness),
        _build_group_fairness(fairness),
        _build_demographic_bias(fairness),
        _build_calibration(fairness),
        _build_disparate_impact(fairness),
        _build_counterfactual(fairness),
        _build_intersectional(fairness),
    ]

    payload = {
        "evaluation_id":  eval_id,
        "status":         "complete",
        "ethical_score":  ev.get("ethical_score", 0.0),
        "report_type":    ev.get("report_type", "DEVELOPER"),
        "records":        ev.get("records", 0),
        "model_id":       ev.get("model_id", ""),

        # SHAP section (SHAPExplanation schema + chart-ready arrays)
        "shap":            _build_shap_section(shap),

        # 7 fairness dimensions with full detail
        "dimensions":      dimensions,

        # Model performance (accuracy, f1, roc_auc)
        "model_performance": _build_model_performance(ev),

        # Raw per-attribute breakdown for any extra tables
        "per_attribute":   fairness.get("per_attribute", {}),

        # SensitiveAttribute[] and FairnessWeight[] records
        "sensitive_attributes": ev.get("sensitive_attributes", []),
        "fairness_weights":     ev.get("fairness_weights", []),

        # Raw FairnessMetrics for any custom frontend use
        "fairness_raw": {
            "individualFairness": fairness.get("individualFairness"),
            "groupFairness":      fairness.get("groupFairness"),
            "demographicParity":  fairness.get("demographicParity"),
            "disparateImpact":    fairness.get("disparateImpact"),
            "calibrationError":   fairness.get("calibrationError"),
            "counterfactual":     fairness.get("counterfactual"),
            "intersectional":     fairness.get("intersectional"),
        },
        "bias_detected": bias_detected,
    }

    return jsonify(payload), 200