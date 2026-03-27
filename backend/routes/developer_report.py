from flask import Blueprint, jsonify
from routes.evaluation import evaluations   # shared in-memory store

developer_report_routes = Blueprint("developer_report_routes", __name__)


def _badge(score: float, threshold: float = 0.80) -> str:
    return "PASS" if score >= threshold else "WARNING"


def _dimension_style(score: float) -> str:
    return "success" if score >= 0.80 else "warning"


# ── Dimension builders ────────────────────────────────────────────────────────

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
    score    = fairness.get("groupFairness", 0.0)
    per_attr = fairness.get("per_attribute", {})

    groups = []
    for attr, metrics in per_attr.items():
        if "error" in metrics:
            continue
        groups.append({
            "attribute":                    attr,
            "equalized_odds_difference":    metrics.get("equalized_odds_difference", None),
            "demographic_parity_difference": metrics.get("demographic_parity_difference", None),
            "fairness_score":               metrics.get("fairness_score", None),
        })

    return {
        "id":     2,
        "icon":   "👪",
        "title":  "Group Fairness",
        "score":  round(score, 4),
        "status": _badge(score),
        "style":  _dimension_style(score),
        "detail": {
            "description": (
                "Evaluates equal opportunity across protected groups. "
                "Reports equalized-odds difference per sensitive attribute."
            ),
            "per_attribute": groups,
        },
    }


def _build_demographic_bias(fairness: dict) -> dict:
    dp_raw   = fairness.get("demographicParity", 0.0)
    dp_score = round(max(0.0, 1.0 - dp_raw), 4)
    per_attr = fairness.get("per_attribute", {})

    disparities = []
    for attr, metrics in per_attr.items():
        if "error" in metrics:
            continue
        dp = abs(metrics.get("demographic_parity_difference", 0.0))
        disparities.append({
            "attribute":   attr,
            "gap_pct":     round(dp * 100, 2),
            "raw_dp_diff": round(metrics.get("demographic_parity_difference", 0.0), 4),
        })

    return {
        "id":     3,
        "icon":   "🌍",
        "title":  "Demographic Bias",
        "score":  dp_score,
        "status": _badge(dp_score),
        "style":  _dimension_style(dp_score),
        "detail": {
            "description": (
                "Measures demographic parity: whether the positive outcome rate is equal "
                "across all groups of each sensitive attribute."
            ),
            "raw_parity_gap": round(dp_raw, 4),
            "disparities":    disparities,
        },
    }


def _build_calibration(fairness: dict) -> dict:
    cal_err   = fairness.get("calibrationError", 0.0)
    cal_score = round(max(0.0, 1.0 - cal_err), 4)
    return {
        "id":     4,
        "icon":   "⚖️",
        "title":  "Calibration",
        "score":  cal_score,
        "status": _badge(cal_score),
        "style":  _dimension_style(cal_score),
        "detail": {
            "description": (
                "Calibration measures whether predicted probabilities match observed frequencies. "
                "Computed as mean Brier score across sensitive groups."
            ),
            "brier_score":     round(cal_err, 4),
            "calibration_score": cal_score,
            "interpretation": (
                "A Brier score of 0 is perfect; 0.25 is no-skill. "
                f"Current score: {round(cal_err, 4)}"
            ),
        },
    }


def _build_disparate_impact(fairness: dict) -> dict:
    di = fairness.get("disparateImpact", 0.0)
    return {
        "id":     5,
        "icon":   "📊",
        "title":  "Disparate Impact",
        "score":  round(di, 4),
        "status": _badge(di, threshold=0.80),
        "style":  _dimension_style(di),
        "detail": {
            "description": (
                "The 80% rule (four-fifths rule): the ratio of the lowest group positive-outcome "
                "rate to the highest. Scores below 0.80 indicate adverse impact."
            ),
            "impact_ratio":  round(di, 4),
            "threshold":     0.80,
            "passes_80_rule": di >= 0.80,
        },
    }


def _build_counterfactual(fairness: dict) -> dict:
    score = fairness.get("counterfactual", 0.0)
    return {
        "id":     6,
        "icon":   "🔄",
        "title":  "Counterfactual Fairness",
        "score":  round(score, 4),
        "status": _badge(score),
        "style":  _dimension_style(score),
        "detail": {
            "description": (
                "Counterfactual fairness checks whether swapping a sensitive attribute "
                "value changes the model's prediction. Measured as 1 minus the std of "
                "group-level positive prediction rates."
            ),
            "consistency_score": round(score, 4),
            "interpretation": (
                "Higher is better. A score of 1.0 means group-level rates are identical."
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
    bias_detected = fairness.get("counterfactual", 1.0) < 0.95

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