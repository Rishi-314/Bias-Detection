from flask import Blueprint, jsonify
from routes.evaluation import evaluations   # shared in-memory store

dashboard_routes = Blueprint("dashboard_routes", __name__)


def _pass_or_warning(score: float, threshold: float = 0.80) -> str:
    return "PASS" if score >= threshold else "WARNING"


def _build_dimensions(fairness: dict) -> list:
    """
    Map FairnessMetrics fields → the 7 dimension cards shown on page4.html.
    Each card: { id, icon, title, score, status, finding }
    """
    di     = fairness.get("disparateImpact", 0.0)
    cal_err = fairness.get("calibrationError", 0.0)
    cal_score = round(1.0 - cal_err, 4)   # invert: lower error = higher score

    dp_raw   = fairness.get("demographicParity", 0.0)   # raw gap, lower = better
    dp_score = round(max(0.0, 1.0 - dp_raw), 4)         # convert to 0-1 score

    indiv  = fairness.get("individualFairness", 0.0)
    group  = fairness.get("groupFairness", 0.0)
    cf     = fairness.get("counterfactual", 0.0)
    inter  = fairness.get("intersectional", 0.0)

    per_attr = fairness.get("per_attribute", {})

    # Build gender-gap finding if available
    gender_gap = None
    for attr in ["gender", "sex", "Gender", "Sex"]:
        if attr in per_attr:
            gap = abs(per_attr[attr].get("demographic_parity_difference", 0.0))
            gender_gap = f"Gender gap: {round(gap * 100, 1)}%"
            break

    # Build disparate-impact finding
    di_finding = f"80% rule: {di:.2f}"

    # Build intersectional finding
    inter_finding = "See intersectional breakdown"
    for attr in per_attr:
        gap = per_attr[attr].get("demographic_parity_difference", None)
        if gap is not None:
            inter_finding = f"{attr}: {round(abs(gap) * 100, 1)}% gap"
            break

    return [
        {
            "id":      1,
            "icon":    "👥",
            "title":   "Individual Fairness",
            "score":   indiv,
            "status":  _pass_or_warning(indiv),
            "finding": f"Consistency: {round(indiv * 100, 1)}%",
        },
        {
            "id":      2,
            "icon":    "👪",
            "title":   "Group Fairness",
            "score":   group,
            "status":  _pass_or_warning(group),
            "finding": gender_gap or f"Equal opportunity score: {group:.2f}",
        },
        {
            "id":      3,
            "icon":    "🌍",
            "title":   "Demographic Bias",
            "score":   dp_score,
            "status":  _pass_or_warning(dp_score),
            "finding": f"Parity gap: {round(dp_raw, 4)}",
        },
        {
            "id":      4,
            "icon":    "⚖️",
            "title":   "Calibration",
            "score":   cal_score,
            "status":  _pass_or_warning(cal_score),
            "finding": f"Brier error: {round(cal_err, 4)}",
        },
        {
            "id":      5,
            "icon":    "📊",
            "title":   "Disparate Impact",
            "score":   di,
            "status":  _pass_or_warning(di, threshold=0.80),
            "finding": di_finding,
        },
        {
            "id":      6,
            "icon":    "🔄",
            "title":   "Counterfactual",
            "score":   cf,
            "status":  _pass_or_warning(cf),
            "finding": f"Consistency: {round(cf * 100, 1)}%",
        },
        {
            "id":      7,
            "icon":    "🔀",
            "title":   "Intersectional",
            "score":   inter,
            "status":  _pass_or_warning(inter),
            "finding": inter_finding,
        },
    ]


def _build_layers(fairness: dict) -> list:
    """
    Map metrics → the 4-layer pyramid shown on page4.html.
    Layer 1 = Systemic, Layer 2 = Group, Layer 3 = Subgroup, Layer 4 = Individual
    """
    indiv  = fairness.get("individualFairness", 0.0)
    group  = fairness.get("groupFairness", 0.0)
    inter  = fairness.get("intersectional", 0.0)

    dp_raw   = fairness.get("demographicParity", 0.0)
    di       = fairness.get("disparateImpact", 0.0)
    systemic = round((group + di) / 2, 4)   # proxy for systemic fairness

    return [
        {
            "layer":       4,
            "title":       "Layer 4: Individual",
            "description": "Person-to-person fairness",
            "score":       indiv,
            "status":      _pass_or_warning(indiv),
        },
        {
            "layer":       3,
            "title":       "Layer 3: Subgroup",
            "description": "Intersectional groups",
            "score":       inter,
            "status":      _pass_or_warning(inter),
        },
        {
            "layer":       2,
            "title":       "Layer 2: Group",
            "description": "Protected attributes",
            "score":       group,
            "status":      _pass_or_warning(group),
        },
        {
            "layer":       1,
            "title":       "Layer 1: Systemic",
            "description": "Institutional patterns",
            "score":       systemic,
            "status":      _pass_or_warning(systemic),
        },
    ]


def _build_insights(fairness: dict, shap: dict) -> dict:
    """
    Build the Quick Insights panel: top 3 strengths, top 3 concerns, recommendations.
    """
    dimension_scores = {
        "Individual Fairness":  fairness.get("individualFairness", 0.0),
        "Group Fairness":       fairness.get("groupFairness", 0.0),
        "Counterfactual":       fairness.get("counterfactual", 0.0),
        "Calibration":          round(1.0 - fairness.get("calibrationError", 0.0), 4),
        "Disparate Impact":     fairness.get("disparateImpact", 0.0),
        "Intersectional":       fairness.get("intersectional", 0.0),
        "Demographic Parity":   round(1.0 - fairness.get("demographicParity", 0.0), 4),
    }

    sorted_dims = sorted(dimension_scores.items(), key=lambda x: x[1], reverse=True)
    strengths   = [f"{name} score: {score:.2f}" for name, score in sorted_dims[:3]]
    concerns    = [f"{name} needs attention (score: {score:.2f})" for name, score in sorted_dims[-3:]]

    # Build recommendation from worst-performing dimension + top SHAP feature
    worst_dim   = sorted_dims[-1][0]
    top_feature = shap.get("topFeature", "key feature") if shap else "key feature"
    recommendations = [
        f"Focus on improving {worst_dim} — consider rebalancing training data",
        f"Top predictive feature is '{top_feature}' — audit for proxy discrimination",
    ]

    return {
        "strengths":       strengths,
        "concerns":        concerns,
        "recommendations": recommendations,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@dashboard_routes.route("/dashboard/<eval_id>", methods=["GET"])
def get_dashboard(eval_id: str):
    """
    GET /dashboard/<eval_id>

    Returns the fully-shaped payload for page4.html.

    Response shape:
    {
        evaluation_id:   str,
        status:          "complete" | "running" | "queued" | "error",
        current_step:    int,           # 0-7, used by progress bar on page3
        ethical_score:   float,         # Evaluation.ethicalScore (0-1)
        overall_status:  "PASS" | "WARNING" | "FAIL",
        report_type:     str,           # Evaluation.reportType enum value
        records:         int,
        model_id:        str,

        dimensions: [                   # 7 cards for dimensions grid
            { id, icon, title, score, status, finding }, ...
        ],

        layers: [                       # 4-layer pyramid
            { layer, title, description, score, status }, ...
        ],

        insights: {
            strengths:       [str, ...],
            concerns:        [str, ...],
            recommendations: [str, ...],
        },

        shap: {                         # SHAPExplanation schema fields
            topFeature, shapMax, shapMin, featureStability,
            feature_importance: { feature: score, ... }
        },

        model_metrics: {                # accuracy, f1, roc_auc
            accuracy, f1_score, roc_auc
        },

        fairness_raw: { ... }           # raw FairnessMetrics for developer view
    }
    """
    ev = evaluations.get(eval_id)
    if ev is None:
        return jsonify({"error": f"Evaluation '{eval_id}' not found"}), 404

    status = ev.get("status", "queued")

    # ── Return progress stub while still running ─────────────────────────
    if status in ("queued", "running"):
        return jsonify({
            "evaluation_id": eval_id,
            "status":        status,
            "current_step":  ev.get("current_step", 0),
            "records":       ev.get("records", 0),
        }), 200

    # ── Propagate errors ─────────────────────────────────────────────────
    if status == "error":
        return jsonify({
            "evaluation_id": eval_id,
            "status":        "error",
            "error":         ev.get("error", "Unknown error"),
        }), 200

    # ── Build full dashboard payload ─────────────────────────────────────
    fairness = ev.get("fairness", {})
    shap     = ev.get("shap", {})

    ethical_score = ev.get("ethical_score", 0.0)

    if ethical_score >= 0.80:
        overall_status = "PASS"
    elif ethical_score >= 0.60:
        overall_status = "WARNING"
    else:
        overall_status = "FAIL"

    payload = {
        "evaluation_id":  eval_id,
        "status":         "complete",
        "current_step":   7,
        "ethical_score":  ethical_score,
        "overall_status": overall_status,
        "report_type":    ev.get("report_type", "DEVELOPER"),
        "records":        ev.get("records", 0),
        "model_id":       ev.get("model_id", ""),

        # 7-dimension cards
        "dimensions":     _build_dimensions(fairness),

        # 4-layer pyramid
        "layers":         _build_layers(fairness),

        # Quick insights panel
        "insights":       _build_insights(fairness, shap),

        # SHAP explanation (SHAPExplanation schema)
        "shap":           shap,

        # Model performance metrics
        "model_metrics":  ev.get("model_metrics", {}),

        # Raw fairness metrics for developer tab
        "fairness_raw":   fairness,
    }

    return jsonify(payload), 200


@dashboard_routes.route("/dashboard", methods=["GET"])
def list_dashboards():
    """
    GET /dashboard

    Lists all completed evaluations with summary data — useful for a
    history/list view that links back to individual dashboard pages.
    """
    summaries = []
    for eid, ev in evaluations.items():
        score = ev.get("ethical_score")
        if score is None:
            continue   # skip incomplete

        if score >= 0.80:
            overall = "PASS"
        elif score >= 0.60:
            overall = "WARNING"
        else:
            overall = "FAIL"

        summaries.append({
            "evaluation_id":  eid,
            "ethical_score":  score,
            "overall_status": overall,
            "report_type":    ev.get("report_type"),
            "records":        ev.get("records", 0),
            "model_id":       ev.get("model_id", ""),
            "status":         ev.get("status"),
        })

    return jsonify({"dashboards": summaries, "count": len(summaries)}), 200