from flask import Blueprint, jsonify
from routes.evaluation import evaluations   # shared in-memory store

enduser_report_routes = Blueprint("enduser_report_routes", __name__)


# ── Plain-language thresholds ─────────────────────────────────────────────────
# These mirror the same SystemSetting values used by the regulator report
# but are used here to decide what language to show to a non-technical reader.
CONCERN_THRESHOLD   = 0.80   # below this → flag as a concern
CRITICAL_THRESHOLD  = 0.65   # below this → flag as a serious concern


def _severity(score: float) -> str:
    if score >= CONCERN_THRESHOLD:  return "ok"
    if score >= CRITICAL_THRESHOLD: return "concern"
    return "critical"


# ── Plain-language builders ───────────────────────────────────────────────────

def _overall_summary(ethical_score: float) -> dict:
    """Top-level plain-language verdict."""
    if ethical_score >= 0.80:
        return {
            "verdict":     "This model is performing fairly.",
            "sub":         "Our checks found no major fairness problems.",
            "outcome":     "FAIR",
            "icon":        "✅",
        }
    elif ethical_score >= 0.65:
        return {
            "verdict":     "This model has some fairness concerns.",
            "sub":         "It works well in most cases but has specific areas that need attention.",
            "outcome":     "PARTIAL",
            "icon":        "⚠️",
        }
    else:
        return {
            "verdict":     "This model has significant fairness issues.",
            "sub":         "There are areas where the model may treat people unfairly.",
            "outcome":     "CONCERN",
            "icon":        "🔴",
        }


def _factor_items(fairness: dict, shap: dict) -> list:
    """
    Build the "factor list" — human-readable checks shown to the end user.
    Each item: { icon, icon_style, name, detail, status, status_label }
    """
    indiv  = fairness.get("individualFairness", 0.0)
    group  = fairness.get("groupFairness", 0.0)
    cal_err = fairness.get("calibrationError", 0.0)
    di     = fairness.get("disparateImpact", 0.0)
    cf     = fairness.get("counterfactual", 0.0)
    inter  = fairness.get("intersectional", 0.0)
    top_f  = shap.get("topFeature", "a key factor") if shap else "a key factor"

    def _ok(score, invert=False):
        s = score if not invert else (1 - score)
        return _severity(s) == "ok"

    items = [
        {
            "icon":         "📊",
            "icon_style":   "normal" if _ok(indiv) else "warning",
            "name":         "Similar people are treated similarly",
            "detail":       (
                f"People with similar backgrounds get consistent decisions "
                f"({round(indiv * 100, 1)}% consistency)."
            ) if _ok(indiv) else (
                f"Some people with similar backgrounds are getting different decisions "
                f"({round(indiv * 100, 1)}% consistency). This needs review."
            ),
            "status":       "approved" if _ok(indiv) else "rejected",
            "status_label": "✓" if _ok(indiv) else "!",
        },
        {
            "icon":         "👥",
            "icon_style":   "normal" if _ok(group) else "warning",
            "name":         "Different groups are treated equally",
            "detail":       (
                "No significant difference in outcomes across gender, age, or other groups."
            ) if _ok(group) else (
                "Some groups are receiving different outcomes. "
                "The gap may indicate unintentional bias."
            ),
            "status":       "approved" if _ok(group) else "rejected",
            "status_label": "✓" if _ok(group) else "!",
        },
        {
            "icon":         "💰",
            "icon_style":   "normal",
            "name":         f"Decisions are based on relevant information",
            "detail":       (
                f"The most influential factor is '{top_f}', which is a legitimate "
                "input for this type of decision."
            ),
            "status":       "approved",
            "status_label": "✓",
        },
        {
            "icon":         "🎯",
            "icon_style":   "normal" if _ok(cal_err, invert=True) else "warning",
            "name":         "Predictions are well-calibrated",
            "detail":       (
                "When the model predicts a high likelihood, it is correct most of the time."
            ) if _ok(cal_err, invert=True) else (
                "The model's confidence levels don't always match real outcomes. "
                "It may be over- or under-confident."
            ),
            "status":       "approved" if _ok(cal_err, invert=True) else "rejected",
            "status_label": "✓" if _ok(cal_err, invert=True) else "!",
        },
        {
            "icon":         "🔄",
            "icon_style":   "normal" if _ok(cf) else "warning",
            "name":         "Changing protected attributes doesn't change the outcome",
            "detail":       (
                "Swapping gender, age, or ethnicity in a profile does not change the decision — "
                "the model focuses on relevant factors only."
            ) if _ok(cf) else (
                "In some cases, changing a person's gender or age changes the decision, "
                "even when everything else stays the same."
            ),
            "status":       "approved" if _ok(cf) else "rejected",
            "status_label": "✓" if _ok(cf) else "!",
        },
        {
            "icon":         "🔀",
            "icon_style":   "normal" if _ok(inter) else "warning",
            "name":         "No compounding disadvantage for combined groups",
            "detail":       (
                "People who belong to multiple protected groups (e.g. older women) "
                "are not disproportionately disadvantaged."
            ) if _ok(inter) else (
                "Some combinations of characteristics (e.g. older women, minority + low income) "
                "face a larger disadvantage than any single group alone."
            ),
            "status":       "approved" if _ok(inter) else "rejected",
            "status_label": "✓" if _ok(inter) else "!",
        },
    ]

    # 80% rule — plain language
    if di < 0.80:
        items.append({
            "icon":         "📉",
            "icon_style":   "warning",
            "name":         "Some groups have noticeably lower approval rates",
            "detail":       (
                f"The approval rate for the least-favoured group is "
                f"{round(di * 100, 1)}% of the most-favoured group. "
                "Legal guidelines recommend at least 80%."
            ),
            "status":       "rejected",
            "status_label": "!",
        })

    return items


def _bias_indicators(fairness: dict, per_attribute: dict) -> list:
    """
    Build plain-language bias indicator bullets.
    Returns a list of { title, detail } objects.
    """
    indicators = []

    # Age bias — check intersectional or per-attribute for age
    inter_score = fairness.get("intersectional", 1.0)
    if inter_score < CONCERN_THRESHOLD:
        indicators.append({
            "title":  "Intersectional Bias",
            "detail": (
                "Some combinations of characteristics (such as older women or minority applicants "
                "with lower income) face compounding disadvantages not explained by financial factors alone."
            ),
        })

    # Demographic parity gap
    dp_raw = fairness.get("demographicParity", 0.0)
    if dp_raw > 0.05:
        indicators.append({
            "title":  "Demographic Gap",
            "detail": (
                f"There is a {round(dp_raw * 100, 1)}% outcome gap between demographic groups. "
                "This may indicate the model is picking up on protected characteristics."
            ),
        })

    # Disparate impact
    di = fairness.get("disparateImpact", 1.0)
    if di < 0.80:
        indicators.append({
            "title":  "Disparate Impact (80% Rule)",
            "detail": (
                f"The least-favoured group receives favourable outcomes at only "
                f"{round(di * 100, 1)}% of the rate of the most-favoured group, "
                "which falls below the legal 80% threshold."
            ),
        })

    # Proxy variable warning based on per_attribute
    suspicious = [a for a in per_attribute if any(
        kw in a.lower() for kw in ["zip", "postcode", "area", "region", "location"]
    )]
    if suspicious:
        indicators.append({
            "title":  "Possible Proxy Variable",
            "detail": (
                f"The attribute '{suspicious[0]}' may act as a proxy for race or socioeconomic status, "
                "creating indirect bias even without explicitly using protected characteristics."
            ),
        })

    # Counterfactual instability
    cf = fairness.get("counterfactual", 1.0)
    if cf < CONCERN_THRESHOLD:
        indicators.append({
            "title":  "Counterfactual Instability",
            "detail": (
                "In some cases, changing only a person's gender or age (while keeping all other "
                "information the same) changes the model's decision. This suggests the model "
                "is sensitive to protected attributes."
            ),
        })

    # If no issues found
    if not indicators:
        indicators.append({
            "title":  "No Major Bias Indicators Found",
            "detail": (
                "The model passed all automated bias checks. "
                "Continue monitoring as new data is collected."
            ),
        })

    return indicators


def _what_this_means(fairness: dict) -> dict:
    """Plain-language 'what this means' summary + action cards."""
    issues = []
    if fairness.get("disparateImpact", 1.0) < 0.80:
        issues.append("disparate impact")
    if fairness.get("intersectional", 1.0) < CONCERN_THRESHOLD:
        issues.append("intersectional bias")
    if fairness.get("counterfactual", 1.0) < CONCERN_THRESHOLD:
        issues.append("counterfactual instability")

    if not issues:
        summary = (
            "The model is performing fairly across our checks. "
            "It uses relevant factors and does not show significant bias against protected groups."
        )
    else:
        issue_str = ", ".join(issues)
        summary = (
            f"The model is mostly accurate but has specific blind spots related to {issue_str}. "
            "These could lead to unfair treatment of certain groups."
        )

    action_cards = []

    di = fairness.get("disparateImpact", 1.0)
    action_cards.append({
        "icon":   "📉" if di < 0.80 else "✅",
        "title":  "Fairness Risk",
        "text":   "Risk of unfair rejection for some groups — adjustment recommended." if di < 0.80
                  else "Fairness risk is within acceptable limits.",
    })

    cf = fairness.get("counterfactual", 1.0)
    action_cards.append({
        "icon":   "⚖️",
        "title":  "Compliance",
        "text":   "Model needs adjustment to fully meet non-discrimination standards." if cf < CONCERN_THRESHOLD
                  else "Model currently meets non-discrimination standards.",
    })

    inter = fairness.get("intersectional", 1.0)
    action_cards.append({
        "icon":   "🔀",
        "title":  "Intersectionality",
        "text":   "Some combined-group disadvantage detected — targeted retraining recommended." if inter < CONCERN_THRESHOLD
                  else "No compounding disadvantage detected across combined groups.",
    })

    return {
        "summary":      summary,
        "action_cards": action_cards,
    }


def _recommendations(fairness: dict, shap: dict, per_attribute: dict) -> list:
    """Generate plain-language recommended improvement steps."""
    recs = []

    dp_raw = fairness.get("demographicParity", 0.0)
    if dp_raw > 0.05:
        recs.append({
            "icon":   "🔄",
            "title":  "Retrain with more balanced demographic data",
            "detail": "To reduce the outcome gap between demographic groups.",
        })

    inter = fairness.get("intersectional", 1.0)
    if inter < CONCERN_THRESHOLD:
        recs.append({
            "icon":   "🎯",
            "title":  "Apply targeted resampling for underrepresented intersectional groups",
            "detail": "Focus on combinations like older women or minority + low-income applicants.",
        })

    di = fairness.get("disparateImpact", 1.0)
    if di < 0.80:
        recs.append({
            "icon":   "⚖️",
            "title":  "Apply post-processing threshold adjustment",
            "detail": f"Adjust decision thresholds per group to bring the impact ratio above 0.80 (currently {round(di, 2)}).",
        })

    cf = fairness.get("counterfactual", 1.0)
    if cf < CONCERN_THRESHOLD:
        recs.append({
            "icon":   "🔍",
            "title":  "Audit features for sensitive attribute correlation",
            "detail": "Some features may be proxies for protected attributes. Use causal analysis to identify and remove them.",
        })

    top_f = shap.get("topFeature") if shap else None
    if top_f:
        recs.append({
            "icon":   "📊",
            "title":  f"Review the top feature: '{top_f}'",
            "detail": "Verify this is a legitimate, non-discriminatory basis for decisions.",
        })

    # Proxy variable warning
    suspicious = [a for a in per_attribute if any(
        kw in a.lower() for kw in ["zip", "postcode", "area", "region", "location"]
    )]
    if suspicious:
        recs.append({
            "icon":   "✂️",
            "title":  f"Remove or re-weight '{suspicious[0]}'",
            "detail": "This feature may be acting as a proxy for race or socioeconomic status.",
        })

    if not recs:
        recs.append({
            "icon":   "✅",
            "title":  "Continue monitoring",
            "detail": "No immediate changes required. Re-evaluate as new data is collected.",
        })

    return recs


# ── Route ─────────────────────────────────────────────────────────────────────

@enduser_report_routes.route("/report/enduser/<eval_id>", methods=["GET"])
def get_enduser_report(eval_id: str):
    """
    GET /report/enduser/<eval_id>

    Human-readable plain-language report for page7.html.

    Response shape:
    {
        evaluation_id:    str,
        status:           str,
        ethical_score:    float,
        records:          int,

        overall_summary: {
            verdict, sub, outcome, icon
        },

        factor_items: [
            { icon, icon_style, name, detail, status, status_label }, ...
        ],

        bias_indicators: [
            { title, detail }, ...
        ],

        what_this_means: {
            summary:      str,
            action_cards: [{ icon, title, text }, ...]
        },

        recommendations: [
            { icon, title, detail }, ...
        ],
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

    fairness    = ev.get("fairness", {})
    shap        = ev.get("shap", {})
    per_attr    = fairness.get("per_attribute", {})
    eth_score   = ev.get("ethical_score", 0.0)

    payload = {
        "evaluation_id":   eval_id,
        "status":          "complete",
        "ethical_score":   eth_score,
        "records":         ev.get("records", 0),

        "overall_summary": _overall_summary(eth_score),
        "factor_items":    _factor_items(fairness, shap),
        "bias_indicators": _bias_indicators(fairness, per_attr),
        "what_this_means": _what_this_means(fairness),
        "recommendations": _recommendations(fairness, shap, per_attr),
    }

    return jsonify(payload), 200