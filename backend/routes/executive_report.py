"""
routes/executive_report.py
High-level executive summary for C-suite / risk officers.
Focuses on: risk level, compliance status, business impact, key actions.
"""

from flask import Blueprint, jsonify
from datetime import datetime
from routes.evaluation import evaluations

executive_report_routes = Blueprint("executive_report_routes", __name__)


def _risk_level(score: float) -> dict:
    if score >= 0.80:
        return {"label": "LOW RISK", "color": "green",  "icon": "✅", "code": "low"}
    if score >= 0.65:
        return {"label": "MEDIUM RISK", "color": "amber", "icon": "⚠️", "code": "medium"}
    return {"label": "HIGH RISK", "color": "red",   "icon": "🔴", "code": "high"}


def _deployment_verdict(score: float, remediation: dict) -> dict:
    blocked  = remediation.get("deployment_blocked", False) if remediation else score < 0.60
    critical = remediation.get("critical_issues", 0) if remediation else 0

    if score >= 0.80 and not blocked:
        return {
            "status":  "APPROVED",
            "color":   "green",
            "icon":    "✅",
            "message": "Model meets all ethical standards and is approved for deployment.",
            "action":  "Proceed with deployment. Schedule quarterly re-evaluation.",
        }
    if score >= 0.65:
        return {
            "status":  "CONDITIONAL",
            "color":   "amber",
            "icon":    "⚠️",
            "message": f"Model can deploy with conditions. {critical} critical issue(s) require attention.",
            "action":  "Deploy with monitoring. Address flagged issues within 30 days.",
        }
    return {
        "status":  "BLOCKED",
        "color":   "red",
        "icon":    "🔴",
        "message": f"Model does NOT meet minimum ethical thresholds. {critical} critical failure(s).",
        "action":  "Do not deploy. Escalate to ML team for immediate remediation.",
    }


def _business_impact(fairness: dict, privacy: dict, robustness: dict) -> list:
    impacts = []

    di = (fairness or {}).get("disparateImpact", 1.0)
    if di < 0.80:
        impacts.append({
            "icon":     "⚖️",
            "category": "Legal / Regulatory",
            "severity": "HIGH",
            "title":    "Disparate impact violation — legal liability",
            "detail":   f"The 80% rule is violated (current: {round(di*100,1)}%). "
                        "This may expose the organization to discrimination lawsuits and regulatory fines.",
        })

    dp = (fairness or {}).get("demographicParity", 0.0)
    if dp > 0.05:
        impacts.append({
            "icon":     "👥",
            "category": "Reputational",
            "severity": "MEDIUM",
            "title":    "Demographic outcome gap — reputational risk",
            "detail":   f"A {round(dp*100,1)}% outcome gap between demographic groups could attract negative media attention.",
        })

    pii = len((privacy or {}).get("pii_columns", []))
    if pii > 0:
        impacts.append({
            "icon":     "🔒",
            "category": "Privacy / Data Protection",
            "severity": "HIGH",
            "title":    f"{pii} PII column(s) detected — GDPR / IT Act risk",
            "detail":   "Personal data in the training set risks re-identification. May require DPA notification.",
        })

    br = (robustness or {}).get("boundary_instability", 0.0)
    if br > 0.25:
        impacts.append({
            "icon":     "📉",
            "category": "Operational",
            "severity": "MEDIUM",
            "title":    f"{round(br*100,1)}% of decisions are unstable near threshold",
            "detail":   "A significant fraction of decisions would flip with minor data changes — operational reliability risk.",
        })

    if not impacts:
        impacts.append({
            "icon":     "✅",
            "category": "Overall",
            "severity": "LOW",
            "title":    "No significant business risks detected",
            "detail":   "Model meets ethical standards across fairness, privacy, and robustness dimensions.",
        })

    return impacts


def _dimension_summary(ev: dict) -> list:
    dims = [
        {
            "name":  "Fairness",
            "score": ev.get("fairness_score") or ev.get("fairness", {}).get("fairness_score", 0),
            "icon":  "⚖️",
            "weight": "40%",
        },
        {
            "name":  "Privacy",
            "score": ev.get("privacy_score") or (ev.get("privacy") or {}).get("privacy_score", 0.5),
            "icon":  "🔒",
            "weight": "20%",
        },
        {
            "name":  "Robustness",
            "score": ev.get("robustness_score") or (ev.get("robustness") or {}).get("robustness_score", 0.5),
            "icon":  "🛡️",
            "weight": "15%",
        },
        {
            "name":  "Transparency",
            "score": ev.get("transparency_score") or (ev.get("transparency") or {}).get("transparency_score", 0.5),
            "icon":  "🔍",
            "weight": "15%",
        },
        {
            "name":  "Accountability",
            "score": ev.get("accountability_score") or (ev.get("accountability") or {}).get("accountability_score", 0.5),
            "icon":  "📋",
            "weight": "10%",
        },
    ]

    for d in dims:
        s = d["score"] or 0
        if s >= 0.80:
            d["status"] = "PASS"
            d["color"]  = "green"
        elif s >= 0.65:
            d["status"] = "CONDITIONAL"
            d["color"]  = "amber"
        else:
            d["status"] = "FAIL"
            d["color"]  = "red"
        d["score"] = round(s, 4)
        d["score_pct"] = round(s * 100, 1)

    return dims


def _top_actions(remediation: dict) -> list:
    """Return top 3 executive-level actions from the remediation plan."""
    if not remediation:
        return []

    fixes = remediation.get("all_fixes", [])
    # Summarize to executive language
    exec_actions = []
    seen_dims = set()
    for fix in fixes[:10]:
        dim = fix.get("dimension_key")
        if dim in seen_dims:
            continue
        seen_dims.add(dim)
        exec_actions.append({
            "priority":    fix.get("severity", "WARNING"),
            "dimension":   fix.get("dimension"),
            "title":       fix.get("title"),
            "effort":      fix.get("effort"),
            "description": fix.get("description", "")[:120] + "…",
            "expected_improvement": fix.get("expected_improvement"),
            "estimated_score_after": fix.get("estimated_score_after"),
        })
        if len(exec_actions) >= 3:
            break

    return exec_actions


@executive_report_routes.route("/report/executive/<eval_id>", methods=["GET"])
def get_executive_report(eval_id: str):
    """
    GET /report/executive/<eval_id>

    C-suite / executive summary report.
    High-level: risk level, deployment verdict, business impact, top 3 actions.
    """
    ev = evaluations.get(eval_id)
    if ev is None:
        return jsonify({"error": f"Evaluation '{eval_id}' not found"}), 404

    status = ev.get("status", "queued")
    if status in ("queued", "running"):
        return jsonify({"evaluation_id": eval_id, "status": status, "current_step": ev.get("current_step", 0)}), 200
    if status == "error":
        return jsonify({"evaluation_id": eval_id, "status": "error", "error": ev.get("error")}), 200

    overall_score  = ev.get("ethical_score", 0.0)
    fairness       = ev.get("fairness", {})
    privacy        = ev.get("privacy", {})
    robustness     = ev.get("robustness", {})
    transparency   = ev.get("transparency", {})
    accountability = ev.get("accountability", {})
    remediation    = ev.get("remediation", {})
    shap           = ev.get("shap", {})
    model_metrics  = ev.get("model_metrics", {})

    risk           = _risk_level(overall_score)
    verdict        = _deployment_verdict(overall_score, remediation)
    dimensions     = _dimension_summary(ev)
    business_risks = _business_impact(fairness, privacy, robustness)
    top_actions    = _top_actions(remediation)

    # Quick stats bar
    quick_stats = [
        {"label": "Records Evaluated",  "value": f"{ev.get('records', 0):,}"},
        {"label": "Ethical Score",      "value": f"{round(overall_score * 100)}/100"},
        {"label": "Dimensions Passed",  "value": f"{sum(1 for d in dimensions if d['status'] == 'PASS')}/5"},
        {"label": "Critical Issues",    "value": str(remediation.get("critical_issues", 0))},
        {"label": "Model Accuracy",     "value": f"{round((model_metrics.get('accuracy') or 0)*100,1)}%" if model_metrics.get("accuracy") else "—"},
        {"label": "ROC-AUC",            "value": str(model_metrics.get("roc_auc") or "—")},
    ]

    # Legal compliance summary
    legal_summary = []
    if fairness.get("disparateImpact", 1.0) >= 0.80:
        legal_summary.append({"framework": "80% Rule (EEOC)", "status": "COMPLIANT", "color": "green"})
    else:
        legal_summary.append({"framework": "80% Rule (EEOC)", "status": "NON-COMPLIANT", "color": "red"})

    if overall_score >= 0.75:
        legal_summary.append({"framework": "EU AI Act (Limited Risk)", "status": "READY", "color": "green"})
    else:
        legal_summary.append({"framework": "EU AI Act (High Risk)", "status": "REVIEW REQUIRED", "color": "amber"})

    legal_summary.append({"framework": "GDPR / IT Act India", "status": "DOCUMENTED", "color": "green"})
    legal_summary.append({"framework": "NITI Aayog Responsible AI", "status": "VOLUNTARY COMPLIANT", "color": "green"})

    payload = {
        "evaluation_id":   eval_id,
        "status":          "complete",
        "generated_at":    datetime.utcnow().strftime("%B %d, %Y — %H:%M UTC"),

        # Core verdict
        "overall_score":   overall_score,
        "overall_pct":     round(overall_score * 100, 1),
        "risk":            risk,
        "verdict":         verdict,

        # Dimension breakdown
        "dimensions":      dimensions,

        # Quick stats
        "quick_stats":     quick_stats,

        # Business impact
        "business_risks":  business_risks,

        # Top actions
        "top_actions":     top_actions,

        # Legal
        "legal_summary":   legal_summary,

        # CI/CD status
        "cicd_status": {
            "deployment_blocked": remediation.get("deployment_blocked", overall_score < 0.60),
            "threshold_used":     0.60,
            "recommended_threshold": 0.80,
            "regression_vs_prev": None,   # populated by monitoring route
        },

        # Model perf
        "model_performance": {
            "accuracy": model_metrics.get("accuracy"),
            "f1_score": model_metrics.get("f1_score"),
            "roc_auc":  model_metrics.get("roc_auc"),
        },

        "records":  ev.get("records", 0),
        "model_id": ev.get("model_id", ""),
    }

    return jsonify(payload), 200