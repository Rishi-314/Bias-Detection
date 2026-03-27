from flask import Blueprint, jsonify, request
from routes.evaluation import evaluations   # shared in-memory store

comparison_routes = Blueprint("comparison_routes", __name__)

DIMENSION_KEYS = [
    ("individualFairness", "Individual Fairness"),
    ("groupFairness",      "Group Fairness"),
    ("demographicParity",  "Demographic Bias"),   # converted to score below
    ("calibrationError",   "Calibration"),         # converted to score below
    ("disparateImpact",    "Disparate Impact"),
    ("counterfactual",     "Counterfactual"),
    ("intersectional",     "Intersectional"),
]

# Dimensions where lower raw value = better (we invert to a 0-1 score)
INVERT_KEYS = {"demographicParity", "calibrationError"}


def _dim_score(fairness: dict, key: str) -> float:
    """Return a 0-1 score for a dimension (inverting where needed)."""
    v = fairness.get(key, 0.0)
    if key in INVERT_KEYS:
        return round(max(0.0, 1.0 - v), 4)
    return round(float(v), 4)


def _eval_summary(eval_id: str, ev: dict) -> dict:
    """Compact summary card for a single evaluation."""
    fairness = ev.get("fairness", {})
    return {
        "evaluation_id": eval_id,
        "ethical_score": ev.get("ethical_score", 0.0),
        "status":        ev.get("status"),
        "report_type":   ev.get("report_type"),
        "model_id":      ev.get("model_id", ""),
        "records":       ev.get("records", 0),
        "dimensions": {
            label: _dim_score(fairness, key)
            for key, label in DIMENSION_KEYS
        },
    }


def _delta_label(delta: float) -> str:
    if delta > 0:
        return f"+{round(delta, 4)} ↑"
    if delta < 0:
        return f"{round(delta, 4)} ↓"
    return "0.0000 —"


def _delta_class(delta: float) -> str:
    if delta > 0.001:  return "improvement"
    if delta < -0.001: return "regression"
    return "neutral"


def _verdict(score_a: float, score_b: float, id_a: str, id_b: str) -> dict:
    """Plain-language verdict for the comparison."""
    delta = score_b - score_a
    if abs(delta) < 0.005:
        return {
            "icon":    "⚖️",
            "title":   "These models perform similarly",
            "detail":  "No meaningful difference in overall fairness score.",
            "winner":  None,
            "badge":   "No clear winner",
        }
    winner_id    = id_b if delta > 0 else id_a
    winner_short = winner_id[:8] + "…"
    improvement  = abs(round(delta * 100, 1))
    return {
        "icon":    "🏆",
        "title":   f"Evaluation {winner_short} is recommended",
        "detail":  f"Shows consistent improvement (+{improvement}% fairness score).",
        "winner":  winner_id,
        "badge":   f"Eval {winner_short} is recommended",
    }


def _build_alerts(ev: dict, eval_id: str) -> list:
    """
    Map Alert[] schema records (or derive from fairness metrics) to alert items.
    Each: { id, message, severity, resolved, eval_id }
    """
    stored = ev.get("alerts", [])   # populated if Prisma Alert records exist
    if stored:
        return stored

    # Derive alerts from fairness metrics when no stored alerts exist
    fairness = ev.get("fairness", {})
    alerts   = []

    di = fairness.get("disparateImpact", 1.0)
    if di < 0.80:
        alerts.append({
            "id":       f"{eval_id[:8]}-di",
            "message":  f"Disparate impact ratio {round(di, 4)} is below the 0.80 threshold",
            "severity": "CRITICAL",
            "resolved": False,
            "eval_id":  eval_id,
        })

    dp = fairness.get("demographicParity", 0.0)
    if dp > 0.05:
        alerts.append({
            "id":       f"{eval_id[:8]}-dp",
            "message":  f"Demographic parity gap of {round(dp * 100, 1)}% exceeds 5% threshold",
            "severity": "WARNING",
            "resolved": False,
            "eval_id":  eval_id,
        })

    inter = fairness.get("intersectional", 1.0)
    if inter < 0.80:
        alerts.append({
            "id":       f"{eval_id[:8]}-inter",
            "message":  f"Intersectional fairness score {round(inter, 4)} below acceptable level",
            "severity": "WARNING",
            "resolved": False,
            "eval_id":  eval_id,
        })

    cf = fairness.get("counterfactual", 1.0)
    if cf < 0.80:
        alerts.append({
            "id":       f"{eval_id[:8]}-cf",
            "message":  f"Counterfactual fairness score {round(cf, 4)} — model sensitive to protected attributes",
            "severity": "WARNING",
            "resolved": False,
            "eval_id":  eval_id,
        })

    if not alerts:
        alerts.append({
            "id":       f"{eval_id[:8]}-ok",
            "message":  "No fairness alerts — all metrics within acceptable thresholds",
            "severity": "INFO",
            "resolved": True,
            "eval_id":  eval_id,
        })

    return alerts


# ── Routes ────────────────────────────────────────────────────────────────────

@comparison_routes.route("/compare/list", methods=["GET"])
def list_evaluations_for_comparison():
    """
    GET /compare/list

    Returns all completed evaluations as summary cards for the version
    selector dropdowns on page8.html.

    Response:
    {
        evaluations: [
            { evaluation_id, ethical_score, status, report_type, model_id, records,
              dimensions: { "Individual Fairness": float, ... } }
        ],
        count: int
    }
    """
    completed = [
        _eval_summary(eid, ev)
        for eid, ev in evaluations.items()
        if ev.get("status") == "complete"
    ]
    return jsonify({"evaluations": completed, "count": len(completed)}), 200


@comparison_routes.route("/compare", methods=["GET"])
def compare_two_evaluations():
    """
    GET /compare?eval_a=<id>&eval_b=<id>

    Side-by-side comparison of two evaluations.

    Response:
    {
        eval_a: { evaluation_id, ethical_score, records, model_id, dimensions: {...} },
        eval_b: { ... },
        delta:  [
            { dimension, score_a, score_b, delta, delta_label, delta_class }, ...
        ],
        verdict: { icon, title, detail, winner, badge },
        overall_delta: float,
    }
    """
    eval_a_id = request.args.get("eval_a")
    eval_b_id = request.args.get("eval_b")

    if not eval_a_id or not eval_b_id:
        return jsonify({"error": "Both eval_a and eval_b query params are required"}), 400

    ev_a = evaluations.get(eval_a_id)
    ev_b = evaluations.get(eval_b_id)

    missing = []
    if ev_a is None: missing.append(eval_a_id)
    if ev_b is None: missing.append(eval_b_id)
    if missing:
        return jsonify({"error": f"Evaluation(s) not found: {', '.join(missing)}"}), 404

    for label, ev in [("eval_a", ev_a), ("eval_b", ev_b)]:
        if ev.get("status") != "complete":
            return jsonify({"error": f"{label} is not yet complete (status: {ev.get('status')})"}), 400

    fairness_a = ev_a.get("fairness", {})
    fairness_b = ev_b.get("fairness", {})

    rows = []
    for key, label in DIMENSION_KEYS:
        sa = _dim_score(fairness_a, key)
        sb = _dim_score(fairness_b, key)
        d  = round(sb - sa, 4)
        rows.append({
            "dimension":   label,
            "score_a":     sa,
            "score_b":     sb,
            "delta":       d,
            "delta_label": _delta_label(d),
            "delta_class": _delta_class(d),
        })

    score_a = ev_a.get("ethical_score", 0.0)
    score_b = ev_b.get("ethical_score", 0.0)

    return jsonify({
        "eval_a":        _eval_summary(eval_a_id, ev_a),
        "eval_b":        _eval_summary(eval_b_id, ev_b),
        "delta":         rows,
        "verdict":       _verdict(score_a, score_b, eval_a_id, eval_b_id),
        "overall_delta": round(score_b - score_a, 4),
    }), 200


@comparison_routes.route("/compare/history", methods=["GET"])
def retraining_history():
    """
    GET /compare/history

    Chronological list of all completed evaluations for the retraining history
    timeline on page8.html. Sorted newest first.

    Response:
    {
        history: [
            {
                evaluation_id, ethical_score, model_id, records,
                timestamp,       # ISO string
                improvement,     # delta vs previous eval (null for first)
                improvement_pct, # percentage change
                trigger,         # "Initial training" | "Re-evaluation" | "Drift detected" (heuristic)
            }
        ],
        trend: [ float, ... ]   # ethical scores oldest→newest for sparkline chart
    }
    """
    completed = [
        (eid, ev) for eid, ev in evaluations.items()
        if ev.get("status") == "complete"
    ]

    # Sort by insertion order (Python dict preserves order) — no timestamps stored,
    # so we use dict order as a proxy. Newest last → reverse for display.
    history = []
    for i, (eid, ev) in enumerate(completed):
        history.append({
            "evaluation_id": eid,
            "ethical_score": ev.get("ethical_score", 0.0),
            "model_id":      ev.get("model_id", ""),
            "records":       ev.get("records", 0),
            "report_type":   ev.get("report_type", "DEVELOPER"),
            # Heuristic trigger label
            "trigger": "Initial training" if i == 0 else (
                "Drift detected" if i % 3 == 1 else "Re-evaluation / Scheduled"
            ),
        })

    # Compute improvement deltas
    for i, item in enumerate(history):
        if i == 0:
            item["improvement"]     = None
            item["improvement_pct"] = None
        else:
            prev  = history[i - 1]["ethical_score"]
            curr  = item["ethical_score"]
            delta = round(curr - prev, 4)
            item["improvement"]     = delta
            item["improvement_pct"] = round(delta * 100, 2)

    # Trend array oldest→newest for sparkline / bar chart
    trend = [item["ethical_score"] for item in history]

    # Reverse for display (newest first)
    history.reverse()

    return jsonify({"history": history, "trend": trend}), 200


@comparison_routes.route("/compare/alerts", methods=["GET"])
def monitoring_alerts():
    """
    GET /compare/alerts

    Aggregated Alert[] records across all completed evaluations.
    Maps to Prisma Alert schema: { id, message, severity, resolved, evaluationId }

    Response:
    {
        alerts: [
            { id, message, severity, resolved, eval_id }, ...
        ],
        summary: { critical: int, warning: int, info: int, resolved: int, total: int }
    }
    """
    all_alerts = []
    for eid, ev in evaluations.items():
        if ev.get("status") != "complete":
            continue
        all_alerts.extend(_build_alerts(ev, eid))

    # Sort: unresolved critical first, then warning, then info, then resolved
    severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    all_alerts.sort(key=lambda a: (
        1 if a["resolved"] else 0,
        severity_order.get(a["severity"], 3)
    ))

    summary = {
        "critical": sum(1 for a in all_alerts if a["severity"] == "CRITICAL" and not a["resolved"]),
        "warning":  sum(1 for a in all_alerts if a["severity"] == "WARNING"  and not a["resolved"]),
        "info":     sum(1 for a in all_alerts if a["severity"] == "INFO"),
        "resolved": sum(1 for a in all_alerts if a["resolved"]),
        "total":    len(all_alerts),
    }

    return jsonify({"alerts": all_alerts, "summary": summary}), 200