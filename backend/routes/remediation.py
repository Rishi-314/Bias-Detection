"""
routes/remediation.py
Bias Remediation Toolkit API + CI/CD deployment gate.

Endpoints:
  GET  /remediation/<eval_id>        → full remediation plan
  GET  /deploy/check/<eval_id>       → deployment gate (pass/block)
  GET  /deploy/regression/<eval_id>  → regression vs previous eval
"""

from flask import Blueprint, jsonify, request
from routes.evaluation import evaluations

remediation_routes = Blueprint("remediation_routes", __name__)


@remediation_routes.route("/remediation/<eval_id>", methods=["GET"])
def get_remediation(eval_id: str):
    """
    GET /remediation/<eval_id>

    Full remediation plan with:
    - Prioritized fix list
    - Before/after score estimates
    - Code snippets
    - Counterfactual examples
    - Quick wins vs high impact fixes
    """
    ev = evaluations.get(eval_id)
    if ev is None:
        return jsonify({"error": f"Evaluation '{eval_id}' not found"}), 404

    status = ev.get("status", "queued")
    if status in ("queued", "running"):
        return jsonify({"evaluation_id": eval_id, "status": status}), 200
    if status == "error":
        return jsonify({"evaluation_id": eval_id, "status": "error", "error": ev.get("error")}), 200

    remediation = ev.get("remediation", {})
    cf_examples = ev.get("counterfactual_examples", [])
    fairness    = ev.get("fairness", {})
    privacy     = ev.get("privacy", {})
    robustness  = ev.get("robustness", {})
    transparency = ev.get("transparency", {})
    accountability = ev.get("accountability", {})

    # Aggregate all findings across dimensions
    all_findings = []
    for dim_key, dim_data in [
        ("Privacy",       privacy),
        ("Robustness",    robustness),
        ("Transparency",  transparency),
        ("Accountability", accountability),
    ]:
        if isinstance(dim_data, dict):
            for f in dim_data.get("findings", []):
                all_findings.append({**f, "dimension": dim_key})

    # Aggregate all recommendations
    all_recs = []
    for dim_key, dim_data in [
        ("Privacy",       privacy),
        ("Robustness",    robustness),
        ("Transparency",  transparency),
        ("Accountability", accountability),
    ]:
        if isinstance(dim_data, dict):
            for r in dim_data.get("recommendations", []):
                all_recs.append({**r, "dimension": dim_key})

    payload = {
        "evaluation_id":           eval_id,
        "status":                  "complete",

        # Core remediation plan
        "remediation":             remediation,

        # Counterfactual examples (concrete "what-if" pairs)
        "counterfactual_examples": cf_examples,

        # Findings from new dimensions
        "findings":                all_findings,
        "recommendations":         all_recs,

        # Score summary
        "scores": {
            "overall":        ev.get("ethical_score", 0),
            "fairness":       ev.get("fairness_score"),
            "privacy":        ev.get("privacy_score"),
            "robustness":     ev.get("robustness_score"),
            "transparency":   ev.get("transparency_score"),
            "accountability": ev.get("accountability_score"),
        },

        # Deployment status
        "deployment_blocked":  remediation.get("deployment_blocked", False),
        "deployment_message":  remediation.get("deployment_message", ""),
        "records":             ev.get("records", 0),
        "model_id":            ev.get("model_id", ""),
    }

    return jsonify(payload), 200


@remediation_routes.route("/deploy/check/<eval_id>", methods=["GET"])
def deployment_check(eval_id: str):
    """
    GET /deploy/check/<eval_id>?threshold=0.75

    CI/CD deployment gate.
    Returns PASS / BLOCK with reason.

    Query params:
        threshold (float, default 0.75) — minimum ethical score to pass
    """
    ev = evaluations.get(eval_id)
    if ev is None:
        return jsonify({"error": f"Evaluation '{eval_id}' not found"}), 404

    status = ev.get("status", "queued")
    if status != "complete":
        return jsonify({
            "evaluation_id": eval_id,
            "gate_status":   "PENDING",
            "reason":        f"Evaluation not complete (status: {status})",
        }), 200

    try:
        threshold = float(request.args.get("threshold", 0.75))
    except (TypeError, ValueError):
        threshold = 0.75

    overall_score = ev.get("ethical_score", 0.0)
    remediation   = ev.get("remediation", {})
    critical      = remediation.get("critical_issues", 0)
    failing_dims  = remediation.get("failing_dimensions", 0)

    # Gate logic
    passed = overall_score >= threshold and critical == 0

    # Build detailed check list for CI/CD output
    checks = [
        {
            "name":    "Overall Ethical Score",
            "passed":  overall_score >= threshold,
            "value":   round(overall_score * 100, 1),
            "threshold": round(threshold * 100, 1),
            "unit":    "%",
        },
        {
            "name":    "Critical Issues",
            "passed":  critical == 0,
            "value":   critical,
            "threshold": 0,
            "unit":    "issues",
        },
        {
            "name":    "Failing Dimensions",
            "passed":  failing_dims <= 1,
            "value":   failing_dims,
            "threshold": 1,
            "unit":    "dimensions",
        },
        {
            "name":    "Fairness Score",
            "passed":  (ev.get("fairness_score") or 0) >= 0.70,
            "value":   round((ev.get("fairness_score") or 0) * 100, 1),
            "threshold": 70,
            "unit":    "%",
        },
        {
            "name":    "Privacy Score",
            "passed":  (ev.get("privacy_score") or 0.5) >= 0.65,
            "value":   round((ev.get("privacy_score") or 0.5) * 100, 1),
            "threshold": 65,
            "unit":    "%",
        },
    ]

    checks_passed = sum(1 for c in checks if c["passed"])
    checks_failed = len(checks) - checks_passed

    return jsonify({
        "evaluation_id":  eval_id,
        "gate_status":    "PASS" if passed else "BLOCK",
        "gate_icon":      "✅" if passed else "🔴",
        "passed":         passed,
        "overall_score":  round(overall_score * 100, 1),
        "threshold_used": round(threshold * 100, 1),
        "critical_issues": critical,
        "failing_dimensions": failing_dims,
        "checks":         checks,
        "checks_passed":  checks_passed,
        "checks_failed":  checks_failed,
        "reason": (
            "All ethical thresholds met — approved for deployment."
            if passed else
            f"Deployment blocked: {checks_failed} check(s) failed. "
            f"Score: {round(overall_score*100,1)}% < threshold {round(threshold*100,1)}%."
            + (f" {critical} critical issue(s) must be resolved." if critical > 0 else "")
        ),
        "remediation_url": f"/remediation/{eval_id}",
        "timestamp": __import__('datetime').datetime.utcnow().isoformat(),
    }), 200


@remediation_routes.route("/deploy/regression/<eval_id>", methods=["GET"])
def regression_check(eval_id: str):
    """
    GET /deploy/regression/<eval_id>?baseline=<baseline_eval_id>

    Compare new evaluation to a baseline.
    Returns dimension-by-dimension regression analysis.
    If no baseline provided, uses the previous completed evaluation.
    """
    ev = evaluations.get(eval_id)
    if ev is None:
        return jsonify({"error": f"Evaluation '{eval_id}' not found"}), 404

    baseline_id = request.args.get("baseline")

    # Find baseline: explicit param or previous completed eval
    baseline_ev = None
    if baseline_id:
        baseline_ev = evaluations.get(baseline_id)
    else:
        # Find the most recent completed eval that isn't this one
        completed = [
            (eid, e) for eid, e in evaluations.items()
            if e.get("status") == "complete" and eid != eval_id
        ]
        if completed:
            baseline_id, baseline_ev = completed[-1]

    if baseline_ev is None:
        return jsonify({
            "evaluation_id": eval_id,
            "baseline_id":   baseline_id,
            "has_baseline":  False,
            "message":       "No baseline evaluation found. Run a second evaluation to enable regression testing.",
        }), 200

    # Compare scores
    score_keys = [
        ("ethical_score",        "Overall Ethics"),
        ("fairness_score",       "Fairness"),
        ("privacy_score",        "Privacy"),
        ("robustness_score",     "Robustness"),
        ("transparency_score",   "Transparency"),
        ("accountability_score", "Accountability"),
    ]

    regressions = []
    improvements = []
    neutral = []

    for key, label in score_keys:
        new_val  = ev.get(key) or 0
        base_val = baseline_ev.get(key) or 0
        delta    = round(new_val - base_val, 4)
        delta_pct = round(delta * 100, 2)

        row = {
            "dimension":   label,
            "key":         key,
            "baseline":    round(base_val, 4),
            "current":     round(new_val, 4),
            "delta":       delta,
            "delta_pct":   delta_pct,
            "direction":   "improvement" if delta > 0.005 else "regression" if delta < -0.005 else "stable",
            "delta_label": f"+{delta_pct}% ↑" if delta > 0.005 else f"{delta_pct}% ↓" if delta < -0.005 else "— stable",
        }

        if delta < -0.005:
            regressions.append(row)
        elif delta > 0.005:
            improvements.append(row)
        else:
            neutral.append(row)

    has_regression = len(regressions) > 0
    overall_delta  = round(
        (ev.get("ethical_score") or 0) - (baseline_ev.get("ethical_score") or 0),
        4
    )

    return jsonify({
        "evaluation_id": eval_id,
        "baseline_id":   baseline_id,
        "has_baseline":  True,
        "has_regression": has_regression,
        "overall_delta":  overall_delta,
        "overall_direction": "improvement" if overall_delta > 0.005 else "regression" if overall_delta < -0.005 else "stable",
        "regressions":    regressions,
        "improvements":   improvements,
        "neutral":        neutral,
        "all_dimensions": regressions + improvements + neutral,
        "verdict": (
            f"⚠️ REGRESSION DETECTED in {len(regressions)} dimension(s). "
            f"Overall score changed by {round(overall_delta*100,2)}%."
        ) if has_regression else (
            f"✅ No regression detected. "
            f"Overall score {'improved' if overall_delta > 0 else 'stable'} by {abs(round(overall_delta*100,2))}%."
        ),
        "block_deployment": has_regression and overall_delta < -0.05,
    }), 200
    
@remediation_routes.route("/remediation/<eval_id>/simulate-fix", methods=["POST"])
def simulate_fix(eval_id: str):
    """
    POST /remediation/<eval_id>/simulate-fix
    Body: { "fix_id": "disparateImpact_0" }

    Simulates applying a specific fix and returns before/after metrics.
    """
    ev = evaluations.get(eval_id)
    if ev is None:
        return jsonify({"error": f"Evaluation '{eval_id}' not found"}), 404

    body   = request.get_json(silent=True) or {}
    fix_id = body.get("fix_id", "")

    fairness    = ev.get("fairness", {})
    remediation = ev.get("remediation", {})

    # Find the fix
    all_fixes = remediation.get("all_fixes", [])
    fix = next((f for f in all_fixes if f.get("id") == fix_id), None)

    if not fix:
        return jsonify({"error": f"Fix '{fix_id}' not found"}), 404

    dim_key = fix.get("dimension_key")

    # Simulate the improvement (conservative: 60% of gap closed)
    current_score   = fix.get("current_score", 0)
    estimated_after = fix.get("estimated_score_after", min(current_score + 0.15, 1.0))
    improvement     = estimated_after - current_score

    # Build before snapshot for ALL dimensions
    before = {
        "ethical_score":      ev.get("ethical_score", 0),
        "fairness_score":     ev.get("fairness_score", 0),
        "disparateImpact":    fairness.get("disparateImpact", 0),
        "demographicParity":  fairness.get("demographicParity", 0),
        "counterfactual":     fairness.get("counterfactual", 0),
        "intersectional":     fairness.get("intersectional", 0),
        "individualFairness": fairness.get("individualFairness", 0),
        "groupFairness":      fairness.get("groupFairness", 0),
    }

    # Start after as a copy of before, then apply the fix
    after = dict(before)

    # Only apply if dim_key is a known key in our metrics
    KNOWN_KEYS = {
        "disparateImpact", "demographicParity", "counterfactual",
        "intersectional", "individualFairness", "groupFairness",
        "calibrationError",
    }

    if dim_key and dim_key in KNOWN_KEYS:
        after[dim_key] = round(min(estimated_after, 1.0), 4)

        # Spillover: related dimensions improve ~30% as much
        spillover_map = {
            "disparateImpact":   ["demographicParity", "groupFairness"],
            "demographicParity": ["disparateImpact", "groupFairness"],
            "counterfactual":    ["individualFairness"],
            "intersectional":    ["groupFairness"],
            "groupFairness":     ["disparateImpact"],
        }
        for related in spillover_map.get(dim_key, []):
            if related in after and after[related] < 1.0:
                after[related] = round(min(after[related] + improvement * 0.3, 1.0), 4)
    elif dim_key in ("privacy_score", "robustness_score", "transparency_score", "accountability_score"):
        # Non-fairness fix — small ethical score bump only, no fairness dimension changes
        pass  # handled below via overall score recalculation

    # Recompute fairness_score from updated after values
    # NOTE: demographicParity and calibrationError are "lower is better",
    # so we invert them when computing the aggregate.
    cal_error = fairness.get("calibrationError", 0)  # unchanged by most fixes
    fairness_components = [
        after.get("individualFairness", 0),
        after.get("groupFairness", 0),
        max(0, 1 - after.get("demographicParity", 0)),   # invert: lower gap = better
        after.get("disparateImpact", 0),
        max(0, 1 - cal_error),                            # invert: lower error = better
        after.get("counterfactual", 0),
        after.get("intersectional", 0),
    ]
    new_fairness_score = round(sum(fairness_components) / len(fairness_components), 4)
    after["fairness_score"] = new_fairness_score

    # Determine which dimension sub-scores changed for non-fairness fixes
    privacy_score_after        = ev.get("privacy_score", 0.5) or 0.5
    robustness_score_after     = ev.get("robustness_score", 0.5) or 0.5
    transparency_score_after   = ev.get("transparency_score", 0.5) or 0.5
    accountability_score_after = ev.get("accountability_score", 0.5) or 0.5

    if dim_key == "privacy_score":
        privacy_score_after = round(min(estimated_after, 1.0), 4)
    elif dim_key == "robustness_score":
        robustness_score_after = round(min(estimated_after, 1.0), 4)
    elif dim_key == "transparency_score":
        transparency_score_after = round(min(estimated_after, 1.0), 4)
    elif dim_key == "accountability_score":
        accountability_score_after = round(min(estimated_after, 1.0), 4)

    # Overall ethical score = fairness 40% + other 60%
    after_ethical = round(
        new_fairness_score       * 0.40 +
        privacy_score_after      * 0.20 +
        robustness_score_after   * 0.15 +
        transparency_score_after * 0.15 +
        accountability_score_after * 0.10,
        4
    )
    after["ethical_score"] = after_ethical

    return jsonify({
        "fix_id":    fix_id,
        "fix_title": fix.get("title"),
        "dimension": fix.get("dimension"),
        "effort":    fix.get("effort"),
        "before":    before,
        "after":     after,
        "improvement": {
            "ethical_score_delta": round(after_ethical - before["ethical_score"], 4),
            "dimension_delta":     round(improvement, 4),
            "verdict": (
                "✅ Applying this fix would bring the model above the deployment threshold."
                if after_ethical >= 0.75 and before["ethical_score"] < 0.75
                else f"📈 Ethical score would improve from {round(before['ethical_score']*100,1)}% to {round(after_ethical*100,1)}%."
                if round(after_ethical - before['ethical_score'], 4) > 0
                else "ℹ️ This fix improves a sub-dimension but has minimal impact on the overall fairness score."
            ),
        },
        "accuracy_impact": "Typically < 1-2% accuracy loss for post-processing fixes.",
        "disclaimer":      "These are conservative estimates based on published research benchmarks. Actual improvement requires retraining or post-processing.",
    }), 200