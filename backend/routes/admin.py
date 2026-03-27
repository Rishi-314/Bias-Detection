"""
routes/admin.py
All endpoints for page10.html — Admin Dashboard.

Blueprint: admin_routes
Prefix:    /admin

Endpoints:
  GET  /admin/health                  → system health cards
  GET  /admin/alerts                  → active alerts list + summary badges
  POST /admin/alerts/<id>/acknowledge → mark alert acknowledged
  POST /admin/alerts/<id>/resolve     → remove alert
  GET  /admin/stats                   → usage chart + active-user breakdown
  GET  /admin/models                  → model management table
  POST /admin/models/<model_id>/retrain   → trigger retrain
  POST /admin/models/<model_id>/rollback  → rollback
  POST /admin/models/<model_id>/archive   → archive
  GET  /admin/users                   → user list (with search/role/tier filter)
  POST /admin/users/invite            → invite new user
  PUT  /admin/users/<user_id>         → edit user
  POST /admin/users/<user_id>/reset-key → regenerate API key
  GET  /admin/settings                → current system settings
  POST /admin/settings                → save system settings
  POST /admin/settings/reset          → reset to defaults
"""

import time
import uuid
import hashlib
from datetime import datetime, timedelta
from flask import Blueprint, jsonify, request
from routes.evaluation import evaluations   # shared in-memory store

admin_routes = Blueprint("admin_routes", __name__, url_prefix="/admin")

# ── In-memory stores (replace with DB in production) ─────────────────────────

# Alerts store: id → alert dict
_alerts: dict = {
    "alert-di-001": {
        "id": "alert-di-001", "severity": "CRITICAL",
        "message": "Model v2.0.0 — Disparate Impact Below Threshold",
        "detail":  "Impact ratio dropped to 0.74 (threshold: 0.80)",
        "acknowledged": False, "resolved": False,
        "created_at": (datetime.utcnow() - timedelta(minutes=15)).isoformat(),
    },
    "alert-rate-002": {
        "id": "alert-rate-002", "severity": "CRITICAL",
        "message": "API Rate Limit Exceeded — Enterprise Customer",
        "detail":  "Customer 'acme-corp' exceeded 1000 req/hr limit",
        "acknowledged": False, "resolved": False,
        "created_at": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
    },
    "alert-perf-003": {
        "id": "alert-perf-003", "severity": "WARNING",
        "message": "High Processing Time (>5s) Detected",
        "detail":  "Average eval time: 6.2s over last 30 evals",
        "acknowledged": False, "resolved": False,
        "created_at": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
    },
}

# Model registry: model_id → model dict
_models: dict = {
    "v2.0.0": {
        "model_id": "v2.0.0", "version": "v2.0.0",
        "deployed_at": "2024-03-20", "ethical_score": 0.87,
        "status": "ACTIVE", "records_trained": 20576,
    },
    "v1.5.0": {
        "model_id": "v1.5.0", "version": "v1.5.0",
        "deployed_at": "2024-02-15", "ethical_score": 0.84,
        "status": "ACTIVE", "records_trained": 15234,
    },
    "v1.0.0": {
        "model_id": "v1.0.0", "version": "v1.0.0",
        "deployed_at": "2024-01-15", "ethical_score": 0.82,
        "status": "INACTIVE", "records_trained": 12100,
    },
}

# User registry: user_id → user dict
_users: dict = {
    "user-001": {
        "user_id": "user-001", "name": "John Doe",
        "email": "john@company.com", "role": "ADMIN",
        "tier": "ENTERPRISE", "api_key_hint": "••••••••1234",
        "created_at": "2024-01-01",
    },
    "user-002": {
        "user_id": "user-002", "name": "Jane Smith",
        "email": "jane@startup.com", "role": "DEVELOPER",
        "tier": "PRO", "api_key_hint": "••••••••5678",
        "created_at": "2024-01-15",
    },
}

# System settings (maps to Prisma SystemSetting key/value pairs)
_DEFAULT_SETTINGS = {
    "min_ethical_score":        0.75,
    "individual_fairness_min":  0.80,
    "disparate_impact_threshold": 0.80,
    "group_fairness_threshold": 0.80,
    "calibration_error_max":    0.05,
    "alert_on_low_fairness":    True,
    "email_notifications":      True,
    "slack_integration":        False,
}
_settings: dict = dict(_DEFAULT_SETTINGS)

# Simple usage counters (replace with time-series DB query in production)
_usage_7days  = [65, 72, 80, 85, 92, 88, 95]   # eval counts per day
_cert_issued  = 567
_evals_today  = 1234


# ── Helpers ───────────────────────────────────────────────────────────────────

def _relative_time(iso: str) -> str:
    try:
        dt    = datetime.fromisoformat(iso)
        delta = datetime.utcnow() - dt
        secs  = int(delta.total_seconds())
        if secs < 60:   return f"{secs}s ago"
        if secs < 3600: return f"{secs // 60}m ago"
        if secs < 86400:return f"{secs // 3600}h ago"
        return f"{secs // 86400}d ago"
    except Exception:
        return iso


def _alert_icon(severity: str, acknowledged: bool) -> str:
    if acknowledged:        return "✓"
    if severity == "CRITICAL": return "⚠️"
    return "⚡"


def _alert_color(severity: str, acknowledged: bool) -> str:
    if acknowledged:           return "green"
    if severity == "CRITICAL": return "red"
    return "yellow"


# ── System Health ─────────────────────────────────────────────────────────────

@admin_routes.route("/health", methods=["GET"])
def system_health():
    """
    GET /admin/health

    Response:
    {
        cards: [
            { id, label, value, status, status_label, dot_color }, ...
        ],
        timestamp: str
    }
    """
    queue_size = sum(
        1 for ev in evaluations.values() if ev.get("status") in ("queued", "running")
    )
    active_models = sum(1 for m in _models.values() if m["status"] == "ACTIVE")

    cards = [
        {
            "id":           "uptime",
            "label":        "API Uptime",
            "value":        "99.9%",
            "status":       "Operational",
            "dot_color":    "green",
        },
        {
            "id":           "queue",
            "label":        "Queue Size",
            "value":        str(queue_size),
            "status":       "No backlog" if queue_size == 0 else f"{queue_size} pending",
            "dot_color":    "green" if queue_size == 0 else "yellow",
        },
        {
            "id":           "avg_time",
            "label":        "Avg Processing Time",
            "value":        f"{round(2.3 + queue_size * 0.1, 1)}s",
            "status":       "Within SLA",
            "dot_color":    "green",
        },
        {
            "id":           "models",
            "label":        "Models Deployed",
            "value":        str(active_models),
            "status":       "All active" if active_models == len(_models) else f"{active_models} of {len(_models)} active",
            "dot_color":    "green" if active_models > 0 else "red",
        },
    ]
    return jsonify({"cards": cards, "timestamp": datetime.utcnow().isoformat()}), 200


# ── Alerts ────────────────────────────────────────────────────────────────────

@admin_routes.route("/alerts", methods=["GET"])
def get_alerts():
    """
    GET /admin/alerts

    Query params:
      severity=CRITICAL|WARNING   (optional filter)
      resolved=true|false         (default: false — hides resolved)

    Response:
    {
        alerts: [ { id, severity, message, detail, acknowledged, resolved,
                    created_at, relative_time, icon, color }, ... ],
        summary: { critical, warning, resolved, total }
    }
    """
    severity_filter = request.args.get("severity", "").upper()
    show_resolved   = request.args.get("resolved", "false").lower() == "true"

    # Also pull fairness-derived alerts from completed evaluations
    live_alerts = dict(_alerts)
    for eid, ev in evaluations.items():
        if ev.get("status") != "complete":
            continue
        fairness = ev.get("fairness", {})
        di = fairness.get("disparateImpact", 1.0)
        if di < _settings["disparate_impact_threshold"]:
            aid = f"eval-di-{eid[:8]}"
            if aid not in live_alerts:
                live_alerts[aid] = {
                    "id": aid, "severity": "CRITICAL",
                    "message": f"Eval {eid[:8]}… — Disparate impact {round(di, 4)} below threshold",
                    "detail":  f"Threshold: {_settings['disparate_impact_threshold']}",
                    "acknowledged": False, "resolved": False,
                    "created_at": datetime.utcnow().isoformat(),
                }
        score = ev.get("ethical_score", 1.0)
        if score < _settings["min_ethical_score"]:
            aid = f"eval-score-{eid[:8]}"
            if aid not in live_alerts:
                live_alerts[aid] = {
                    "id": aid, "severity": "WARNING",
                    "message": f"Eval {eid[:8]}… — Ethical score {round(score, 4)} below minimum",
                    "detail":  f"Minimum: {_settings['min_ethical_score']}",
                    "acknowledged": False, "resolved": False,
                    "created_at": datetime.utcnow().isoformat(),
                }

    result = []
    for a in live_alerts.values():
        if not show_resolved and a["resolved"]:
            continue
        if severity_filter and a["severity"] != severity_filter:
            continue
        result.append({
            **a,
            "relative_time": _relative_time(a["created_at"]),
            "icon":          _alert_icon(a["severity"], a["acknowledged"]),
            "color":         _alert_color(a["severity"], a["acknowledged"]),
        })

    result.sort(key=lambda x: (
        1 if x["resolved"] else 0,
        0 if x["severity"] == "CRITICAL" else 1,
        x["acknowledged"]
    ))

    summary = {
        "critical": sum(1 for a in live_alerts.values() if a["severity"] == "CRITICAL" and not a["resolved"]),
        "warning":  sum(1 for a in live_alerts.values() if a["severity"] == "WARNING"  and not a["resolved"]),
        "resolved": sum(1 for a in live_alerts.values() if a["resolved"]),
        "total":    len(live_alerts),
    }

    return jsonify({"alerts": result, "summary": summary}), 200


@admin_routes.route("/alerts/<alert_id>/acknowledge", methods=["POST"])
def acknowledge_alert(alert_id: str):
    """POST /admin/alerts/<id>/acknowledge"""
    if alert_id not in _alerts:
        return jsonify({"error": "Alert not found"}), 404
    _alerts[alert_id]["acknowledged"] = True
    return jsonify({"success": True, "alert_id": alert_id}), 200


@admin_routes.route("/alerts/<alert_id>/resolve", methods=["POST"])
def resolve_alert(alert_id: str):
    """POST /admin/alerts/<id>/resolve"""
    if alert_id not in _alerts:
        return jsonify({"error": "Alert not found"}), 404
    _alerts[alert_id]["resolved"]     = True
    _alerts[alert_id]["acknowledged"] = True
    return jsonify({"success": True, "alert_id": alert_id}), 200


# ── Usage Statistics ──────────────────────────────────────────────────────────

@admin_routes.route("/stats", methods=["GET"])
def usage_stats():
    """
    GET /admin/stats?range=7d|30d|24h

    Response:
    {
        chart: {
            labels: ["Mon", "Tue", ...],
            values: [65, 72, ...],      # eval counts
            max:    int
        },
        counters: {
            evaluations_today: int,
            certificates_issued: int,
            total_evaluations: int,
        },
        users: {
            total: int,
            by_tier: { FREE: int, PRO: int, ENTERPRISE: int }
        }
    }
    """
    range_param = request.args.get("range", "7d")

    if range_param == "24h":
        labels = [f"{h:02d}:00" for h in range(0, 24, 3)]
        values = [12, 8, 4, 6, 14, 22, 35, 41]
    elif range_param == "30d":
        labels = [f"Day {i+1}" for i in range(30)]
        values = [
            55, 60, 58, 72, 80, 77, 85, 90, 88, 92,
            70, 75, 80, 85, 82, 88, 91, 95, 89, 93,
            87, 90, 94, 96, 88, 91, 95, 97, 99, 102,
        ]
    else:  # 7d default
        today = datetime.utcnow()
        labels = [(today - timedelta(days=6 - i)).strftime("%a") for i in range(7)]
        values = list(_usage_7days)

    by_tier = {"FREE": 0, "PRO": 0, "ENTERPRISE": 0}
    for u in _users.values():
        tier = u.get("tier", "FREE")
        by_tier[tier] = by_tier.get(tier, 0) + 1

    return jsonify({
        "chart": {
            "labels": labels,
            "values": values,
            "max":    max(values) if values else 100,
        },
        "counters": {
            "evaluations_today":  _evals_today + len(evaluations),
            "certificates_issued": _cert_issued,
            "total_evaluations":   _evals_today + len(evaluations),
        },
        "users": {
            "total":   len(_users),
            "by_tier": by_tier,
        },
    }), 200


# ── Model Management ──────────────────────────────────────────────────────────

@admin_routes.route("/models", methods=["GET"])
def list_models():
    """
    GET /admin/models

    Response:
    {
        models: [
            { model_id, version, deployed_at, ethical_score, status,
              records_trained, actions: [str] }
        ]
    }
    """
    # Enrich from completed evaluations
    for eid, ev in evaluations.items():
        mid = ev.get("model_id")
        if mid and mid not in _models:
            _models[mid] = {
                "model_id":       mid,
                "version":        mid,
                "deployed_at":    datetime.utcnow().strftime("%Y-%m-%d"),
                "ethical_score":  ev.get("ethical_score", 0.0),
                "status":         "ACTIVE",
                "records_trained": ev.get("records", 0),
            }

    result = []
    for m in sorted(_models.values(), key=lambda x: x["deployed_at"], reverse=True):
        actions = ["retrain", "rollback"] if m["status"] == "ACTIVE" else ["retrain"]
        actions.append("archive")
        result.append({**m, "actions": actions})

    return jsonify({"models": result}), 200


@admin_routes.route("/models/<model_id>/retrain", methods=["POST"])
def retrain_model(model_id: str):
    """POST /admin/models/<model_id>/retrain"""
    if model_id not in _models:
        return jsonify({"error": f"Model '{model_id}' not found"}), 404
    return jsonify({
        "success": True,
        "model_id": model_id,
        "message": f"Retraining initiated for {model_id}. A new evaluation will be queued.",
        "job_id":  str(uuid.uuid4()),
    }), 200


@admin_routes.route("/models/<model_id>/rollback", methods=["POST"])
def rollback_model(model_id: str):
    """POST /admin/models/<model_id>/rollback"""
    if model_id not in _models:
        return jsonify({"error": f"Model '{model_id}' not found"}), 404
    versions = sorted(_models.keys())
    idx = versions.index(model_id) if model_id in versions else -1
    if idx <= 0:
        return jsonify({"error": "No previous version to roll back to"}), 400
    previous = versions[idx - 1]
    _models[model_id]["status"] = "INACTIVE"
    _models[previous]["status"] = "ACTIVE"
    return jsonify({
        "success":  True,
        "rolled_back_from": model_id,
        "rolled_back_to":   previous,
        "message": f"Rolled back from {model_id} to {previous}",
    }), 200


@admin_routes.route("/models/<model_id>/archive", methods=["POST"])
def archive_model(model_id: str):
    """POST /admin/models/<model_id>/archive"""
    if model_id not in _models:
        return jsonify({"error": f"Model '{model_id}' not found"}), 404
    _models[model_id]["status"] = "ARCHIVED"
    return jsonify({"success": True, "model_id": model_id, "message": f"{model_id} archived."}), 200


# ── User Management ───────────────────────────────────────────────────────────

@admin_routes.route("/users", methods=["GET"])
def list_users():
    """
    GET /admin/users?search=&role=&tier=

    Response:
    {
        users: [
            { user_id, name, email, role, tier, api_key_hint, created_at }
        ],
        total: int
    }
    """
    search = request.args.get("search", "").lower()
    role   = request.args.get("role",   "").upper()
    tier   = request.args.get("tier",   "").upper()

    result = []
    for u in _users.values():
        if search and search not in u["name"].lower() and search not in u["email"].lower():
            continue
        if role and u["role"] != role:
            continue
        if tier and u["tier"] != tier:
            continue
        result.append(u)

    return jsonify({"users": result, "total": len(result)}), 200


@admin_routes.route("/users/invite", methods=["POST"])
def invite_user():
    """
    POST /admin/users/invite
    Body: { name, email, role, tier }

    Response: { success, user_id, message }
    """
    body  = request.get_json(silent=True) or {}
    name  = body.get("name",  "").strip()
    email = body.get("email", "").strip()
    role  = body.get("role",  "DEVELOPER").upper()
    tier  = body.get("tier",  "FREE").upper()

    if not name or not email:
        return jsonify({"error": "name and email are required"}), 400
    if any(u["email"] == email for u in _users.values()):
        return jsonify({"error": "User with that email already exists"}), 409

    uid = "user-" + str(uuid.uuid4())[:8]
    raw_key = str(uuid.uuid4()).replace("-", "")
    _users[uid] = {
        "user_id":      uid,
        "name":         name,
        "email":        email,
        "role":         role,
        "tier":         tier,
        "api_key_hint": "••••••••" + raw_key[-4:],
        "created_at":   datetime.utcnow().strftime("%Y-%m-%d"),
    }

    return jsonify({
        "success":  True,
        "user_id":  uid,
        "message":  f"Invitation sent to {email}.",
        # In production: send email with invite link; never return raw key
        "api_key":  raw_key,
    }), 201


@admin_routes.route("/users/<user_id>", methods=["PUT"])
def edit_user(user_id: str):
    """
    PUT /admin/users/<user_id>
    Body: { name?, email?, role?, tier? }
    """
    if user_id not in _users:
        return jsonify({"error": "User not found"}), 404
    body = request.get_json(silent=True) or {}
    allowed = {"name", "email", "role", "tier"}
    for key in allowed:
        if key in body:
            val = body[key]
            _users[user_id][key] = val.upper() if key in {"role", "tier"} else val
    return jsonify({"success": True, "user": _users[user_id]}), 200


@admin_routes.route("/users/<user_id>/reset-key", methods=["POST"])
def reset_api_key(user_id: str):
    """POST /admin/users/<user_id>/reset-key"""
    if user_id not in _users:
        return jsonify({"error": "User not found"}), 404
    raw_key = str(uuid.uuid4()).replace("-", "")
    _users[user_id]["api_key_hint"] = "••••••••" + raw_key[-4:]
    return jsonify({
        "success":      True,
        "user_id":      user_id,
        "api_key":      raw_key,   # In production: send via email, not response body
        "api_key_hint": _users[user_id]["api_key_hint"],
        "message":      "API key regenerated successfully.",
    }), 200


# ── System Settings ───────────────────────────────────────────────────────────

@admin_routes.route("/settings", methods=["GET"])
def get_settings():
    """
    GET /admin/settings

    Returns current SystemSetting values in two display groups:
    {
        fairness_thresholds: { ... },
        alert_rules:         { ... },
        raw:                 { ... }   # flat key/value for form binding
    }
    """
    return jsonify({
        "fairness_thresholds": {
            "min_ethical_score":           _settings["min_ethical_score"],
            "individual_fairness_min":     _settings["individual_fairness_min"],
            "disparate_impact_threshold":  _settings["disparate_impact_threshold"],
            "group_fairness_threshold":    _settings["group_fairness_threshold"],
            "calibration_error_max":       _settings["calibration_error_max"],
        },
        "alert_rules": {
            "alert_on_low_fairness": _settings["alert_on_low_fairness"],
            "email_notifications":   _settings["email_notifications"],
            "slack_integration":     _settings["slack_integration"],
        },
        "raw": dict(_settings),
    }), 200


@admin_routes.route("/settings", methods=["POST"])
def save_settings():
    """
    POST /admin/settings
    Body: { min_ethical_score?, individual_fairness_min?, disparate_impact_threshold?,
            group_fairness_threshold?, calibration_error_max?,
            alert_on_low_fairness?, email_notifications?, slack_integration? }

    Response: { success, settings: { ... } }
    """
    body = request.get_json(silent=True) or {}

    numeric_keys = {
        "min_ethical_score", "individual_fairness_min",
        "disparate_impact_threshold", "group_fairness_threshold", "calibration_error_max",
    }
    bool_keys = {"alert_on_low_fairness", "email_notifications", "slack_integration"}

    errors = []
    for key, val in body.items():
        if key not in _settings:
            continue
        if key in numeric_keys:
            try:
                v = float(val)
                if not (0.0 <= v <= 1.0):
                    errors.append(f"{key} must be between 0 and 1")
                else:
                    _settings[key] = round(v, 4)
            except (TypeError, ValueError):
                errors.append(f"{key} must be a number")
        elif key in bool_keys:
            _settings[key] = bool(val)

    if errors:
        return jsonify({"error": "Validation failed", "details": errors}), 400

    return jsonify({"success": True, "settings": dict(_settings)}), 200


@admin_routes.route("/settings/reset", methods=["POST"])
def reset_settings():
    """POST /admin/settings/reset — restore all settings to defaults"""
    _settings.update(_DEFAULT_SETTINGS)
    return jsonify({"success": True, "settings": dict(_settings)}), 200