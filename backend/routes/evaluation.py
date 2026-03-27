from flask import Blueprint, request, jsonify
import pandas as pd
import os
import uuid
import threading

from services.fairness import compute_fairness
from services.shap_explainer import compute_shap
from services.model_loader import get_model, preprocess_for_inference
from services.privacy_scorer import compute_privacy_score
from services.robustness_scorer import compute_robustness_score
from services.accountability_scorer import compute_accountability_score, compute_transparency_score
from services.remediation import generate_remediation_plan, generate_counterfactual_examples

evaluation_routes = Blueprint("evaluation_routes", __name__)

UPLOAD_FOLDER = "uploads"

# In-memory evaluation store
evaluations: dict = {}

VALID_REPORT_TYPES = {"developer", "regulator", "enduser", "all", "executive"}
REPORT_TYPE_MAP    = {
    "developer": "DEVELOPER",
    "regulator": "REGULATOR",
    "enduser":   "ENDUSER",
    "all":       "ALL",
    "executive": "EXECUTIVE",
}


def _find_file(uid: str) -> str:
    matches = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith(uid)]
    if not matches:
        raise FileNotFoundError(f"No uploaded file found for id: {uid}")
    return os.path.join(UPLOAD_FOLDER, matches[0])


def _read_file(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    elif ext == ".json":
        return pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _compute_overall_ethics_score(
    fairness_score: float,
    privacy_score: float,
    robustness_score: float,
    transparency_score: float,
    accountability_score: float,
) -> float:
    """
    Weighted aggregate of all 5 ethical dimensions.
    Fairness is weighted most heavily as core requirement.
    """
    weights = {
        "fairness":       0.40,
        "privacy":        0.20,
        "robustness":     0.15,
        "transparency":   0.15,
        "accountability": 0.10,
    }
    scores = {
        "fairness":       fairness_score       or 0.0,
        "privacy":        privacy_score        or 0.5,
        "robustness":     robustness_score     or 0.5,
        "transparency":   transparency_score   or 0.5,
        "accountability": accountability_score or 0.5,
    }
    total = sum(weights[k] * scores[k] for k in weights)
    return round(max(0.0, min(1.0, total)), 4)


def _run_evaluation(
    eval_id: str,
    dataset_path: str,
    target_variable: str,
    prediction_variable: str,
    sensitive_attributes: list,
    fairness_weights: dict,
    report_type: str,
    model_id: str,
):
    try:
        # ── Step 1: Data validation ───────────────────────────────────────────
        evaluations[eval_id].update({"status": "running", "current_step": 1})
        df = _read_file(dataset_path)
        records = len(df)
        columns = list(df.columns)
        evaluations[eval_id]["records"] = records
        evaluations[eval_id]["columns"] = columns

        # ── Step 2: EDA ───────────────────────────────────────────────────────
        evaluations[eval_id]["current_step"] = 2
        target_col = target_variable if target_variable in columns else None
        if target_col is None:
            for col in columns:
                if df[col].nunique() <= 5 and df[col].dtype != object:
                    target_col = col
                    break
        if target_col is None:
            raise ValueError(f"Could not determine target variable. Available columns: {columns}")
        evaluations[eval_id]["resolved_target"] = target_col

        # ── Step 3: Model inference / prediction resolution ───────────────────
        evaluations[eval_id]["current_step"] = 3
        try:
            model, _, _, _ = get_model()
            feature_df = df.drop(columns=[target_col])
            X_input    = preprocess_for_inference(feature_df)
            y_prob_arr = model.predict_proba(X_input)[:, 1]
            df["_prediction_score"] = y_prob_arr
            df["_prediction_label"] = (df["_prediction_score"] >= 0.5).astype(int)
            pred_col = "_prediction_score"
            evaluations[eval_id]["resolved_prediction"] = pred_col
            evaluations[eval_id]["prediction_source"]   = "pretrained_model"
        except Exception as model_err:
            import traceback
            print(f"[model_loader fallback] {model_err}\n{traceback.format_exc()}")
            pred_col = None
            if prediction_variable and prediction_variable in columns:
                pred_col = prediction_variable
            else:
                for candidate in ["prediction", "prediction_score", "score", "label", "output", "pred"]:
                    if candidate in columns:
                        pred_col = candidate
                        break
            if pred_col is None:
                raise ValueError(
                    f"Pre-trained model failed AND no prediction column found. "
                    f"Model error: {model_err}. Available columns: {columns}"
                )
            evaluations[eval_id]["resolved_prediction"] = pred_col
            evaluations[eval_id]["prediction_source"]   = "dataset_column"

        # Resolve sensitive attributes
        valid_attrs = [a for a in (sensitive_attributes or []) if a in columns]
        if not valid_attrs:
            for col in columns:
                if col not in (target_col, pred_col) and df[col].dtype == object:
                    valid_attrs = [col]
                    break
        evaluations[eval_id]["resolved_attrs"] = valid_attrs

        # ── Step 4: Fairness analysis ─────────────────────────────────────────
        # Normalize sensitive attributes (CRITICAL)

        if "gender" not in df.columns:
            if "is_male" in df.columns:
                df["gender"] = df["is_male"].apply(lambda x: "Male" if x == 1 else "Female")

        if "caste_proxy" not in df.columns:
            def map_caste(row):
                if row.get("is_sc", 0): return "SC"
                if row.get("is_st", 0): return "ST"
                if row.get("is_obc", 0): return "OBC"
                return "General"
            df["caste_proxy"] = df.apply(map_caste, axis=1)

        # Force correct attributes
        valid_attrs = ["gender", "caste_proxy"]

        evaluations[eval_id]["current_step"] = 4
        preds_df = df[["_prediction_label"]].rename(columns={"_prediction_label": "prediction"})
        fairness_results = compute_fairness(
            df.drop(columns=[pred_col]),
            preds_df,
            target_col=target_col,
            sensitive_attrs=valid_attrs,
        )
        
        if fairness_results["counterfactual"] < 0.95:
            fairness_results["warning"] = "Counterfactual bias detected despite high aggregate fairness"

        # ── Step 5: Model performance metrics ────────────────────────────────
        evaluations[eval_id]["current_step"] = 5
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        import numpy as np

        y_true = df[target_col].astype(int)
        raw    = df[pred_col]
        n      = min(len(y_true), len(raw))
        y_true = y_true.iloc[:n].reset_index(drop=True)
        y_prob = raw.iloc[:n].reset_index(drop=True)
        y_pred = (y_prob >= 0.5).astype(int)

        model_metrics = {}
        try:
            model_metrics["accuracy"] = round(float(accuracy_score(y_true, y_pred)), 4)
        except Exception:
            pass
        try:
            model_metrics["f1_score"] = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
        except Exception:
            pass
        try:
            model_metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        except Exception:
            pass
        evaluations[eval_id]["model_metrics"] = model_metrics

        # ── Step 6: SHAP explanation ──────────────────────────────────────────
        evaluations[eval_id]["current_step"] = 6
        shap_results = compute_shap(
            df.drop(columns=[pred_col], errors="ignore"),
            target_col=target_col
        )

        # ── Step 6b: New dimensions (privacy, robustness, accountability, transparency) ──
        evaluations[eval_id]["current_step"] = 6

        # Privacy
        try:
            privacy_results = compute_privacy_score(
                df.drop(columns=[pred_col], errors="ignore"),
                preds_df,
                target_col,
                valid_attrs,
            )
        except Exception as e:
            print(f"[privacy_scorer] Error: {e}")
            privacy_results = {"privacy_score": 0.5, "findings": [], "recommendations": []}

        # Robustness
        try:
            robustness_results = compute_robustness_score(
                df.drop(columns=[pred_col], errors="ignore"),
                preds_df,
                target_col,
                valid_attrs,
            )
        except Exception as e:
            print(f"[robustness_scorer] Error: {e}")
            robustness_results = {"robustness_score": 0.5, "findings": [], "recommendations": []}

        # Accountability
        eval_metadata = {
            "records":     records,
            "report_type": REPORT_TYPE_MAP.get(report_type.lower(), "DEVELOPER"),
        }
        try:
            accountability_results = compute_accountability_score(
                df.drop(columns=[pred_col], errors="ignore"),
                valid_attrs,
                target_col,
                fairness_results,
                shap_results,
                eval_metadata,
            )
        except Exception as e:
            print(f"[accountability_scorer] Error: {e}")
            accountability_results = {"accountability_score": 0.5, "checks": [], "findings": [], "recommendations": []}

        # Transparency
        try:
            transparency_results = compute_transparency_score(
                shap_results,
                fairness_results,
                df.drop(columns=[pred_col], errors="ignore"),
                target_col,
            )
        except Exception as e:
            print(f"[transparency_scorer] Error: {e}")
            transparency_results = {"transparency_score": 0.5, "findings": [], "recommendations": []}

        # Counterfactual examples
        try:
            cf_examples = generate_counterfactual_examples(
                df.drop(columns=[pred_col], errors="ignore"),
                preds_df,
                target_col,
                valid_attrs,
                n_examples=5,
            )
        except Exception as e:
            print(f"[counterfactual_examples] Error: {e}")
            cf_examples = []

        # ── Step 6c: LIME local explanations ──────────────────────────────
        try:
            from services.lime_explainer import compute_lime
            lime_results = compute_lime(
                df.drop(columns=[pred_col], errors="ignore"),
                target_col=target_col,
                n_samples=5,
            )
        except Exception as e:
            print(f"[lime_explainer] Error: {e}")
            lime_results = {"method": "unavailable", "instances": [], "global_summary": "LIME not available."}
        
        # ── Step 7: Report generation ─────────────────────────────────────────
        evaluations[eval_id]["current_step"] = 7
        report_type_enum = REPORT_TYPE_MAP.get(report_type.lower(), "DEVELOPER")

        # Overall ethical score (new multi-dimensional)
        fairness_score_raw    = fairness_results["fairness_score"]
        privacy_score_val     = privacy_results.get("privacy_score", 0.5)
        robustness_score_val  = robustness_results.get("robustness_score", 0.5)
        transparency_score_val = transparency_results.get("transparency_score", 0.5)
        accountability_score_val = accountability_results.get("accountability_score", 0.5)

        overall_ethics_score = _compute_overall_ethics_score(
            fairness_score_raw,
            privacy_score_val,
            robustness_score_val,
            transparency_score_val,
            accountability_score_val,
        )

        # Remediation plan
        try:
            remediation_plan = generate_remediation_plan(
                fairness_results,
                privacy_results,
                robustness_results,
                transparency_results,
                accountability_results,
                overall_ethics_score,
                valid_attrs,
                shap_results,
            )
        except Exception as e:
            print(f"[remediation] Error: {e}")
            remediation_plan = {}

        sensitive_attr_records  = [{"name": a} for a in valid_attrs]
        fairness_weight_records = [{"dimension": k, "weight": v} for k, v in (fairness_weights or {}).items()]

        evaluations[eval_id].update({
            "status":       "complete",
            "current_step": 7,

            # Scores
            "ethical_score":          overall_ethics_score,    # NEW: multi-dimensional
            "fairness_score":         fairness_score_raw,       # original fairness-only score
            "privacy_score":          privacy_score_val,
            "robustness_score":       robustness_score_val,
            "transparency_score":     transparency_score_val,
            "accountability_score":   accountability_score_val,

            "report_type":            report_type_enum,
            "report_type_original":   report_type,
            "model_id":               model_id,

            # Fairness
            "fairness": {
                "individualFairness": fairness_results["individualFairness"],
                "groupFairness":      fairness_results["groupFairness"],
                "demographicParity":  fairness_results["demographicParity"],
                "disparateImpact":    fairness_results["disparateImpact"],
                "calibrationError":   fairness_results["calibrationError"],
                "counterfactual":     fairness_results["counterfactual"],
                "intersectional":     fairness_results["intersectional"],
                "fairness_score":     fairness_score_raw,
                "per_attribute":      fairness_results.get("per_attribute", {}),
                "records_evaluated":  fairness_results["records_evaluated"],
                "target_column":      fairness_results["target_column"],
                "prediction_column":  pred_col,
            },

            # New dimensions
            "privacy":        privacy_results,
            "robustness":     robustness_results,
            "transparency":   transparency_results,
            "accountability": accountability_results,

            # SHAP
            "shap": {
                "topFeature":         shap_results["topFeature"],
                "shapMax":            shap_results["shapMax"],
                "shapMin":            shap_results["shapMin"],
                "featureStability":   shap_results["featureStability"],
                "feature_importance": shap_results.get("feature_importance", {}),
            },
            
            "lime": lime_results,

            # Counterfactual examples
            "counterfactual_examples": cf_examples,

            # Remediation
            "remediation": remediation_plan,

            # Meta
            "sensitive_attributes": sensitive_attr_records,
            "fairness_weights":     fairness_weight_records,
            "model_metrics":        model_metrics,
        })

    except Exception as e:
        evaluations[eval_id].update({"status": "error", "error": str(e)})
        import traceback
        print(f"[evaluation error] {eval_id}:\n{traceback.format_exc()}")


# ── Routes ────────────────────────────────────────────────────────────────────

@evaluation_routes.route("/evaluate", methods=["POST"])
def start_evaluation():
    data = request.json or {}

    dataset_id           = data.get("dataset_id")
    model_id             = data.get("model_id")
    sensitive_attributes = data.get("sensitive_attributes", [])
    fairness_weights     = data.get("fairness_weights", {})
    report_type          = data.get("report_type", "developer").lower()
    target_variable      = (data.get("target_variable")     or "").strip()
    prediction_variable  = (data.get("prediction_variable") or "").strip()

    missing = [f for f, v in [
        ("dataset_id",      dataset_id),
        ("model_id",        model_id),
        ("target_variable", target_variable),
    ] if not v]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    if report_type not in VALID_REPORT_TYPES:
        return jsonify({"error": f"Invalid report_type '{report_type}'"}), 400

    try:
        dataset_path = _find_file(dataset_id)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    eval_id = str(uuid.uuid4())
    evaluations[eval_id] = {
        "status":              "queued",
        "current_step":        0,
        "records":             0,
        "dataset_id":          dataset_id,
        "model_id":            model_id,
        "target_variable":     target_variable,
        "prediction_variable": prediction_variable,
        "report_type":         REPORT_TYPE_MAP.get(report_type, "DEVELOPER"),
    }

    thread = threading.Thread(
        target=_run_evaluation,
        args=(
            eval_id, dataset_path, target_variable, prediction_variable,
            sensitive_attributes, fairness_weights, report_type, model_id,
        ),
        daemon=True,
    )
    thread.start()

    return jsonify({
        "evaluation_id":  eval_id,
        "status":         "queued",
        "estimated_time": "2-3 minutes",
        "poll_url":       f"/evaluate/{eval_id}",
    }), 202


@evaluation_routes.route("/evaluate/<eval_id>", methods=["GET"])
def get_evaluation(eval_id: str):
    result = evaluations.get(eval_id)
    if result is None:
        return jsonify({"error": f"Evaluation '{eval_id}' not found"}), 404
    return jsonify({"evaluation_id": eval_id, **result}), 200


@evaluation_routes.route("/evaluate", methods=["GET"])
def list_evaluations():
    summary = [
        {
            "evaluation_id":    eid,
            "status":           v.get("status"),
            "current_step":     v.get("current_step"),
            "ethical_score":    v.get("ethical_score"),
            "fairness_score":   v.get("fairness_score"),
            "privacy_score":    v.get("privacy_score"),
            "robustness_score": v.get("robustness_score"),
            "report_type":      v.get("report_type"),
        }
        for eid, v in evaluations.items()
    ]
    return jsonify({"evaluations": summary, "count": len(summary)}), 200