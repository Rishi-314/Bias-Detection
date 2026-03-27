from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
)
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_absolute_error,
)


def compute_fairness(
    df: pd.DataFrame,
    preds: pd.DataFrame,
    target_col: str = "approved",
    sensitive_attrs: list = None,
) -> dict:
    """
    Compute all 7 fairness metrics required by the schema's FairnessMetrics model:

        individualFairness  – consistency of predictions for similar individuals (proxy)
        groupFairness       – average equal-opportunity across groups
        demographicParity   – demographic parity difference (main fairlearn metric)
        disparateImpact     – 80% rule ratio (favorable outcome rate ratio)
        calibrationError    – mean calibration error across sensitive groups
        counterfactual      – proxy: variance of pred changes across groups
        intersectional      – worst-case parity gap across attribute intersections

    Also returns:
        fairness_score      – maps to Evaluation.ethicalScore  (0–1, higher is better)
        per_attribute       – per-attribute breakdown
        records_evaluated
        target_column
        prediction_column
    """
    sensitive_attrs = sensitive_attrs or []

    # ── Resolve ground-truth column ─────────────────────────────────────
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataset. "
            f"Available: {list(df.columns)}"
        )
    y_true = df[target_col].copy()

    # ── Resolve prediction column ────────────────────────────────────────
    pred_col = None
    for candidate in ["prediction", "prediction_score", "score", "label", "output", "pred"]:
        if candidate in preds.columns:
            pred_col = candidate
            break
    if pred_col is None:
        numeric_cols = preds.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            pred_col = numeric_cols[0]
        else:
            raise ValueError(
                "Could not find prediction column. "
                "Expected: prediction, prediction_score, score, label, output"
            )

    raw_scores = preds[pred_col].copy()

    # Binarise predictions (keep raw for calibration)
    if raw_scores.between(0, 1).all():
        y_pred = (raw_scores >= 0.5).astype(int)
        y_prob = raw_scores
    else:
        y_pred = raw_scores.astype(int)
        y_prob = None

    # ── Align lengths ────────────────────────────────────────────────────
    n = min(len(y_true), len(y_pred))
    y_true     = y_true.iloc[:n].reset_index(drop=True).astype(int)
    y_pred     = y_pred.iloc[:n].reset_index(drop=True)
    if y_prob is not None:
        y_prob = y_prob.iloc[:n].reset_index(drop=True)
    df_trim    = df.iloc[:n].reset_index(drop=True)

    # ── Filter valid sensitive attributes ────────────────────────────────
    valid_attrs = [a for a in sensitive_attrs if a in df_trim.columns]
    
    if len(valid_attrs) == 0:
        raise ValueError("No valid sensitive attributes found for fairness computation")

    # ── Per-attribute metrics ────────────────────────────────────────────
    per_attribute = {}
    dp_values, eo_values = [], []

    for attr in valid_attrs:
        sensitive = df_trim[attr]
        try:
            dp = float(demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive))
            eo = float(equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive))
            dp_values.append(dp)
            eo_values.append(eo)
            per_attribute[attr] = {
                "demographic_parity_difference": round(dp, 4),
                "equalized_odds_difference":     round(eo, 4),
                "fairness_score":                round(1 - abs(dp), 4),
            }
        except Exception as e:
            per_attribute[attr] = {"error": str(e)}

    # ── 1. Demographic Parity ────────────────────────────────────────────
    # Average |dp| across attrs; 0 = perfect parity
    demographic_parity = float(np.mean([abs(v) for v in dp_values])) if dp_values else 0.0

    # ── 2. Individual Fairness ───────────────────────────────────────────
    # Proxy: 1 - std(prediction) within near-identical records (simplified)
    # Full Lipschitz individual fairness needs a similarity metric; here we
    # use consistency: proportion of neighbours with same prediction.
    try:
        numeric_df = df_trim.select_dtypes(include="number").drop(
            columns=[target_col], errors="ignore"
        )
        if not numeric_df.empty and len(numeric_df) > 10:
            # Sample 500 rows for speed
            idx      = np.random.choice(len(numeric_df), min(500, len(numeric_df)), replace=False)
            sample   = numeric_df.iloc[idx].values
            preds_s  = y_pred.iloc[idx].values

            # For each row, find 5-NN and check prediction consistency
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=6).fit(sample)
            _, indices = nn.kneighbors(sample)
            consistency = np.mean([
                np.mean(preds_s[indices[i][1:]] == preds_s[i])
                for i in range(len(sample))
            ])
            individual_fairness = float(round(consistency, 4))
        else:
            individual_fairness = 0.5   # not enough data
    except Exception:
        individual_fairness = 0.5

    # ── 3. Group Fairness ────────────────────────────────────────────────
    # Average equal-opportunity (1 - |EOD|) across attrs
    eo_scores = [1 - abs(v) for v in eo_values]
    group_fairness = float(np.mean(eo_scores)) if eo_scores else 0.5

    # ── 4. Disparate Impact ──────────────────────────────────────────────
    # 80% rule: min(P(y=1|g)) / max(P(y=1|g)) across groups
    # If no attrs, compute globally (=1.0 perfect)
    di_ratios = []
    for attr in valid_attrs:
        sensitive = df_trim[attr]
        groups    = sensitive.unique()
        rates     = []
        for g in groups:
            mask = sensitive == g
            if mask.sum() > 0:
                rates.append(y_pred[mask].mean())
        if len(rates) >= 2:
            di_ratios.append(min(rates) / max(rates) if max(rates) > 0 else 1.0)

    disparate_impact = float(np.mean(di_ratios)) if di_ratios else 1.0

    # ── 5. Calibration Error ─────────────────────────────────────────────
    # Mean calibration error: |P(y=1|score_bucket) - score_bucket| per group
    if y_prob is not None and valid_attrs:
        cal_errors = []
        for attr in valid_attrs:
            sensitive = df_trim[attr]
            for g in sensitive.unique():
                mask = sensitive == g
                if mask.sum() > 5:
                    try:
                        cal_errors.append(
                            float(brier_score_loss(y_true[mask], y_prob[mask]))
                        )
                    except Exception:
                        pass
        calibration_error = float(np.mean(cal_errors)) if cal_errors else 0.0
    else:
        # Fallback: overall brier score
        if y_prob is not None:
            try:
                calibration_error = float(brier_score_loss(y_true, y_prob))
            except Exception:
                calibration_error = 0.0
        else:
            calibration_error = 0.0

    # ── 6. Counterfactual Fairness ───────────────────────────────────────
    # Proxy: variance of group-level prediction rates — lower = more counterfactually fair
    cf_scores = []
    for attr in valid_attrs:
        sensitive = df_trim[attr]
        group_rates = [
            y_pred[sensitive == g].mean()
            for g in sensitive.unique()
            if (sensitive == g).sum() > 0
        ]
        if len(group_rates) >= 2:
            # Normalise: 1 - std (std=0 means identical rates = fair)
            cf_scores.append(max(0.0, 1.0 - float(np.std(group_rates))))

    counterfactual = float(np.mean(cf_scores)) if cf_scores else 0.5

    # ── 7. Intersectional Fairness ───────────────────────────────────────
    # Worst-case demographic parity gap across all pairwise attr intersections
    if len(valid_attrs) >= 2:
        intersect_gaps = []
        for i in range(len(valid_attrs)):
            for j in range(i + 1, len(valid_attrs)):
                a1, a2 = valid_attrs[i], valid_attrs[j]
                combo   = df_trim[a1].astype(str) + "_" + df_trim[a2].astype(str)
                rates   = [
                    y_pred[combo == g].mean()
                    for g in combo.unique()
                    if (combo == g).sum() > 0
                ]
                if len(rates) >= 2:
                    intersect_gaps.append(max(rates) - min(rates))
        intersectional = float(1.0 - np.mean(intersect_gaps)) if intersect_gaps else 0.5
    elif len(valid_attrs) == 1:
        intersectional = group_fairness   # same as group fairness with one attr
    else:
        intersectional = 0.5

    # ── Aggregate fairness score → Evaluation.ethicalScore ───────────────
    # Weighted average of the 7 dimensions (equal weights by default)
    metrics_list = [
        individual_fairness,
        group_fairness,
        1.0 - demographic_parity,    # higher = less parity gap = better
        disparate_impact,
        1.0 - calibration_error,     # lower error = better
        counterfactual,
        intersectional,
    ]
    # Clamp all to [0, 1]
    metrics_clamped = [max(0.0, min(1.0, m)) for m in metrics_list]
    cf_penalty = 0.0

    if counterfactual < 0.95:   # threshold
        cf_penalty = (0.95 - counterfactual)

    fairness_score = float(round(max(0.0, np.mean(metrics_clamped) - cf_penalty), 4))

    return {
        # ── 7 fields mapping to FairnessMetrics schema ──────────────────
        "individualFairness": round(individual_fairness, 4),
        "groupFairness":      round(group_fairness, 4),
        "demographicParity":  round(demographic_parity, 4),     # raw gap (lower = better)
        "disparateImpact":    round(disparate_impact, 4),
        "calibrationError":   round(calibration_error, 4),
        "counterfactual":     round(counterfactual, 4),
        "intersectional":     round(intersectional, 4),

        # ── Evaluation.ethicalScore ──────────────────────────────────────
        "fairness_score":     fairness_score,

        # ── Extra context ────────────────────────────────────────────────
        "per_attribute":      per_attribute,
        "records_evaluated":  int(n),
        "target_column":      target_col,
        "prediction_column":  pred_col,
    }