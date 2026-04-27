"""
AEGIS -- Model Evaluator
==========================

Computes both performance and fairness metrics for every trained model
produced by the Trainer module.

Performance metrics:
    - accuracy, precision, recall, F1

Fairness metrics:
    - demographic_parity_diff
    - equal_opportunity_diff
    - disparate_impact
    - fpr_gap   (false positive rate gap)
    - fnr_gap   (false negative rate gap)

Also supports:
    - Group-wise metric breakdown
    - Intersectional group metrics
    - Before vs after comparison tables
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from .utils import get_logger, safe_divide

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

EvaluationResult = Dict[str, Any]
# {
#     "performance": {accuracy, precision, recall, f1},
#     "fairness": {demographic_parity_diff, equal_opportunity_diff, ...},
#     "group_metrics": {group_name: {accuracy, fpr, fnr, ...}},
#     "intersectional_metrics": {...},
# }


# ---------------------------------------------------------------------------
# Core Metric Functions
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Classification Metrics
# ---------------------------------------------------------------------------

def compute_performance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute standard classification performance metrics."""
    n_classes = len(np.unique(y_true))
    avg = "binary" if n_classes <= 2 else "weighted"

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0, average=avg)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0, average=avg)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0, average=avg)),
    }


def compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compute per-group classification metrics."""
    groups = np.unique(sensitive)
    result: Dict[str, Dict[str, float]] = {}

    for group in groups:
        mask = sensitive == group
        yt = y_true[mask]
        yp = y_pred[mask]

        if len(yt) == 0:
            result[str(group)] = {"count": 0, "accuracy": 0.0, "positive_rate": 0.0, "true_positive_rate": 0.0, "false_positive_rate": 0.0, "false_negative_rate": 0.0}
            continue

        tp = float(((yt == 1) & (yp == 1)).sum())
        fp = float(((yt == 0) & (yp == 1)).sum())
        tn = float(((yt == 0) & (yp == 0)).sum())
        fn = float(((yt == 1) & (yp == 0)).sum())

        result[str(group)] = {
            "count": int(mask.sum()),
            "accuracy": float(accuracy_score(yt, yp)),
            "positive_rate": safe_divide(float((yp == 1).sum()), float(len(yp))),
            "true_positive_rate": safe_divide(tp, tp + fn),
            "false_positive_rate": safe_divide(fp, fp + tn),
            "false_negative_rate": safe_divide(fn, fn + tp),
        }

    return result


# ---------------------------------------------------------------------------
# Regression Metrics
# ---------------------------------------------------------------------------

def compute_regression_performance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute standard regression performance metrics."""
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        # Add a fake 'accuracy' for UI consistency if needed, but MSE is better
        "accuracy": 1.0 / (1.0 + float(mean_absolute_error(y_true, y_pred))),
    }


def compute_regression_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compute per-group regression metrics."""
    groups = np.unique(sensitive)
    result: Dict[str, Dict[str, float]] = {}

    for group in groups:
        mask = sensitive == group
        yt = y_true[mask]
        yp = y_pred[mask]

        if len(yt) == 0:
            result[str(group)] = {"count": 0, "mse": 0.0, "mae": 0.0, "mean_prediction": 0.0}
            continue

        result[str(group)] = {
            "count": int(mask.sum()),
            "mse": float(mean_squared_error(yt, yp)),
            "mae": float(mean_absolute_error(yt, yp)),
            "r2": float(r2_score(yt, yp)) if len(yt) > 1 else 0.0,
            "mean_prediction": float(np.mean(yp)),
            # For UI compatibility with classification tables
            "positive_rate": float(np.mean(yp)), 
            "accuracy": 1.0 / (1.0 + float(mean_absolute_error(yt, yp))),
        }

    return result


def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    problem_type: str = "classification",
) -> Dict[str, float]:
    """Compute standard fairness metrics for classification or regression."""
    if problem_type == "regression":
        group_metrics = compute_regression_group_metrics(y_true, y_pred, sensitive)
        mean_preds = [gm["mean_prediction"] for gm in group_metrics.values()]
        maes = [gm["mae"] for gm in group_metrics.values()]
        
        return {
            "demographic_parity_diff": round(max(mean_preds) - min(mean_preds), 4) if mean_preds else 0.0,
            "disparate_impact": round(safe_divide(min(mean_preds), max(mean_preds)), 4) if mean_preds else 0.0,
            "mean_absolute_error_gap": round(max(maes) - min(maes), 4) if maes else 0.0,
        }
    else:
        group_metrics = compute_group_metrics(y_true, y_pred, sensitive)

        positive_rates = [gm["positive_rate"] for gm in group_metrics.values()]
        tprs = [gm["true_positive_rate"] for gm in group_metrics.values()]
        fprs = [gm["false_positive_rate"] for gm in group_metrics.values()]
        fnrs = [gm["false_negative_rate"] for gm in group_metrics.values()]

        dp_diff = max(positive_rates) - min(positive_rates) if positive_rates else 0.0
        eo_diff = max(tprs) - min(tprs) if tprs else 0.0
        di_ratio = safe_divide(min(positive_rates), max(positive_rates), default=0.0) if positive_rates else 0.0
        fpr_gap = max(fprs) - min(fprs) if fprs else 0.0
        fnr_gap = max(fnrs) - min(fnrs) if fnrs else 0.0

        return {
            "demographic_parity_diff": round(dp_diff, 4),
            "equal_opportunity_diff": round(eo_diff, 4),
            "disparate_impact": round(di_ratio, 4),
            "fpr_gap": round(fpr_gap, 4),
            "fnr_gap": round(fnr_gap, 4),
        }


def compute_intersectional_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for intersectional subgroups.

    Creates composite groups from all combinations of sensitive features
    and computes per-group metrics for each combination.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        sensitive_features: Dict mapping feature name -> array of group labels.

    Returns:
        Dict mapping composite group name -> metrics dict.
    """
    if len(sensitive_features) < 2:
        return {}

    # Build composite group labels
    feature_names = sorted(sensitive_features.keys())
    composite = pd.Series([""] * len(y_true))
    for name in feature_names:
        vals = sensitive_features[name]
        composite = composite + vals.astype(str) + "_"

    return compute_group_metrics(y_true, y_pred, composite.values)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    sensitive_features: Optional[Dict[str, np.ndarray]] = None,
    problem_type: str = "classification",
) -> EvaluationResult:
    """
    Full evaluation of a single model.
    """
    if problem_type == "regression":
        performance = compute_regression_performance_metrics(y_true, y_pred)
        group = compute_regression_group_metrics(y_true, y_pred, sensitive)
    else:
        performance = compute_performance_metrics(y_true, y_pred)
        group = compute_group_metrics(y_true, y_pred, sensitive)
        
    fairness = compute_fairness_metrics(y_true, y_pred, sensitive, problem_type)

    intersectional = {}
    if sensitive_features and len(sensitive_features) >= 2:
        if problem_type == "regression":
            # Intersectional regression - build composite labels
            feature_names = sorted(sensitive_features.keys())
            composite = pd.Series([""] * len(y_true))
            for name in feature_names:
                vals = sensitive_features[name]
                composite = composite + vals.astype(str) + "_"
            intersectional = compute_regression_group_metrics(y_true, y_pred, composite.values)
        else:
            intersectional = compute_intersectional_metrics(
                y_true, y_pred, sensitive_features
            )

    return {
        "performance": performance,
        "fairness": fairness,
        "group_metrics": group,
        "intersectional_metrics": intersectional,
    }


def evaluate_all_models(
    training_output: Dict[str, Any],
    sensitive_features: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, EvaluationResult]:
    """
    Evaluate every trained model from the Trainer output.

    Args:
        training_output: Dict mapping pipeline names to TrainedModel dicts.
        sensitive_features: Optional dict of all sensitive features.

    Returns:
        Dict mapping pipeline names to EvaluationResult dicts.
    """
    logger.info(f"Evaluating {len(training_output)} models ...")

    results: Dict[str, EvaluationResult] = {}

    for pipeline_name, trained in training_output.items():
        try:
            y_true = np.asarray(trained["y_test"])
            y_pred = np.asarray(trained["predictions"])
            sensitive = np.asarray(trained["sensitive_test"])
            problem_type = trained.get("problem_type", "classification")

            # Build per-feature sensitive arrays for intersectional analysis
            feat_dict = None
            if sensitive_features:
                # Use test indices to slice
                test_idx = trained["sensitive_test"].index
                feat_dict = {}
                for feat_name, feat_arr in sensitive_features.items():
                    if isinstance(feat_arr, pd.Series):
                        try:
                            feat_dict[feat_name] = feat_arr.loc[test_idx].values
                        except (KeyError, ValueError):
                            # Fallback if indices don't match exactly
                            feat_dict[feat_name] = np.asarray(feat_arr)[:len(y_true)]
                    else:
                        feat_dict[feat_name] = np.asarray(feat_arr)[:len(y_true)]

            evaluation = evaluate_model(y_true, y_pred, sensitive, feat_dict, problem_type)
            results[pipeline_name] = evaluation

            acc_or_mse = evaluation['performance'].get('accuracy', 0) if problem_type == 'classification' else evaluation['performance'].get('mse', 0)
            metric_label = "acc" if problem_type == 'classification' else "mse"
            
            logger.info(
                f"  [{pipeline_name}] "
                f"{metric_label}={acc_or_mse:.4f}  "
                f"dp_diff={evaluation['fairness'].get('demographic_parity_diff', 0):.4f}  "
            )

        except Exception as e:
            logger.error(f"Evaluation failed for '{pipeline_name}': {e}", exc_info=True)
            continue

    return results


def compare_before_after(
    baseline_eval: EvaluationResult,
    mitigated_eval: EvaluationResult,
) -> Dict[str, Any]:
    """
    Compute a before-vs-after comparison between baseline and mitigated
    evaluations.

    Args:
        baseline_eval: Evaluation result for the baseline (no mitigation).
        mitigated_eval: Evaluation result for the best mitigated model.

    Returns:
        Dict with deltas and improvement flags for each metric.
    """
    comparison: Dict[str, Any] = {}

    # Performance comparison
    for metric in ["accuracy", "precision", "recall", "f1"]:
        before = baseline_eval["performance"].get(metric, 0)
        after = mitigated_eval["performance"].get(metric, 0)
        comparison[f"{metric}_before"] = round(before, 4)
        comparison[f"{metric}_after"] = round(after, 4)
        comparison[f"{metric}_delta"] = round(after - before, 4)

    # Fairness comparison (lower is better for gap metrics)
    for metric in ["demographic_parity_diff", "equal_opportunity_diff", "fpr_gap", "fnr_gap"]:
        before = baseline_eval["fairness"].get(metric, 0)
        after = mitigated_eval["fairness"].get(metric, 0)
        comparison[f"{metric}_before"] = round(before, 4)
        comparison[f"{metric}_after"] = round(after, 4)
        comparison[f"{metric}_delta"] = round(after - before, 4)
        comparison[f"{metric}_improved"] = after < before

    # Disparate impact (higher is better, closer to 1.0)
    di_before = baseline_eval["fairness"].get("disparate_impact", 0)
    di_after = mitigated_eval["fairness"].get("disparate_impact", 0)
    comparison["disparate_impact_before"] = round(di_before, 4)
    comparison["disparate_impact_after"] = round(di_after, 4)
    comparison["disparate_impact_delta"] = round(di_after - di_before, 4)
    comparison["disparate_impact_improved"] = di_after > di_before

    # Overall fairness improvement score
    fairness_improvements = [
        v for k, v in comparison.items()
        if k.endswith("_improved") and isinstance(v, bool) and v
    ]
    comparison["fairness_improvement_count"] = len(fairness_improvements)

    return comparison
