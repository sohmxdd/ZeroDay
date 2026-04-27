"""
AEGIS — Model Explainability
================================

Phase 4 of the AEGIS pipeline.  Provides interpretability and
transparency for the models trained during bias mitigation.

Analysis performed:
    - Feature importance (coefficients / tree importance)
    - SHAP values (if ``shap`` is available)
    - Permutation importance (fallback)
    - Model comparison table (baseline vs best)
    - LLM-generated explanation of model behaviour
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..config import get_config, get_logger

logger = get_logger(__name__)


def _safe_import(module_name: str):
    """Attempt to import a module, returning None if unavailable."""
    try:
        import importlib
        return importlib.import_module(module_name)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explain_model(
    model_output: Dict[str, Any],
    mitigation_result: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate model explainability analysis.
    """
    logger.info("Running model explainability analysis (Phase 4)...")

    cfg = get_config(config)
    result: Dict[str, Any] = {
        "feature_importance": {},
        "model_comparison": {},
        "predictions_analysis": {},
        "shap_summary": None,
        "explanation": "",
    }

    baseline_model = model_output.get("baseline_model")
    best_model = model_output.get("best_model")
    best_strategy = model_output.get("best_strategy", "unknown")

    # Detect problem type
    training_output = mitigation_result.get("_training_output", {})
    baseline_trained = training_output.get("baseline", {})
    problem_type = baseline_trained.get("problem_type", "classification")

    # --- Feature Importance ---
    result["feature_importance"] = _extract_feature_importance(
        baseline_model, best_model, mitigation_result
    )

    # --- Model Comparison ---
    result["model_comparison"] = _build_model_comparison(model_output)

    # --- Predictions Analysis ---
    result["predictions_analysis"] = _analyse_predictions(model_output, problem_type)

    # --- SHAP (Optional) ---
    result["shap_summary"] = _compute_shap(best_model, mitigation_result)

    # --- LLM Explanation ---
    result["explanation"] = _generate_explainability_text(
        result, best_strategy, problem_type, cfg
    )

    logger.info(f"Model explainability analysis complete (Type: {problem_type}).")
    return result


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------

def _extract_feature_importance(
    baseline_model: Any,
    best_model: Any,
    mitigation_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract feature importance from models."""
    importance = {
        "baseline": {},
        "best": {},
    }

    # Get feature names from training output
    training_output = mitigation_result.get("_training_output", {})
    baseline_trained = training_output.get("baseline", {})
    feature_names = None

    if baseline_trained:
        X_train = baseline_trained.get("X_train")
        if X_train is not None and hasattr(X_train, "columns"):
            feature_names = list(X_train.columns)

    for model, key in [(baseline_model, "baseline"), (best_model, "best")]:
        if model is None:
            continue

        fi = _get_importance_from_model(model, feature_names)
        if fi:
            importance[key] = fi

    return importance


def _get_importance_from_model(
    model: Any,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Try to extract feature importance from an sklearn model."""
    importances = None

    # Tree-based models
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    # Linear models
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if len(coef.shape) > 1:
            importances = np.abs(coef[0])
        else:
            importances = np.abs(coef)

    # Fairlearn wrapper
    elif hasattr(model, "predictors_"):
        try:
            # Try to get from the first predictor
            for pred in model.predictors_:
                return _get_importance_from_model(pred, feature_names)
        except Exception:
            pass

    if importances is None:
        return {}

    if feature_names and len(feature_names) == len(importances):
        sorted_pairs = sorted(
            zip(feature_names, importances),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return {name: round(float(imp), 4) for name, imp in sorted_pairs}

    return {
        f"feature_{i}": round(float(v), 4)
        for i, v in enumerate(importances)
    }


# ---------------------------------------------------------------------------
# Model Comparison
# ---------------------------------------------------------------------------

def _build_model_comparison(model_output: Dict[str, Any]) -> Dict[str, Any]:
    """Build a comparison between baseline and best models."""
    comparison_table = model_output.get("comparison_table", [])
    metrics_before_after = model_output.get("metrics_before_after", {})

    # Extract baseline and best from comparison table
    baseline_row = None
    best_row = None

    for row in comparison_table:
        if row.get("strategy") == "baseline":
            baseline_row = row
        elif row.get("rank") == 1:
            best_row = row

    return {
        "baseline_metrics": baseline_row or {},
        "best_metrics": best_row or {},
        "improvement_summary": metrics_before_after,
        "strategies_evaluated": len(comparison_table),
        "comparison_table": comparison_table,
    }


# ---------------------------------------------------------------------------
# Predictions Analysis
# ---------------------------------------------------------------------------

def _analyse_predictions(model_output: Dict[str, Any], problem_type: str = "classification") -> Dict[str, Any]:
    """Analyse prediction distributions."""
    baseline_preds = model_output.get("baseline_predictions")
    best_preds = model_output.get("best_predictions")

    analysis = {}

    if baseline_preds is not None:
        baseline_arr = np.asarray(baseline_preds)
        if problem_type == "classification":
            analysis["baseline"] = {
                "total": len(baseline_arr),
                "positive_count": int((baseline_arr == 1).sum()),
                "negative_count": int((baseline_arr == 0).sum()),
                "positive_rate": round(float((baseline_arr == 1).mean()), 4),
            }
        else:
            analysis["baseline"] = {
                "total": len(baseline_arr),
                "mean": round(float(np.mean(baseline_arr)), 4),
                "std": round(float(np.std(baseline_arr)), 4),
                "min": round(float(np.min(baseline_arr)), 4),
                "max": round(float(np.max(baseline_arr)), 4),
            }

    if best_preds is not None:
        best_arr = np.asarray(best_preds)
        if problem_type == "classification":
            analysis["best"] = {
                "total": len(best_arr),
                "positive_count": int((best_arr == 1).sum()),
                "negative_count": int((best_arr == 0).sum()),
                "positive_rate": round(float((best_arr == 1).mean()), 4),
            }
        else:
            analysis["best"] = {
                "total": len(best_arr),
                "mean": round(float(np.mean(best_arr)), 4),
                "std": round(float(np.std(best_arr)), 4),
                "min": round(float(np.min(best_arr)), 4),
                "max": round(float(np.max(best_arr)), 4),
            }

    if baseline_preds is not None and best_preds is not None:
        baseline_arr = np.asarray(baseline_preds)
        best_arr = np.asarray(best_preds)
        min_len = min(len(baseline_arr), len(best_arr))
        if min_len > 0:
            if problem_type == "classification":
                agreement = float((baseline_arr[:min_len] == best_arr[:min_len]).mean())
                analysis["prediction_agreement"] = round(agreement, 4)
                analysis["predictions_changed"] = int((baseline_arr[:min_len] != best_arr[:min_len]).sum())
            else:
                correlation = float(np.corrcoef(baseline_arr[:min_len], best_arr[:min_len])[0, 1])
                analysis["prediction_correlation"] = round(correlation, 4)
                analysis["mean_shift"] = round(float(np.mean(best_arr[:min_len]) - np.mean(baseline_arr[:min_len])), 4)

    return analysis


# ---------------------------------------------------------------------------
# SHAP Integration
# ---------------------------------------------------------------------------

def _compute_shap(
    model: Any,
    mitigation_result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Compute SHAP values if the shap library is available."""
    if model is None:
        return None

    shap_mod = _safe_import("shap")
    if shap_mod is None:
        logger.info("SHAP not available — skipping SHAP analysis.")
        return None

    try:
        training_output = mitigation_result.get("_training_output", {})
        best_trained = None
        for key, val in training_output.items():
            if key != "baseline":
                best_trained = val
                break

        if not best_trained:
            best_trained = training_output.get("baseline", {})

        X_test = best_trained.get("X_test")
        if X_test is None:
            return None

        # Use a small sample for efficiency
        sample_size = min(100, len(X_test))
        X_sample = X_test.head(sample_size)

        # Try TreeExplainer first, then fallback to KernelExplainer
        try:
            explainer = shap_mod.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        except Exception:
            try:
                explainer = shap_mod.LinearExplainer(model, X_sample)
                shap_values = explainer.shap_values(X_sample)
            except Exception:
                return None

        # Compute mean absolute SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification

        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        feature_names = list(X_sample.columns) if hasattr(X_sample, "columns") else [
            f"feature_{i}" for i in range(len(mean_abs_shap))
        ]

        sorted_pairs = sorted(
            zip(feature_names, mean_abs_shap),
            key=lambda x: x[1],
            reverse=True,
        )

        return {
            "top_features": {
                name: round(float(val), 4) for name, val in sorted_pairs[:10]
            },
            "sample_size": sample_size,
            "method": "shap",
        }

    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# LLM Explanation
# ---------------------------------------------------------------------------

def _generate_explainability_text(
    result: Dict[str, Any],
    best_strategy: str,
    problem_type: str,
    config: Dict[str, Any],
) -> str:
    """Generate a text explanation of model behaviour."""
    from ..llm.gemini_client import GeminiClient

    # Build context for LLM
    fi = result.get("feature_importance", {}).get("best", {})
    top_features = list(fi.keys())[:5] if fi else []

    preds = result.get("predictions_analysis", {})
    agreement = preds.get("prediction_agreement", preds.get("prediction_correlation", "N/A"))

    comparison = result.get("model_comparison", {})
    baseline_metrics = comparison.get("baseline_metrics", {})
    best_metrics = comparison.get("best_metrics", {})

    client = GeminiClient(config=config)

    metrics_str = f"Baseline Accuracy: {baseline_metrics.get('accuracy', 'N/A')}, New Accuracy: {best_metrics.get('accuracy', 'N/A')}"
    if problem_type == "regression":
        metrics_str = f"Baseline MSE: {baseline_metrics.get('mse', 'N/A')}, New MSE: {best_metrics.get('mse', 'N/A')}"

    prompt = f"""You are an AI explainability expert. Provide a brief explanation of
how the bias mitigation affected the model's behavior:

- Problem type: {problem_type}
- Strategy used: {best_strategy}
- Top important features: {top_features}
- Prediction similarity (agreement/correlation) with baseline: {agreement}
- {metrics_str}

Explain in 2-3 sentences what changed and why. Be concise.
"""

    response = client._call(prompt)
    if response:
        return response

    # Fallback
    return (
        f"The '{best_strategy}' strategy was applied. "
        f"Top features: {', '.join(top_features[:3]) if top_features else 'N/A'}. "
        f"See metrics for detailed comparison."
    )
