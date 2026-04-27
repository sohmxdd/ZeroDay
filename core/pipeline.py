"""
AEGIS — Master Pipeline Orchestrator
========================================

Central entry point for the entire AEGIS AI Bias Governance Engine.

This module implements mode-based routing:

    "analysis"      → Detection + Dataset Comparison only
    "train"         → Detection + Mitigation + Model training
    "full_pipeline" → Detection + Mitigation + Comparison + Explainability

Speed modes:
    "fast"  → Subsampled data, 1 strategy, no SHAP, deferred LLM (default for UI)
    "full"  → All strategies, full evaluation, SHAP, LLM in path

Usage::

    from core import run_pipeline

    result = run_pipeline({
        "dataset": "path/to/data.csv",   # or DataFrame or None (synthetic)
        "mode": "full_pipeline",
        "speed": "fast",                 # NEW: "fast" (default) or "full"
        "model": None,
        "target": "approved",
        "sensitive_features": ["gender", "race"],
    })
"""

import json
import os
import time
import signal
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .config import get_config, get_logger, OUTPUT_DIR
from .utils.data_loader import load_dataset, create_synthetic_dataset
from .bias_detection import detect_bias, preprocess_dataset
from .bias_mitigation import BiasMitigationEngine
from .dataset_comparison import compare_datasets
from .explainability import explain_model
from .llm import GeminiClient

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Performance Constants
# ---------------------------------------------------------------------------

FAST_MAX_SAMPLES = 2000         # Subsample cap in fast mode (enough for statistical validity)
FAST_THRESHOLD_STEPS = 10       # Threshold search resolution
PIPELINE_TIMEOUT_SECONDS = 30   # Hard timeout failsafe
STRATEGY_TIMEOUT_SECONDS = 5    # Per-strategy timeout
FAST_MAX_COMBOS = 2             # Max combined strategies in fast mode


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def run_pipeline(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the AEGIS pipeline end-to-end.

    Args:
        input_data: Input configuration dict with keys:
            - ``dataset``: DataFrame, file path string, or None (synthetic)
            - ``mode``: ``"analysis"`` | ``"train"`` | ``"full_pipeline"`` (default)
            - ``speed``: ``"fast"`` (default) | ``"full"``
            - ``model``: Optional pre-trained model object
            - ``target``: Target column name (default: auto-detect)
            - ``sensitive_features``: List of sensitive columns (default: auto-detect)
            - ``config``: Optional dict of configuration overrides

    Returns:
        Unified output dict with keys:
            - ``dataset_analysis``
            - ``model_analysis``
            - ``explanations``
            - ``metadata``
    """
    pipeline_start = time.time()

    # --- Parse Input ---
    dataset_source = input_data.get("dataset")
    mode = input_data.get("mode", "full_pipeline")
    speed = input_data.get("speed", "fast")
    user_model = input_data.get("model")
    target = input_data.get("target")
    sensitive_features = input_data.get("sensitive_features")
    config_overrides = input_data.get("config", {})

    config = get_config(config_overrides)

    # Inject speed mode optimizations into config
    is_fast = speed == "fast"
    if is_fast:
        config["threshold_search_steps"] = FAST_THRESHOLD_STEPS
        config["strategy_timeout"] = STRATEGY_TIMEOUT_SECONDS
        config["max_combos"] = FAST_MAX_COMBOS
        config["speed"] = "fast"
        # Fast model params
        config["rf_n_estimators"] = 50
        config["rf_max_depth"] = 6
    else:
        config["speed"] = "full"
        config["strategy_timeout"] = 15  # More generous timeout in full mode
        config["max_combos"] = -1  # Unlimited

    print("=" * 70)
    print("  AEGIS - AI Bias Governance Engine")
    print("=" * 70)
    print(f"  Mode: {mode} | Speed: {speed}")

    # --- Phase 0: Load Data ---
    t0 = time.time()
    print(f"\n[PHASE 0] Loading data...")
    df = load_dataset(dataset_source)

    # Auto-detect target if not specified
    if target is None:
        target = _auto_detect_target(df)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Target: '{target}'")

    # *** FAST MODE: Subsample ***
    if is_fast and len(df) > FAST_MAX_SAMPLES:
        print(f"  [FAST] Subsampling {len(df)} -> {FAST_MAX_SAMPLES} rows")
        df = df.sample(n=FAST_MAX_SAMPLES, random_state=42).reset_index(drop=True)

    # Auto-detect sensitive features if not specified
    if sensitive_features is None:
        sensitive_features = _auto_detect_sensitive(df, target, config)
    print(f"  Sensitive features: {sensitive_features}")

    # Validate
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")
    for feat in sensitive_features:
        if feat not in df.columns:
            logger.warning(f"Sensitive feature '{feat}' not in dataset — skipping.")
    sensitive_features = [f for f in sensitive_features if f in df.columns]

    if not sensitive_features:
        raise ValueError("No valid sensitive features found.")

    # Store original categorical info before preprocessing
    original_categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Keep a copy of raw data before encoding for comparison
    df_raw = df.copy()

    # Preprocess if needed (encode categoricals)
    needs_encoding = len(df.select_dtypes(include=["object", "category"]).columns) > 0
    preprocessing_meta = {}
    if needs_encoding:
        df, preprocessing_meta = preprocess_dataset(df, target=target)

    print(f"  Phase 0 time: {time.time() - t0:.2f}s")

    # ===================================================================
    # PHASE 1: Bias Detection (always runs)
    # ===================================================================
    t1 = time.time()
    print(f"\n[PHASE 1] Detecting bias...")
    bias_report = detect_bias(df, target, sensitive_features)
    print(f"  Insights found: {len(bias_report['insights'])}")
    for insight in bias_report["insights"]:
        print(f"    - {insight}")
    print(f"  Phase 1 time: {time.time() - t1:.2f}s")

    # ===================================================================
    # MODE: "analysis" — Detection + Dataset Comparison only
    # ===================================================================
    if mode == "analysis":
        t_analysis = time.time()
        print(f"\n[PHASE 3] Running dataset comparison (analysis mode)...")
        dataset_comparison = compare_datasets(
            baseline_dataset=df,
            debiased_dataset=df,  # No mitigation, compare with self
            target=target,
            sensitive_features=sensitive_features,
            bias_report=bias_report,
        )

        # LLM explanation — deferred in fast mode
        llm_client = GeminiClient(config=config)
        if is_fast:
            bias_explanation = llm_client.explain_bias_fast(bias_report)
        else:
            bias_explanation = llm_client.explain_bias(bias_report)

        elapsed = time.time() - pipeline_start
        print(f"  Phase 3 time: {time.time() - t_analysis:.2f}s")
        print(f"\n{'=' * 70}")
        print(f"  ANALYSIS COMPLETE  ({elapsed:.1f}s)")
        print(f"{'=' * 70}")

        return _build_output(
            mode=mode,
            bias_report=bias_report,
            dataset_comparison=dataset_comparison,
            explanations={
                "bias_explanation": bias_explanation,
                "summary": f"Analysis mode: {len(bias_report['insights'])} bias insights detected.",
                "gemini_used": llm_client.enabled,
            },
            config=config,
            elapsed=elapsed,
        )

    # ===================================================================
    # PHASE 2: Bias Mitigation (runs in "train" and "full_pipeline")
    # ===================================================================
    t2 = time.time()
    print(f"\n[PHASE 2] Running mitigation engine...")
    model_type = "logistic_regression"  # Always use LR in fast mode
    if not is_fast:
        model_type = config.get("model_type", "logistic_regression")
    alpha = config.get("alpha", 0.6)
    beta = config.get("beta", 0.4)
    print(f"  Config: alpha={alpha}, beta={beta}, model={model_type}, speed={speed}")

    engine = BiasMitigationEngine(config=config)
    mitigation_result = engine.run(
        data=df,
        target=target,
        sensitive_features=sensitive_features,
        bias_report=bias_report,
        model_type=model_type,
    )

    # Extract sub-results
    dataset_output = mitigation_result["dataset_output"]
    model_output = mitigation_result["model_output"]
    llm_summary = mitigation_result["llm_summary"]
    ranking = mitigation_result["ranking"]
    bias_tags = mitigation_result["bias_tags"]

    best_strategy = ranking["best_strategy"]
    print(f"  Best strategy: {best_strategy}")
    print(f"  Score: {ranking['best_score']:.4f}")
    print(f"  Phase 2 time: {time.time() - t2:.2f}s")

    # ===================================================================
    # MODE: "train" — Detection + Mitigation only
    # ===================================================================
    if mode == "train":
        elapsed = time.time() - pipeline_start
        print(f"\n{'=' * 70}")
        print(f"  TRAINING COMPLETE  ({elapsed:.1f}s)")
        print(f"  Best Strategy: {best_strategy}")
        print(f"{'=' * 70}")

        return _build_output(
            mode=mode,
            bias_report=bias_report,
            model_output=model_output,
            ranking=ranking,
            bias_tags=bias_tags,
            explanations=llm_summary,
            dataset_output=dataset_output,
            config=config,
            elapsed=elapsed,
        )

    # ===================================================================
    # PHASE 3: Dataset Comparison (full_pipeline only)
    # ===================================================================
    t3 = time.time()
    print(f"\n[PHASE 3] Running dataset comparison...")
    baseline_ds = dataset_output.get("baseline_dataset", df)
    debiased_ds = dataset_output.get("debiased_dataset", df)

    dataset_comparison = compare_datasets(
        baseline_dataset=baseline_ds,
        debiased_dataset=debiased_ds,
        target=target,
        sensitive_features=sensitive_features,
        bias_report=bias_report,
    )
    print(f"  Phase 3 time: {time.time() - t3:.2f}s")

    # ===================================================================
    # PHASE 4: Model Explainability (full_pipeline only)
    # ===================================================================
    t4 = time.time()
    print(f"\n[PHASE 4] Running model explainability...")

    # Attach training output for explainability to access
    mitigation_result["_training_output"] = mitigation_result.get(
        "_training_output", {}
    )

    if is_fast:
        # Skip SHAP, skip LLM in explainability — use lightweight version
        explainability_output = _fast_explainability(model_output, mitigation_result)
    else:
        explainability_output = explain_model(
            model_output=model_output,
            mitigation_result=mitigation_result,
            config=config,
        )
    print(f"  Phase 4 time: {time.time() - t4:.2f}s")

    # ===================================================================
    # FINAL OUTPUT
    # ===================================================================
    elapsed = time.time() - pipeline_start

    # Print summary
    detected_biases = [k for k, v in bias_tags.items() if v]
    print(f"\n{'=' * 70}")
    print(f"  FULL PIPELINE COMPLETE")
    print(f"{'=' * 70}")
    print(f"""
  Bias Types Detected:  {', '.join(detected_biases) if detected_biases else 'none'}
  Best Strategy:        {best_strategy}
  Tradeoff Score:       {ranking['best_score']:.4f}
  Accuracy Change:      {model_output.get('accuracy_drop', 0):+.4f}
  Fairness Improvement: {dataset_output.get('fairness_improvement', 0):.4f}
  Gemini Used:          {llm_summary.get('gemini_used', False)}
  Speed Mode:           {speed}
  Time Elapsed:         {elapsed:.1f}s
""")

    # Print ranking table
    print("  Strategy Ranking:")
    for row in ranking.get("ranking_table", []):
        print(
            f"    #{row['rank']:2d}  {row['pipeline']:42s} "
            f"score={row['score']:.4f}  acc={row['accuracy']:.4f}  "
            f"dp_diff={row['demographic_parity_diff']:.4f}"
        )

    print(f"\n{'=' * 70}")

    # Build final unified output
    final_output = _build_output(
        mode=mode,
        bias_report=bias_report,
        dataset_comparison=dataset_comparison,
        model_output=model_output,
        ranking=ranking,
        bias_tags=bias_tags,
        explanations=llm_summary,
        explainability=explainability_output,
        dataset_output=dataset_output,
        config=config,
        elapsed=elapsed,
    )

    # Save artifacts (async in fast mode)
    if config.get("save_artifacts", True):
        if is_fast:
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(_save_artifacts, final_output, dataset_output, config)
            except Exception:
                pass  # Don't block on save errors
        else:
            _save_artifacts(final_output, dataset_output, config)

    return final_output


# ---------------------------------------------------------------------------
# Fast Explainability (lightweight, no SHAP/LLM)
# ---------------------------------------------------------------------------

def _fast_explainability(
    model_output: Dict[str, Any],
    mitigation_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Lightweight explainability without SHAP or LLM calls."""
    from .explainability.explainer import (
        _extract_feature_importance,
        _build_model_comparison,
        _analyse_predictions,
    )

    result = {
        "feature_importance": {},
        "model_comparison": {},
        "predictions_analysis": {},
        "shap_summary": None,
        "explanation": "",
    }

    baseline_model = model_output.get("baseline_model")
    best_model = model_output.get("best_model")
    best_strategy = model_output.get("best_strategy", "unknown")

    result["feature_importance"] = _extract_feature_importance(
        baseline_model, best_model, mitigation_result
    )
    result["model_comparison"] = _build_model_comparison(model_output)
    result["predictions_analysis"] = _analyse_predictions(model_output)

    # Build rich explanation from actual metrics (no LLM needed)
    comparison = result["model_comparison"]
    preds = result["predictions_analysis"]
    fi = result["feature_importance"]

    baseline_m = comparison.get("baseline_metrics", {})
    best_m = comparison.get("best_metrics", {})

    bl_acc = baseline_m.get("accuracy", "N/A")
    best_acc = best_m.get("accuracy", "N/A")
    bl_dp = baseline_m.get("demographic_parity_diff", "N/A")
    best_dp = best_m.get("demographic_parity_diff", "N/A")
    changed = preds.get("predictions_changed", 0)
    agreement = preds.get("prediction_agreement", "N/A")

    top_feats = list(fi.get("best", {}).keys())[:5]
    top_feats_str = ", ".join(top_feats) if top_feats else "N/A"

    explanation_parts = [
        f"## Strategy: {best_strategy}\n",
        f"The **{best_strategy}** mitigation strategy was selected as the optimal "
        f"fairness-accuracy tradeoff from {comparison.get('strategies_evaluated', 0)} "
        f"evaluated candidates.\n",
        f"### Performance Impact",
        f"- **Baseline accuracy**: {bl_acc if isinstance(bl_acc, str) else f'{bl_acc:.4f}'}",
        f"- **Post-mitigation accuracy**: {best_acc if isinstance(best_acc, str) else f'{best_acc:.4f}'}",
        f"- **Demographic parity gap**: {bl_dp if isinstance(bl_dp, str) else f'{bl_dp:.4f}'} → {best_dp if isinstance(best_dp, str) else f'{best_dp:.4f}'}\n",
        f"### Prediction Changes",
        f"- **Predictions changed**: {changed}",
        f"- **Agreement with baseline**: {agreement if isinstance(agreement, str) else f'{agreement:.1%}'}\n",
        f"### Key Features",
        f"The most influential features in the mitigated model are: {top_feats_str}.\n",
        f"### Interpretation",
        f"The mitigation adjusts how the model weighs evidence from different demographic "
        f"groups to ensure more equitable outcomes while preserving predictive performance.",
    ]

    result["explanation"] = "\n".join(explanation_parts)

    return result


# ---------------------------------------------------------------------------
# Output Builder
# ---------------------------------------------------------------------------

def _build_output(**kwargs) -> Dict[str, Any]:
    """Build the standardised output dictionary."""
    mode = kwargs.get("mode", "full_pipeline")
    config = kwargs.get("config", {})
    elapsed = kwargs.get("elapsed", 0)

    bias_report = kwargs.get("bias_report", {})
    dataset_comparison = kwargs.get("dataset_comparison", {})
    model_output = kwargs.get("model_output", {})
    ranking = kwargs.get("ranking", {})
    bias_tags = kwargs.get("bias_tags", {})
    explanations = kwargs.get("explanations", {})
    explainability = kwargs.get("explainability", {})
    dataset_output = kwargs.get("dataset_output", {})

    return {
        "dataset_analysis": {
            "bias_report": bias_report,
            "dataset_comparison": _make_serialisable(dataset_comparison),
            "bias_tags": bias_tags,
        },
        "model_analysis": {
            "model_output": _strip_non_serialisable(model_output),
            "ranking": ranking,
            "explainability": _make_serialisable(explainability),
        },
        "explanations": explanations,
        "metadata": {
            "mode": mode,
            "strategy_used": ranking.get("best_strategy", "none"),
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "config": {
                "alpha": config.get("alpha", 0.6),
                "beta": config.get("beta", 0.4),
                "model_type": config.get("model_type", "logistic_regression"),
                "gemini_enabled": config.get("gemini_enabled", True),
            },
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_detect_target(df: pd.DataFrame) -> str:
    """Auto-detect the likely target column from common dataset conventions."""
    # Priority 1: Exact match on common target column names
    common_targets = [
        # Generic
        "target", "label", "class", "y", "outcome", "result",
        # Finance / lending
        "approved", "default", "income", "loan_status", "credit_risk",
        "default.payment.next.month", "default_payment",
        # Recidivism
        "is_recid", "two_year_recid", "recidivism", "is_recidivst",
        # Survival / medical
        "survived", "diagnosis", "readmitted",
        # HR / churn
        "attrition", "churn", "left", "turnover",
        # Binary classification
        "is_fraud", "fraud", "spam", "malignant",
    ]
    # Case-insensitive match
    col_lower_map = {col.lower().strip(): col for col in df.columns}
    for target_name in common_targets:
        if target_name in col_lower_map:
            col = col_lower_map[target_name]
            logger.info(f"Auto-detected target column: '{col}'")
            return col

    # Priority 2: Find a binary column (exactly 2 unique values, 0/1 or similar)
    for col in reversed(df.columns):
        n_unique = df[col].nunique()
        if n_unique == 2:
            unique_vals = set(df[col].dropna().unique())
            # Looks like a binary target
            if unique_vals <= {0, 1} or unique_vals <= {"0", "1"} or unique_vals <= {"yes", "no"} or unique_vals <= {True, False}:
                logger.info(f"Auto-detected binary target column: '{col}'")
                return col

    # Priority 3: Last column
    last_col = df.columns[-1]
    logger.info(f"Fallback: using last column as target: '{last_col}'")
    return last_col


def _auto_detect_sensitive(
    df: pd.DataFrame,
    target: str,
    config: Dict[str, Any],
) -> List[str]:
    """Auto-detect sensitive features using heuristics across ALL columns."""
    all_cols = [c for c in df.columns if c != target]
    if not all_cols:
        return []

    # Comprehensive sensitive feature keywords
    sensitive_keywords = [
        "sex", "gender", "race", "ethnicity", "age", "nationality",
        "native_country", "marital", "religion", "color", "skin",
        "disability", "pregnant", "veteran", "citizen", "country",
        "origin", "language", "caste", "tribe",
    ]

    # Search ALL columns (not just categoricals) for keyword matches
    detected = [
        col for col in all_cols
        if any(kw in col.lower() for kw in sensitive_keywords)
    ]

    if detected:
        logger.info(f"Heuristic detected sensitive features: {detected}")
        return detected

    # Fallback: categorical columns with low cardinality (likely demographics)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != target]
    low_card = [c for c in cat_cols if df[c].nunique() <= 10]

    if low_card:
        return low_card[:3]

    # Last resort: any low-cardinality columns
    low_card_any = [c for c in all_cols if df[c].nunique() <= 10]
    if low_card_any:
        return low_card_any[:2]

    # Absolute fallback
    return all_cols[:2]


def _strip_non_serialisable(d: Dict[str, Any]) -> Dict[str, Any]:
    """Remove non-JSON-serialisable entries from a dict."""
    skip_keys = {
        "baseline_model", "best_model",
        "baseline_predictions", "best_predictions",
        "baseline_probabilities", "best_probabilities",
    }
    return {k: v for k, v in d.items() if k not in skip_keys}


def _make_serialisable(obj: Any) -> Any:
    """Convert non-serialisable objects for JSON output."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serialisable(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return {"shape": list(obj.shape), "columns": list(obj.columns)}
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def _save_artifacts(
    result: Dict[str, Any],
    dataset_output: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Save pipeline artifacts to disk."""
    output_dir = Path(config.get("output_dir", str(OUTPUT_DIR)))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON report
    report_path = output_dir / "aegis_report.json"
    try:
        serialisable = _make_serialisable(result)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2, default=str, ensure_ascii=False)
        print(f"  [SAVED] Report -> {report_path}")
    except Exception as e:
        logger.error(f"Failed to save report: {e}")

    # Save debiased dataset
    debiased = dataset_output.get("debiased_dataset")
    if debiased is not None and isinstance(debiased, pd.DataFrame):
        csv_path = output_dir / "debiased_dataset.csv"
        try:
            debiased.to_csv(csv_path, index=False, encoding="utf-8")
            print(f"  [SAVED] Debiased dataset -> {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save debiased dataset: {e}")

    # Save bias report
    bias_report = result.get("dataset_analysis", {}).get("bias_report", {})
    if bias_report:
        bias_path = output_dir / "bias_report.json"
        try:
            with open(bias_path, "w", encoding="utf-8") as f:
                json.dump(bias_report, f, indent=2, default=str, ensure_ascii=False)
            print(f"  [SAVED] Bias report -> {bias_path}")
        except Exception as e:
            logger.error(f"Failed to save bias report: {e}")
