"""
AEGIS -- Model Trainer
=======================

Trains separate models for every candidate mitigation pipeline.

Supported model types:
    - Logistic Regression  (baseline, always available)
    - Random Forest
    - XGBoost (optional -- graceful fallback if not installed)

Each pipeline receives its own training run using the transformed data
(and optional sample weights) produced by the strategies module.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score

from .strategies import execute_strategy, MitigationResult
from .utils import get_config, get_logger, safe_import

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

TrainedModel = Dict[str, Any]
# {
#     "pipeline": str,
#     "model_type": str,
#     "model": <fitted sklearn estimator>,
#     "X_train": pd.DataFrame,
#     "X_test": pd.DataFrame,
#     "y_train": pd.Series,
#     "y_test": pd.Series,
#     "sensitive_train": pd.Series,
#     "sensitive_test": pd.Series,
#     "predictions": np.ndarray,
#     "probabilities": np.ndarray,
#     "mitigation_result": MitigationResult,
# }

TrainingOutput = Dict[str, TrainedModel]


# ---------------------------------------------------------------------------
# Problem Type & Cleaning
# ---------------------------------------------------------------------------

def _detect_problem_type(y: pd.Series) -> str:
    """Detect if the target variable is for classification or regression."""
    if y.dtype.kind in "iub" and len(np.unique(y)) <= 20:
        return "classification"
    if y.dtype.kind in "fc":
        # If float but only few unique values, could be classification
        if len(np.unique(y)) <= 10:
            return "classification"
        return "regression"
    if y.dtype.kind in "OUS": # Objects/Strings
        return "classification"
    
    # Generic fallback based on unique ratio
    unique_ratio = len(np.unique(y)) / len(y)
    if unique_ratio > 0.05 and len(np.unique(y)) > 20:
        return "regression"
    return "classification"


def _clean_target(y: pd.Series, problem_type: str) -> Tuple[pd.Series, np.ndarray]:
    """Clean the target variable by handling NaNs and rare classes."""
    # 1. Handle NaNs
    valid_mask = ~y.isna()
    y_clean = y[valid_mask]
    
    if problem_type == "classification":
        # 2. Handle rare classes (min 2 members required for stratify/val)
        counts = y_clean.value_counts()
        rare_classes = counts[counts < 2].index
        if len(rare_classes) > 0:
            logger.warning(f"Found {len(rare_classes)} rare classes with < 2 samples. Merging into 'Other' or removing.")
            # If string-like, merge
            if y_clean.dtype.kind in "OUS":
                y_clean = y_clean.mask(y_clean.isin(rare_classes), "Other")
            else:
                # If numeric, and too many rare classes, maybe it should be regression
                # For now just remove them to prevent crash
                y_clean = y_clean[~y_clean.isin(rare_classes)]
        
        # Final check: if only 1 class remains, we can't train
        if len(np.unique(y_clean)) < 2:
            logger.error("Only one unique class remains after cleaning. Training impossible.")
            return y_clean, np.array([]) # Will be handled by valid_mask
            
    # Re-calculate final mask after possible filtering
    final_mask = y.index.isin(y_clean.index)
    return y_clean, final_mask


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

def _get_model_instance(
    model_type: str,
    problem_type: str,
    config: Dict[str, Any],
) -> Any:
    """
    Instantiate a model by name and problem type (Classification vs Regression).
    """
    rs = config.get("random_state", 42)

    if problem_type == "regression":
        if model_type == "logistic_regression": # Map to Linear
            return LinearRegression()
        if model_type == "random_forest":
            return RandomForestRegressor(n_estimators=100, max_depth=10, random_state=rs)
        if model_type == "xgboost":
            xgb = safe_import("xgboost")
            if xgb is not None:
                return xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=rs)
            return RandomForestRegressor(n_estimators=100, max_depth=10, random_state=rs)
    else:
        # Classification
        if model_type == "logistic_regression":
            return LogisticRegression(max_iter=200, random_state=rs, solver="lbfgs")
        if model_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=rs, n_jobs=-1)
        if model_type == "xgboost":
            xgb = safe_import("xgboost")
            if xgb is not None:
                return xgb.XGBClassifier(
                    n_estimators=100, max_depth=6, random_state=rs,
                    use_label_encoder=False, eval_metric="logloss"
                )
            return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=rs, n_jobs=-1)

    raise ValueError(f"Unknown model type: {model_type} for problem: {problem_type}")


# ---------------------------------------------------------------------------
# Encoding Helper
# ---------------------------------------------------------------------------

def _encode_for_training(
    X: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Encode categorical columns for sklearn models."""
    X_enc = X.copy()
    encoders: Dict[str, LabelEncoder] = {}
    for col in X_enc.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X_enc[col] = le.fit_transform(X_enc[col].astype(str))
        encoders[col] = le
    return X_enc, encoders


# ---------------------------------------------------------------------------
# Single Pipeline Trainer
# ---------------------------------------------------------------------------

def _train_single_pipeline(
    pipeline_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    sensitive: pd.Series,
    model_type: str = "logistic_regression",
    config: Optional[Dict[str, Any]] = None,
) -> TrainedModel:
    """
    Execute a single mitigation pipeline and train a model on the result.
    """
    cfg = get_config(config)
    logger.info(f"Training pipeline '{pipeline_name}' with model '{model_type}' ...")

    # 1. Detect and Clean target
    problem_type = _detect_problem_type(y)
    y_clean, valid_mask = _clean_target(y, problem_type)
    
    if len(y_clean) == 0:
        raise ValueError("No valid samples left after cleaning target variable.")

    X_clean = X[valid_mask]
    sensitive_clean = sensitive[valid_mask]

    # --- Apply mitigation ---
    if pipeline_name == "baseline":
        mitigation_result: MitigationResult = {
            "X": X_clean.copy(),
            "y": y_clean.copy(),
            "sensitive": sensitive_clean.copy(),
            "sample_weight": None,
            "model": None,
            "thresholds": None,
            "metadata": {"technique": "baseline"},
        }
    else:
        mitigation_result = execute_strategy(
            pipeline_name, X_clean, y_clean, sensitive_clean, config=cfg
        )

    mit_X = mitigation_result["X"]
    mit_y = mitigation_result["y"]
    mit_sensitive = mitigation_result.get("sensitive")
    sample_weight = mitigation_result.get("sample_weight")
    pre_fitted_model = mitigation_result.get("model")
    thresholds = mitigation_result.get("thresholds")

    # --- Encode ---
    mit_X_enc, encoders = _encode_for_training(mit_X)

    # Alignment
    mit_X_enc = mit_X_enc.reset_index(drop=True)
    mit_y = mit_y.reset_index(drop=True)
    sensitive_aligned = (mit_sensitive if mit_sensitive is not None else sensitive_clean).reset_index(drop=True)

    if sample_weight is not None:
        sw_series = pd.Series(sample_weight).reset_index(drop=True)
    else:
        sw_series = None

    # --- Train/test split ---
    # Robust Stratification logic
    should_stratify = False
    if problem_type == "classification":
        counts = mit_y.value_counts()
        if len(counts) > 1 and counts.min() >= 2:
            should_stratify = True
        else:
            logger.warning("Stratification disabled: some classes have too few samples.")

    split_kwargs = {
        "test_size": cfg["test_size"],
        "random_state": cfg["random_state"],
        "stratify": mit_y if should_stratify else None
    }

    if sw_series is not None:
        X_train, X_test, y_train, y_test, s_train, s_test, sw_train, sw_test = train_test_split(
            mit_X_enc, mit_y, sensitive_aligned, sw_series, **split_kwargs
        )
    else:
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            mit_X_enc, mit_y, sensitive_aligned, **split_kwargs
        )
        sw_train = None

    # --- Fit model ---
    if pre_fitted_model is not None:
        model = pre_fitted_model
    else:
        model = _get_model_instance(model_type, problem_type, cfg)
        fit_kwargs: Dict[str, Any] = {}
        if sw_train is not None:
            fit_kwargs["sample_weight"] = sw_train.values
        model.fit(X_train, y_train, **fit_kwargs)

    # --- Predict ---
    if problem_type == "classification":
        if thresholds is not None:
            probabilities = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test).astype(float)
            predictions = np.zeros(len(X_test), dtype=int)
            for group, thresh in thresholds.items():
                mask = s_test == group
                if mask.any():
                    predictions[mask.values] = (probabilities[mask.values] >= thresh).astype(int)
            unknown_mask = ~s_test.isin(thresholds.keys())
            if unknown_mask.any():
                predictions[unknown_mask.values] = (probabilities[unknown_mask.values] >= 0.5).astype(int)
        else:
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else predictions.astype(float)
    else:
        # Regression
        predictions = model.predict(X_test)
        probabilities = predictions # No real probabilities in regression

    logger.info(f"Pipeline '{pipeline_name}' training complete (Type: {problem_type}).")

    return {
        "pipeline": pipeline_name,
        "model_type": model_type,
        "model": model,
        "problem_type": problem_type, 
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "sensitive_train": s_train,
        "sensitive_test": s_test,
        "predictions": predictions,
        "probabilities": probabilities,
        "mitigation_result": mitigation_result,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_models(
    candidate_pipelines: List[str],
    X: pd.DataFrame,
    y: pd.Series,
    sensitive: pd.Series,
    model_type: str = "logistic_regression",
    config: Optional[Dict[str, Any]] = None,
) -> TrainingOutput:
    """
    Train a model for every candidate mitigation pipeline.

    Args:
        candidate_pipelines: List of pipeline names from the generator.
        X: Original feature matrix.
        y: Original target vector.
        sensitive: Primary sensitive feature Series.
        model_type: Model type to use for all pipelines.
        config: Configuration overrides.

    Returns:
        Dictionary mapping pipeline names to TrainedModel dicts.
    """
    cfg = get_config(config)
    logger.info(
        f"Training {len(candidate_pipelines)} pipelines with '{model_type}' ..."
    )

    results: TrainingOutput = {}

    for pipeline in candidate_pipelines:
        try:
            trained = _train_single_pipeline(
                pipeline_name=pipeline,
                X=X,
                y=y,
                sensitive=sensitive,
                model_type=model_type,
                config=cfg,
            )
            results[pipeline] = trained
        except Exception as e:
            logger.error(f"Pipeline '{pipeline}' failed: {e}", exc_info=True)
            continue

    logger.info(
        f"Successfully trained {len(results)}/{len(candidate_pipelines)} pipelines."
    )
    return results
