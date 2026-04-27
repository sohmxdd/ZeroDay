"""
AEGIS — Data Preprocessing
=============================

Handles missing value imputation, junk column removal, and categorical
encoding.  Produces clean, numeric data ready for bias detection and
model training.

Hardened to handle messy real-world datasets (COMPAS, German Credit,
Titanic, UCI Adult, etc.) without crashing.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ..config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Junk Column Patterns
# ---------------------------------------------------------------------------

_ID_PATTERNS = [
    "unnamed", "index", "row_id", "row_num", "rowid", "record_id",
    "_id", "serial", "sr_no", "s_no", "sno",
]

_DATE_DTYPES = ["datetime64", "datetime64[ns]", "datetimetz"]


def _is_id_column(col: str, series: pd.Series) -> bool:
    """Detect if a column is likely an ID / index column."""
    col_lower = col.lower().strip()

    # Name-based detection
    if col_lower in ("id",) or any(p in col_lower for p in _ID_PATTERNS):
        return True

    # All unique integer values that look like row indices
    if series.dtype in (np.int64, np.int32, np.float64):
        n_unique = series.nunique()
        if n_unique == len(series) and n_unique > 20:
            return True

    return False


def _is_junk_column(col: str, series: pd.Series, n_rows: int) -> bool:
    """Detect columns that should be dropped before training."""
    # All NaN
    if series.isnull().all():
        return True

    # Date/time columns
    if series.dtype.name in _DATE_DTYPES:
        return True
    if series.dtype == object:
        # Heuristic: if many values parse as dates, it's a date column
        sample = series.dropna().head(20)
        try:
            parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
            if parsed.notna().sum() > len(sample) * 0.8:
                return True
        except Exception:
            pass

    # Name-like string columns (first name, last name, full name)
    name_keywords = ["name", "first_name", "last_name", "full_name", "surname"]
    if col.lower().strip() in name_keywords:
        return True

    # Extremely high cardinality categoricals (> 50 unique for object cols)
    if series.dtype == object:
        n_unique = series.nunique()
        if n_unique > 50:
            return True

    return False


def preprocess_dataset(
    df: pd.DataFrame,
    target: Optional[str] = None,
    categorical_cols: Optional[List[str]] = None,
    numerical_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Preprocess a dataset for the AEGIS pipeline.

    Steps:
        1. Drop junk columns (IDs, dates, names, high-cardinality, all-NaN)
        2. Handle missing values (median for numeric, mode for categorical)
        3. Label-encode remaining categorical columns
        4. Final NaN safety net (fill any lingering NaN with 0)
        5. Clip Inf values

    Args:
        df: Raw input DataFrame.
        target: Optional target column name (excluded from junk detection).
        categorical_cols: Explicit list of categorical columns.
            If None, auto-detected from dtype.
        numerical_cols: Explicit list of numerical columns.
            If None, auto-detected from dtype.

    Returns:
        Tuple of (processed DataFrame, metadata dict).
    """
    logger.info("Preprocessing dataset...")
    df = df.copy()
    n_rows = len(df)
    dropped_cols: List[str] = []

    # === Step 0: Drop junk columns ===
    protect_cols = set()
    if target:
        protect_cols.add(target)

    for col in list(df.columns):
        if col in protect_cols:
            continue

        series = df[col]

        # Check if it's an ID column
        if _is_id_column(col, series):
            logger.info(f"  Dropping ID column: '{col}'")
            dropped_cols.append(col)
            df.drop(columns=[col], inplace=True)
            continue

        # Check if it's junk
        if _is_junk_column(col, series, n_rows):
            logger.info(f"  Dropping junk column: '{col}' (dtype={series.dtype}, nunique={series.nunique()})")
            dropped_cols.append(col)
            df.drop(columns=[col], inplace=True)
            continue

    # Auto-detect column types if not provided (after junk removal)
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
    else:
        numerical_cols = [c for c in numerical_cols if c in df.columns]

    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    else:
        categorical_cols = [c for c in categorical_cols if c in df.columns]

    # === Step 1: Handle Missing Values ===
    imputation_info = {}

    for col in numerical_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0
            df[col] = df[col].fillna(median_val)
            imputation_info[col] = {"method": "median", "value": float(median_val)}

    for col in categorical_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            mode_vals = df[col].mode()
            mode_val = mode_vals[0] if len(mode_vals) > 0 else "unknown"
            df[col] = df[col].fillna(mode_val)
            imputation_info[col] = {"method": "mode", "value": str(mode_val)}

    # === Step 2: Encode Categorical Columns ===
    encoders = {}
    encoded_cols = []

    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            encoded_cols.append(col)

    # === Step 3: Final NaN Safety Net ===
    # After all processing, fill any remaining NaN with column median or 0
    remaining_nan = df.isnull().sum().sum()
    if remaining_nan > 0:
        logger.warning(f"  Safety net: filling {remaining_nan} remaining NaN values")
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in (np.float64, np.float32, np.int64, np.int32):
                    fill_val = df[col].median()
                    if pd.isna(fill_val):
                        fill_val = 0
                    df[col] = df[col].fillna(fill_val)
                else:
                    df[col] = df[col].fillna(0)

    # === Step 4: Clip Inf values ===
    numeric_cols_final = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols_final:
        if np.isinf(df[col]).any():
            logger.warning(f"  Clipping Inf values in '{col}'")
            col_finite = df[col].replace([np.inf, -np.inf], np.nan)
            col_max = col_finite.max() if col_finite.notna().any() else 0
            col_min = col_finite.min() if col_finite.notna().any() else 0
            df[col] = df[col].replace(np.inf, col_max).replace(-np.inf, col_min)

    metadata = {
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
        "encoded_columns": encoded_cols,
        "encoders": encoders,
        "imputation": imputation_info,
        "dropped_columns": dropped_cols,
        "total_rows": len(df),
        "total_columns": len(df.columns),
    }

    logger.info(
        f"Preprocessing complete: {len(df)} rows, "
        f"{len(encoded_cols)} columns encoded, "
        f"{len(imputation_info)} columns imputed, "
        f"{len(dropped_cols)} junk columns dropped"
    )

    return df, metadata
