"""
AEGIS — FastAPI Backend Server
=================================

HTTP API wrapper around the AEGIS pipeline.

Usage::

    # Start server:
    python server.py

    # Or with uvicorn directly:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import io
import json
import traceback
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core import run_pipeline

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AEGIS API",
    description="AI Bias Governance Engine — Pipeline API",
    version="1.0.0",
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Custom JSON encoder for numpy/pandas types
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return {"shape": list(obj.shape), "columns": list(obj.columns)}
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def make_serializable(obj):
    """Deep-convert all numpy/pandas types to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
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


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok", "engine": "AEGIS"}


# ---------------------------------------------------------------------------
# Run Pipeline
# ---------------------------------------------------------------------------

@app.post("/api/run-pipeline")
async def api_run_pipeline(
    dataset: Optional[UploadFile] = File(None),
    mode: str = Form("full_pipeline"),
    speed: str = Form("fast"),
    model: Optional[UploadFile] = File(None),
    model_type: str = Form("logistic_regression"),
    alpha: float = Form(0.6),
    beta: float = Form(0.4),
):
    """
    Execute the AEGIS bias analysis pipeline.

    - **dataset**: CSV file (optional — uses synthetic data if omitted)
    - **mode**: ``analysis`` | ``train`` | ``full_pipeline``
    - **speed**: ``fast`` (default) | ``full``
    - **model**: Optional pre-trained model (.pkl)
    - **model_type**: ``logistic_regression`` | ``random_forest`` | ``xgboost``
    - **alpha**: Accuracy weight (0-1)
    - **beta**: Fairness weight (0-1)
    """
    try:
        # --- Parse dataset ---
        df = None
        if dataset and dataset.filename:
            print(f"\n{'='*60}")
            print(f"  [API] Received dataset: {dataset.filename}")
            contents = await dataset.read()
            df = pd.read_csv(io.BytesIO(contents))
            print(f"  [API] Dataset shape: {df.shape}")
            print(f"  [API] Columns: {list(df.columns)}")
            print(f"  [API] Mode: {mode} | Speed: {speed}")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print(f"  [API] No dataset uploaded — using synthetic/default data")
            print(f"  [API] Mode: {mode} | Speed: {speed}")
            print(f"{'='*60}\n")

        # --- Parse model ---
        user_model = None
        if model and model.filename:
            print(f"  [API] Received model: {model.filename}")

        # --- Build input ---
        input_data = {
            "dataset": df,  # None = synthetic/default
            "mode": mode,
            "speed": speed,  # "fast" (default) or "full"
            "model": user_model,
            "target": None,  # auto-detect
            "sensitive_features": None,  # auto-detect
            "config": {
                "alpha": alpha,
                "beta": beta,
                "model_type": model_type,
                "gemini_enabled": True,
            },
        }

        # --- Execute pipeline in thread (prevents UI freeze) ---
        result = await asyncio.to_thread(run_pipeline, input_data)

        # --- Serialize and return ---
        serialized = make_serializable(result)
        return JSONResponse(content=serialized)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "type": type(e).__name__,
            },
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    print("Starting AEGIS API server on http://localhost:8000")
    print("Docs available at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
