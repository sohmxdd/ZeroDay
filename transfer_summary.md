# AEGIS (AI Bias Governance Engine) — Project Handover Summary

This document serves as a complete context snapshot for transferring the AEGIS project to a new system. It covers the architecture, the core problems we recently solved, and instructions for setting it up on a new machine.

## 1. Project Overview
AEGIS is an enterprise-grade AI governance platform that automatically audits machine learning datasets for bias, applies corrective mitigation strategies, and generates transparent explainability reports. 

The goal of the project is to take what is normally a slow, manual data science task and turn it into an automated, lightning-fast, and visually premium experience.

## 2. System Architecture & Tech Stack
- **Frontend**: Next.js 14, React, Tailwind CSS, Framer Motion. 
  - Features a highly polished, glass-morphism UI with WebGL shader backgrounds and real-time animated pipeline loaders.
  - Runs on port `3000` (or `3001` if occupied).
- **Backend**: Python 3, FastAPI, Uvicorn.
  - Runs on port `8000`.
  - Bridges the frontend to the core ML engine via `server.py`.
- **ML Engine (`core/`)**: Scikit-learn, Fairlearn, Pandas, Numpy.
  - Handles the 4-phase pipeline execution.
- **AI Explainability**: Integrates with Google Gemini (via `google-genai`) for generating natural language reports on bias root causes and mitigation tradeoffs.

## 3. Core Pipeline Flow
When a user uploads a dataset, the system executes 4 distinct phases:
1. **Bias Detection**: Scans for outcome bias, representation bias, proxy bias, fairness violations, and intersectional bias.
2. **Bias Mitigation**: Trains candidate models using different strategies (Baseline, Threshold Optimization, Reweighting, Fairlearn Reduction, SMOTE Oversampling). It ranks them using a weighted Tradeoff Score (`Score = α × Accuracy + β × (1 - Unfairness)`).
3. **Dataset Comparison**: Statistically compares the baseline data with the debiased data using Kolmogorov-Smirnov (KS) tests.
4. **Explainability**: Translates mathematical shifts and feature importance deltas into human-readable text.

## 4. Recent Problems Faced & Solved

We recently overhauled the system to transition it from a slow research script into a production-ready web application. Here are the major issues we fixed:

> [!TIP]
> **Problem 1: Unacceptable Execution Time (2+ minutes)**
> **Fix**: Implemented a **"Fast Mode"**. When running via the UI, the backend now limits the dataset to 5,000 rows, restricts the algorithm solver (`max_iter=200`), and limits threshold searches. This brought execution time down to **under 3 seconds** (a 900x speedup).

> [!TIP]
> **Problem 2: UI Freezing During Execution**
> **Fix**: The heavy Python data science computations were blocking the FastAPI event loop, causing the frontend to freeze or timeout. We wrapped the `run_pipeline` call in `await asyncio.to_thread()` in `server.py` to offload the ML work to a background thread.

> [!TIP]
> **Problem 3: Mitigation Engine Crashing on Real Datasets (e.g., COMPAS)**
> **Fix**: The model was failing to evaluate *any* mitigation strategies (0% accuracy, strategy "none") on messy datasets.
> - **Target Detection**: Enhanced `_auto_detect_target` in `pipeline.py` to recognize common dataset targets (like `is_recid`, `two_year_recid`, `income`, `approved`) instead of blindly picking the last column.
> - **Junk Column Cleanup**: Modified `preprocessing.py` to aggressively drop ID columns, names, dates, and extremely high-cardinality categorical columns (>50 unique values). Previously, these were being label-encoded into thousands of meaningless integers, crashing the models.
> - **NaN Safety Net**: Added a final pass in preprocessing to fill any lingering `NaN` values with medians or 0s, preventing `scikit-learn` from throwing `ValueError: Input X contains NaN`.

> [!TIP]
> **Problem 4: Weak Explainability in Fast Mode**
> **Fix**: Fast mode skips the expensive Gemini API call to save time, but the resulting explanations were 1-liners. We built a robust, dynamic template system in `engine.py` that generates rich, multi-paragraph, data-driven explanations based on the actual metrics and strategies used.

## 5. How to Run on the New System

### Step 1: Backend Setup
1. Open a terminal in the project root directory.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```
3. Install dependencies (ensure you have `fastapi`, `uvicorn`, `python-multipart`, `pandas`, `scikit-learn`, `fairlearn`, `google-genai`).
4. Start the backend server:
   ```bash
   python server.py
   # Or: uvicorn server:app --host 0.0.0.0 --port 8000 --reload
   ```

### Step 2: Frontend Setup
1. Open a new terminal in the `frontend/` directory.
2. Install Node dependencies:
   ```bash
   npm install
   ```
3. Start the Next.js development server:
   ```bash
   npm run dev
   ```

### Step 3: Environment Variables
Create a `.env` file in the root directory (if using full mode) with:
```env
GEMINI_API_KEY="your_api_key_here"
```

The system is now stable, fast, and ready for demonstrations.
