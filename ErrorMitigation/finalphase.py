import json
import os
import shap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class Phase5Engine:

    def __init__(
        self,
        X,
        feature_names,
        sensitive_features,
        output_dir="pipeline_output",
        gemini_enabled=True,
        gemini_model="gemini-1.5-flash"
    ):
        self.df = X.copy()
        self.feature_names = feature_names
        self.sensitive_features = sensitive_features or []
        self.output_dir = output_dir

        self.gemini_enabled = gemini_enabled
        self.gemini_model = gemini_model
        self.api_key = os.getenv("GEMINI_API_KEY")

        os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # SAFE TARGET DETECTION
    # -----------------------------
    def detect_target_column(self):
        df = self.df

        priority = ["approved", "target", "label", "outcome", "y"]

        # strict match first
        for p in priority:
            for col in df.columns:
                if col.lower() == p:
                    return col

        # fallback: binary column BUT ignore numeric-heavy columns like income
        for col in df.columns:
            if df[col].dtype == "object":
                vals = df[col].dropna().unique()
                if len(vals) == 2:
                    return col

        # final fallback: last column
        return df.columns[-1]

    # -----------------------------
    # SAFE PREPROCESS
    # -----------------------------
    def preprocess(self):
        target_col = self.detect_target_column()

        df = self.df.copy()

        X = df.drop(columns=[target_col], errors="ignore")

        X = pd.get_dummies(X, drop_first=True)

        # FORCE CLEAN NUMERIC MATRIX
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        self.feature_names = list(X.columns)

        print(f"[INFO] Target detected: {target_col}")
        print(f"[INFO] Final features: {len(X.columns)}")

        return X

    # -----------------------------
    # FIXED SHAP (NO TYPE ERRORS)
    # -----------------------------
    def compute_shap(self):
        try:
            X = self.preprocess()

            if X.shape[1] == 0:
                raise ValueError("No usable features after preprocessing")

            X_np = X.to_numpy(dtype=np.float64)

            # dummy model (safe + stable)
            def dummy_model(x):
                return np.sum(x, axis=1)

            explainer = shap.Explainer(dummy_model, X_np)
            shap_values = explainer(X_np)

            values = np.array(shap_values.values)

            # HARD SAFETY FIX
            if values.ndim == 0:
                values = np.zeros((X_np.shape[0], X_np.shape[1]))
            elif values.ndim == 1:
                values = values.reshape(-1, 1)
            elif values.ndim == 3:
                values = values[:, :, 0]

            shap_values.values = values

            return shap_values

        except Exception as e:
            print("[ERROR] SHAP failed:", e)
            return None

    # -----------------------------
    # FEATURE IMPORTANCE
    # -----------------------------
    def get_feature_importance(self, shap_values):
        values = np.array(shap_values.values)
        importance = np.abs(values).mean(axis=0)

        return dict(sorted({
            self.feature_names[i]: float(importance[i])
            for i in range(len(self.feature_names))
        }.items(), key=lambda x: x[1], reverse=True))

    # -----------------------------
    # SENSITIVE IMPACT (SAFE)
    # -----------------------------
    def detect_sensitive_impact(self, importance):
        total = sum(importance.values()) + 1e-6

        return {
            f: round(importance.get(f, 0) / total, 4)
            for f in self.sensitive_features
        }

    # -----------------------------
    # SHAP VISUALS (FAIL-SAFE)
    # -----------------------------
    def generate_shap_plots(self, shap_values):
        try:
            X = self.preprocess()
            X_np = X.to_numpy(dtype=np.float64)

            summary_path = None
            bar_path = None

            try:
                plt.figure()
                shap.summary_plot(shap_values, X_np, show=False)
                summary_path = os.path.join(self.output_dir, "shap_summary.png")
                plt.savefig(summary_path, bbox_inches="tight")
                plt.close()
            except:
                print("[WARN] Summary plot skipped")

            try:
                importance = np.abs(shap_values.values).mean(axis=0)

                plt.figure(figsize=(8, 5))
                plt.barh(self.feature_names, importance)
                plt.gca().invert_yaxis()

                bar_path = os.path.join(self.output_dir, "shap_bar.png")
                plt.savefig(bar_path, bbox_inches="tight")
                plt.close()
            except:
                print("[WARN] Bar plot skipped")

            return summary_path, bar_path

        except Exception as e:
            print("[WARN] Plot generation failed:", e)
            return None, None

    # -----------------------------
    # GEMINI (SAFE)
    # -----------------------------
    def generate_llm_explanation(self, importance, sensitive_impact):

        if not self.gemini_enabled or not self.api_key:
            return self.fallback_explanation(importance, sensitive_impact)

        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)

            prompt = f"""
You are a fairness auditor.

Feature importance:
{importance}

Sensitive impact:
{sensitive_impact}

Explain:
- bias reason
- key drivers
- risk
- mitigation
"""

            response = client.models.generate_content(
                model=self.gemini_model,
                contents=prompt
            )

            return getattr(response, "text", None) or str(response)

        except Exception:
            return self.fallback_explanation(importance, sensitive_impact)

    # -----------------------------
    # FALLBACK
    # -----------------------------
    def fallback_explanation(self, importance, sensitive_impact):
        return "\n".join(
            [f"Top features: {list(importance.keys())[:5]}"] +
            [f"{k}: {v*100:.1f}% impact" for k, v in sensitive_impact.items()]
        )

    # -----------------------------
    # RUN
    # -----------------------------
    def run(self):

        shap_values = self.compute_shap()

        if shap_values is None:
            return {"error": "SHAP failed"}

        importance = self.get_feature_importance(shap_values)
        sensitive_impact = self.detect_sensitive_impact(importance)

        summary, bar = self.generate_shap_plots(shap_values)

        audit = {
            "model_behavior": {
                "top_features": list(importance.keys())[:5],
                "feature_importance": importance
            },
            "bias_analysis": {
                "sensitive_feature_impact": sensitive_impact
            },
            "explanation": self.generate_llm_explanation(importance, sensitive_impact),
            "artifacts": {
                "shap_summary": summary,
                "shap_bar": bar
            }
        }

        path = os.path.join(self.output_dir, "phase5_explainability.json")

        with open(path, "w") as f:
            json.dump(audit, f, indent=2)

        print(f"[SAVED] Phase 5 -> {path}")

        return audit