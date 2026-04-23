'''
AEGIS -- Model Evaluator
==========================

This combines the results of phase 1 and phase 2 to provide a comprehensive 
evaluation of the bias mitigation efforts. 
It compares the fairness metrics before and after mitigation, 
computes a bias severity score, ranks the features by their bias levels, 
and generates insights and a final verdict on the effectiveness of the mitigation.

'''

import json
import os
import matplotlib.pyplot as plt


class Phase3Engine:

    def __init__(self, phase1_path, before_path, after_path):
        self.phase1 = self.load_json(phase1_path)
        self.before = self.load_json(before_path)
        self.after = self.load_json(after_path)

        self.dataset_info = self.phase1.get("phase_0", {}).get("dataset_info", {})
        self.features = self.get_features()
        self.llm_explanation = self.phase1.get("phase_2", {}).get("llm_explanation", "")

    # -----------------------------
    # LOAD
    # -----------------------------
    def load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def get_features(self):
        try:
            return self.phase1["phase_2"]["sensitive_features"]["detected_features"]
        except:
            return list(self.before.get("fairness_metrics", {}).keys())

    # -----------------------------
    # FAIRNESS EXTRACTION
    # -----------------------------
    def get_fairness(self, data):
        if "after_fairness_metrics" in data:
            return data["after_fairness_metrics"]

        if "fairness_metrics" in data:
            return data["fairness_metrics"]

        return {}

    # -----------------------------
    # FAIRNESS COMPARISON + IMPROVEMENT %
    # -----------------------------
    def compare_fairness(self):
        results = {}

        before_f = self.get_fairness(self.before)
        after_f = self.get_fairness(self.after)

        for f in self.features:
            if f not in before_f:
                continue

            b = before_f[f]
            a = after_f.get(f, b)

            dp_b = b.get("demographic_parity_difference", 0)
            dp_a = a.get("demographic_parity_difference", dp_b)

            di_b = b.get("disparate_impact_ratio", 0)
            di_a = a.get("disparate_impact_ratio", di_b)

            # % improvement
            spd_improvement = ((abs(dp_b) - abs(dp_a)) / (abs(dp_b) + 1e-6)) * 100
            di_improvement = ((abs(1 - di_b) - abs(1 - di_a)) / (abs(1 - di_b) + 1e-6)) * 100

            results[f] = {
                "SPD": {
                    "before": dp_b,
                    "after": dp_a,
                    "improved": abs(dp_a) < abs(dp_b),
                    "improvement_percent": round(spd_improvement, 2)
                },
                "DI": {
                    "before": di_b,
                    "after": di_a,
                    "improved": abs(1 - di_a) < abs(1 - di_b),
                    "improvement_percent": round(di_improvement, 2)
                }
            }

        return results

    # -----------------------------
    # BIAS SEVERITY SCORE
    # -----------------------------
    def compute_severity(self, fairness):
        severity = {}

        for f, metrics in fairness.items():
            spd = abs(metrics["SPD"]["after"])
            di = abs(1 - metrics["DI"]["after"])

            score = (spd + di) / 2  # simple combined metric

            severity[f] = {
                "severity_score": round(score, 4),
                "level": (
                    "LOW" if score < 0.1 else
                    "MEDIUM" if score < 0.25 else
                    "HIGH"
                )
            }

        return severity

    # -----------------------------
    # FEATURE RANKING
    # -----------------------------
    def rank_features(self, severity):
        return sorted(
            severity.items(),
            key=lambda x: x[1]["severity_score"],
            reverse=True
        )

    # -----------------------------
    # DISTRIBUTION
    # -----------------------------
    def compare_distribution(self):
        results = {}
        before_d = self.before.get("distribution_bias", {})
        after_d = self.after.get("distribution_bias", {})

        for f in self.features:
            if f not in before_d:
                continue

            b = before_d[f].get("imbalance_ratio", 0)
            a = after_d.get(f, {}).get("imbalance_ratio", b)

            results[f] = {
                "before": b,
                "after": a,
                "improved": a < b
            }

        return results

    # -----------------------------
    # OUTCOME
    # -----------------------------
    def compare_outcomes(self):
        results = {}
        before_o = self.before.get("outcome_bias", {})
        after_o = self.after.get("outcome_bias", {})

        for f in self.features:
            if f not in before_o:
                continue

            b = before_o[f].get("disparity", 0)
            a = after_o.get(f, {}).get("disparity", b)

            results[f] = {
                "before": b,
                "after": a,
                "improved": a < b
            }

        return results

    # -----------------------------
    # INSIGHTS (SMART)
    # -----------------------------
    def generate_insights(self, fairness, severity):
        insights = []

        for f in fairness:
            spd_imp = fairness[f]["SPD"]["improvement_percent"]
            di_imp = fairness[f]["DI"]["improvement_percent"]
            level = severity[f]["level"]

            if spd_imp > 0:
                insights.append(f"{f}: SPD improved by {spd_imp}%")
            else:
                insights.append(f"{f}: SPD worsened by {abs(spd_imp)}%")

            if di_imp > 0:
                insights.append(f"{f}: DI improved by {di_imp}%")

            insights.append(f"{f}: Bias severity = {level}")

        return insights

    # -----------------------------
    # VERDICT
    # -----------------------------
    def generate_verdict(self, fairness):
        improved = sum(1 for f in fairness if fairness[f]["SPD"]["improved"])
        total = len(fairness)

        if total == 0:
            return "No comparable features"

        ratio = improved / total

        if ratio > 0.8:
            return "Strong bias reduction"
        elif ratio > 0.4:
            return "Moderate improvement"
        elif ratio > 0:
            return "Minor improvement"
        else:
            return "No bias reduction"

    # -----------------------------
    # PLOTS (ENHANCED)
    # -----------------------------
    def plot_spd(self):
        fairness = self.compare_fairness()
        features = list(fairness.keys())

        if not features:
            print("No SPD data")
            return

        before = [fairness[f]["SPD"]["before"] for f in features]
        after = [fairness[f]["SPD"]["after"] for f in features]

        x = range(len(features))

        plt.figure(figsize=(10, 5))
        plt.bar(x, before, width=0.4, label="Before")
        plt.bar([i + 0.4 for i in x], after, width=0.4, label="After")

        plt.xticks([i + 0.2 for i in x], features)
        plt.axhline(0, linestyle="--")
        plt.title("SPD (Lower = Better)")
        plt.legend()

        os.makedirs("pipeline_output", exist_ok=True)
        plt.savefig("pipeline_output/spd_comparison.png")
        plt.close()

    def plot_di(self):
        fairness = self.compare_fairness()
        features = list(fairness.keys())

        if not features:
            print("No DI data")
            return

        before = [fairness[f]["DI"]["before"] for f in features]
        after = [fairness[f]["DI"]["after"] for f in features]

        x = range(len(features))

        plt.figure(figsize=(10, 5))
        plt.bar(x, before, width=0.4, label="Before")
        plt.bar([i + 0.4 for i in x], after, width=0.4, label="After")

        plt.xticks([i + 0.2 for i in x], features)
        plt.axhline(1, linestyle="--")
        plt.axhline(0.8, linestyle=":")
        plt.title("DI (Closer to 1 = Better)")
        plt.legend()

        os.makedirs("pipeline_output", exist_ok=True)
        plt.savefig("pipeline_output/di_comparison.png")
        plt.close()

    # -----------------------------
    # FINAL RUN
    # -----------------------------
    def run(self):
        fairness = self.compare_fairness()
        dist = self.compare_distribution()
        outcome = self.compare_outcomes()

        severity = self.compute_severity(fairness)
        ranking = self.rank_features(severity)
        insights = self.generate_insights(fairness, severity)
        verdict = self.generate_verdict(fairness)

        return {
            "dataset_summary": {
                "rows": self.dataset_info.get("total_rows"),
                "columns": self.dataset_info.get("total_columns"),
                "target": self.dataset_info.get("target_column"),
                "sensitive_features": self.features
            },
            "fairness_comparison": fairness,
            "bias_severity": severity,
            "feature_ranking": ranking,
            "distribution_comparison": dist,
            "outcome_comparison": outcome,
            "insights": insights,
            "initial_explanation": self.llm_explanation,
            "final_verdict": verdict
        }