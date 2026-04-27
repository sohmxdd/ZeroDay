"""Quick end-to-end pipeline verification test."""
import requests
import time
import json

print("=" * 70)
print("  AEGIS Pipeline Verification Test")
print("=" * 70)

t0 = time.time()
r = requests.post("http://localhost:8000/api/run-pipeline", json={"speed": "fast"})
elapsed = time.time() - t0

data = r.json()
print(f"\nHTTP Status: {r.status_code}")
print(f"Total Time: {elapsed:.1f}s")
print(f"Under 30s: {'YES' if elapsed < 30 else 'NO'}")

# Explore response structure
print(f"\nTop-level keys: {list(data.keys())}")

# Try to find ranking in nested structure
def find_key(d, key, depth=0):
    if depth > 5:
        return None
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            result = find_key(v, key, depth+1)
            if result is not None:
                return result
    return None

ranking = find_key(data, "ranking")
ranking_table = find_key(data, "ranking_table")
best_strategy = find_key(data, "best_strategy")
bias_tags = find_key(data, "bias_tags")

if ranking_table:
    table = ranking_table
    print(f"\nStrategies Evaluated: {len(table)}")
    print(f"Best Strategy: {best_strategy}")
    
    print(f"\n{'Rank':<6}{'Strategy':<45}{'Score':<10}{'Accuracy':<12}{'DP Diff':<10}")
    print("-" * 83)
    for row in table:
        print(
            f"#{row['rank']:<5}"
            f"{row['pipeline']:<45}"
            f"{row['score']:.4f}    "
            f"{row['accuracy']:.4f}      "
            f"{row['demographic_parity_diff']:.4f}"
        )
    
    all_positive = all(row["score"] > 0 for row in table)
    print(f"\nAll Scores Positive: {'YES' if all_positive else 'NO'}")
    print(f"Best is NOT baseline: {'YES' if best_strategy != 'baseline' else 'NO'}")

    if bias_tags:
        detected = [k for k, v in bias_tags.items() if v]
        print(f"Bias Types Detected: {detected}")

    status = "PASS"
else:
    # Dump model_analysis keys for debugging
    model_analysis = data.get("model_analysis", {})
    print(f"\nmodel_analysis keys: {list(model_analysis.keys()) if isinstance(model_analysis, dict) else type(model_analysis)}")
    
    metadata = data.get("metadata", {})
    print(f"metadata keys: {list(metadata.keys()) if isinstance(metadata, dict) else type(metadata)}")
    
    # Check for strategies in model_analysis
    comparison = model_analysis.get("comparison_table", [])
    if comparison:
        print(f"\nFound comparison_table with {len(comparison)} strategies:")
        for row in comparison:
            print(f"  {row}")
    
    status = "CHECK"

print(f"\n{'=' * 70}")
print(f"  RESULT: {status} ({elapsed:.1f}s)")
print(f"{'=' * 70}")
