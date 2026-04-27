"""Test elaborate explanations."""
import time, json, urllib.request, urllib.parse

t = time.time()
data = urllib.parse.urlencode({"mode": "full_pipeline", "speed": "fast"}).encode()
req = urllib.request.Request("http://localhost:8000/api/run-pipeline", data=data, method="POST")
req.add_header("Content-Type", "application/x-www-form-urlencoded")

with urllib.request.urlopen(req, timeout=120) as resp:
    body = json.loads(resp.read())

elapsed = time.time() - t
meta = body.get("metadata", {})
expl = body.get("explanations", {})

print(f"Time: {elapsed:.1f}s | Strategy: {meta.get('strategy_used')}")
print(f"\n{'='*60}")
print("BIAS EXPLANATION (first 300 chars):")
print(expl.get("bias_explanation", "MISSING")[:300])
print(f"\n{'='*60}")
print("STRATEGY JUSTIFICATION (first 300 chars):")
print(expl.get("strategy_justification", "MISSING")[:300])
print(f"\n{'='*60}")
print("RECOMMENDATION (first 300 chars):")
print(expl.get("recommendation", "MISSING")[:300])
