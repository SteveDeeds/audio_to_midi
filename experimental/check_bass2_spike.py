"""Quick diagnostic of bass2 percussive_spike distribution."""
import json
from pathlib import Path
import numpy as np

path = Path("model_profiles/bass2.json")
data = json.loads(path.read_text())

pos_spikes = []
neg_spikes = []

for item in data.get("cand", []):
    row = item.get("row", [])
    if len(row) < 4:
        continue
    spike = float(row[3])
    label = int(item.get("label", 0))
    if label == 1:
        pos_spikes.append(spike)
    else:
        neg_spikes.append(spike)

pos = np.array(pos_spikes)
neg = np.array(neg_spikes)

print(f"bass2 percussive_spike distribution:")
print(f"\nPositive (keep): n={len(pos)}")
print(f"  Values: {np.min(pos):.4f} to {np.max(pos):.4f}")
print(f"  Median: {np.median(pos):.4f}")
print(f"  Mean: {np.mean(pos):.4f}")
print(f"  Std: {np.std(pos):.4f}")
print(f"  Non-zero: {np.sum(pos > 0.01)}/{len(pos)}")
print(f"  Distribution: {np.percentile(pos, [0, 25, 50, 75, 100])}")

print(f"\nNegative (drop): n={len(neg)}")
print(f"  Values: {np.min(neg):.4f} to {np.max(neg):.4f}")
print(f"  Median: {np.median(neg):.4f}")
print(f"  Mean: {np.mean(neg):.4f}")
print(f"  Std: {np.std(neg):.4f}")
print(f"  Non-zero: {np.sum(neg > 0.01)}/{len(neg)}")
print(f"  Distribution: {np.percentile(neg, [0, 25, 50, 75, 100])}")

print(f"\nOverall:")
print(f"  All non-zero (>0.01): {np.sum(np.concatenate([pos, neg]) > 0.01)}/{len(pos)+len(neg)} ({100*(np.sum(np.concatenate([pos, neg]) > 0.01)//(len(pos)+len(neg)))}%)")
