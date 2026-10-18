"""
experimental/analyze_model_profiles.py
======================================
Analyze three model profiles (Bass, Guitar, Piano) to find:
1. Feature importance (which parameters predict keep/drop best?)
2. Cross-instrument similarities and differences
3. Weak predictors that might need removing or fixing
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = ["transient", "chroma_delta", "energy_db", "percussive_spike"]


def load_profile(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Load candidate training data from a profile."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    labels = []

    if "cand" in payload:
        for item in payload.get("cand", []):
            row = item.get("row")
            if not isinstance(row, list) or len(row) < 4:
                continue
            rows.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
            labels.append(int(item.get("label", 0)))
    else:
        # Backward compatibility
        for row, label in zip(payload.get("cand_X", []), payload.get("cand_y", [])):
            if not isinstance(row, list) or len(row) < 3:
                continue
            vals = [float(v) for v in row]
            if len(vals) >= 5:
                mapped = [vals[0], vals[1], vals[2], vals[4]]
            elif len(vals) == 4:
                mapped = [vals[0], vals[1], vals[2], 0.0]
            else:
                while len(vals) < 4:
                    vals.append(0.0)
                mapped = vals[:4]
            rows.append(mapped)
            labels.append(int(label))

    name = payload.get("name", path.stem)
    return np.array(rows, dtype=float), np.array(labels, dtype=int), str(name)


def analyze_profile(X: np.ndarray, y: np.ndarray, name: str) -> dict:
    """Analyze feature importance and stats for one profile."""
    if len(X) < 2 or len(np.unique(y)) < 2:
        return {}

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(Xs, y)

    coefs = lr.coef_[0]
    intercept = lr.intercept_[0]

    # Feature stats
    stats = {}
    for i, feat_name in enumerate(FEATURE_NAMES):
        pos_vals = X[y == 1, i]
        neg_vals = X[y == 0, i]
        stats[feat_name] = {
            "pos_median": float(np.median(pos_vals)) if len(pos_vals) > 0 else 0.0,
            "neg_median": float(np.median(neg_vals)) if len(neg_vals) > 0 else 0.0,
            "pos_std": float(np.std(pos_vals)) if len(pos_vals) > 0 else 0.0,
            "neg_std": float(np.std(neg_vals)) if len(neg_vals) > 0 else 0.0,
            "coef": float(coefs[i]),
            "coef_abs": float(np.abs(coefs[i])),
        }

    return {
        "name": name,
        "n_samples": len(X),
        "n_positive": int(np.sum(y == 1)),
        "n_negative": int(np.sum(y == 0)),
        "intercept": float(intercept),
        "coefficients": [
            {"feature": FEATURE_NAMES[i], "coef": float(coefs[i]), "abs": float(np.abs(coefs[i]))}
            for i in range(len(coefs))
        ],
        "feature_stats": stats,
    }


def print_analysis(profiles_data: list[dict]) -> None:
    """Print formatted analysis of profiles."""
    print("\n" + "=" * 100)
    print("MODEL PROFILE ANALYSIS: FEATURE IMPORTANCE & CROSS-INSTRUMENT COMPARISON")
    print("=" * 100)

    # Overview
    print("\nOVERVIEW:")
    for pdata in profiles_data:
        print(
            f"  {pdata['name']:15s}  n={pdata['n_samples']:3d}  "
            f"(positive={pdata['n_positive']:2d}, negative={pdata['n_negative']:2d})"
        )

    # Feature coefficients (sorted by absolute value)
    print("\n" + "-" * 100)
    print("FEATURE IMPORTANCE (Logistic Regression Coefficients)")
    print("-" * 100)
    print("Higher |coef| = stronger predictor. Positive coef = pushes towards 'keep note'.\n")

    for pdata in profiles_data:
        print(f"{pdata['name']}:")
        coefs = sorted(pdata["coefficients"], key=lambda x: x["abs"], reverse=True)
        for i, c in enumerate(coefs, 1):
            bar = "█" * max(1, int(c["abs"] * 20))
            print(f"  {i}. {c['feature']:18s}  coef={c['coef']:7.3f}  {bar}")
        print()

    # Feature statistics
    print("-" * 100)
    print("FEATURE DISTRIBUTIONS (by positive/negative labels)")
    print("-" * 100)
    for pdata in profiles_data:
        print(f"\n{pdata['name']}:")
        stats = pdata["feature_stats"]
        for feat, st in stats.items():
            pos_med = st["pos_median"]
            neg_med = st["neg_median"]
            pos_std = st["pos_std"]
            neg_std = st["neg_std"]
            print(
                f"  {feat:18s}  "
                f"keep:  median={pos_med:8.2f} ±{pos_std:6.2f}  |  "
                f"drop:  median={neg_med:8.2f} ±{neg_std:6.2f}  |  "
                f"Δ_median={pos_med - neg_med:7.2f}"
            )

    # Cross-profile comparison
    print("\n" + "-" * 100)
    print("CROSS-PROFILE COMPARISON (Feature Importance Ranking)")
    print("-" * 100)
    feat_rank = {feat: [] for feat in FEATURE_NAMES}
    for pdata in profiles_data:
        for i, c in enumerate(sorted(pdata["coefficients"], key=lambda x: x["abs"], reverse=True), 1):
            feat_rank[c["feature"]].append((pdata["name"], i, c["abs"]))

    for feat in FEATURE_NAMES:
        print(f"\n{feat}:")
        for name, rank, abs_coef in sorted(feat_rank[feat], key=lambda x: x[1]):
            print(f"  {name:15s}  rank={rank}  |coef|={abs_coef:.3f}")

    # Weak predictors (low absolute coefficients across all profiles)
    print("\n" + "-" * 100)
    print("WEAK PREDICTORS (Low absolute coefficients across profiles)")
    print("-" * 100)
    avg_abs_coef = {}
    for feat in FEATURE_NAMES:
        vals = [x[2] for x in feat_rank[feat]]
        avg_abs_coef[feat] = np.mean(vals)

    for feat, avg_coef in sorted(avg_abs_coef.items(), key=lambda x: x[1]):
        print(f"  {feat:18s}  avg|coef|={avg_coef:.3f}")

    weak_threshold = np.percentile(list(avg_abs_coef.values()), 33)
    weak_feats = [f for f, c in avg_abs_coef.items() if c < weak_threshold]
    if weak_feats:
        print(f"\n  → Consider removing or recalibrating: {', '.join(weak_feats)}")
        print(f"    (average |coef| < {weak_threshold:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze model profiles for feature importance")
    parser.add_argument("--profile-dir", default="model_profiles", help="Directory with profiles")
    parser.add_argument("--profiles", nargs="+", default=["bass", "Guitar", "Piano_Chords"],
                        help="Profile names to analyze (without .json)")
    args = parser.parse_args()

    profiles_dir = Path(args.profile_dir)
    profiles_data = []

    for profile_name in args.profiles:
        path = profiles_dir / f"{profile_name}.json"
        if not path.exists():
            print(f"[warn] {path} not found, skipping")
            continue
        try:
            X, y, name = load_profile(path)
            analysis = analyze_profile(X, y, name)
            if analysis:
                profiles_data.append(analysis)
        except Exception as exc:
            print(f"[error] {path}: {exc}")

    if profiles_data:
        print_analysis(profiles_data)
    else:
        raise SystemExit("No profiles loaded")


if __name__ == "__main__":
    main()
