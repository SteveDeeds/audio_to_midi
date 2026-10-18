"""
Analyze pitch fingerprint quality across saved model profiles.

Focuses on:
- per-profile pitch class coverage
- leave-one-out 1-NN (cosine) accuracy
- same-note vs different-note cosine similarity
- cross-profile transfer (train on one profile, test on another)
- most/least discriminative fingerprint bins
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.metrics.pairwise import cosine_similarity


CQT_FMIN_MIDI = 12  # C0 in this project


def _load_pitch_rows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    X = []
    y = []
    for item in data.get("pitch", []):
        vec = item.get("vec", [])
        notes = item.get("notes", [])
        if not vec or not notes:
            continue
        # Use root/lowest note for label consistency with monophonic matching behavior.
        label = int(sorted(set(int(n) for n in notes))[0])
        X.append(np.array(vec, dtype=np.float32))
        y.append(label)
    if not X:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=np.int32)
    X_arr = np.vstack(X)
    # L2-normalize to match CorrectionModel.pitch_suggestions behavior.
    norms = np.linalg.norm(X_arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X_arr = X_arr / norms
    return X_arr, np.array(y, dtype=np.int32)


def _loo_1nn_accuracy(X: np.ndarray, y: np.ndarray) -> float:
    n = len(y)
    if n < 2:
        return 0.0
    sims = cosine_similarity(X)
    np.fill_diagonal(sims, -np.inf)
    nn_idx = np.argmax(sims, axis=1)
    pred = y[nn_idx]
    return float(np.mean(pred == y))


def _same_diff_similarity(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(y) < 2:
        return 0.0, 0.0
    sims = cosine_similarity(X)
    iu = np.triu_indices(len(y), k=1)
    pair_sims = sims[iu]
    yi = y[iu[0]]
    yj = y[iu[1]]
    same_mask = yi == yj
    diff_mask = ~same_mask
    same_mean = float(np.mean(pair_sims[same_mask])) if np.any(same_mask) else 0.0
    diff_mean = float(np.mean(pair_sims[diff_mask])) if np.any(diff_mask) else 0.0
    return same_mean, diff_mean


def _top_discriminative_bins(X: np.ndarray, y: np.ndarray, k: int = 8):
    if len(np.unique(y)) < 2 or X.shape[0] < 4:
        return [], []
    scores, _ = f_classif(X, y)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(scores)
    top = order[::-1][:k]
    weak = order[:k]

    def _fmt(idx: int) -> dict:
        midi = CQT_FMIN_MIDI + int(idx)
        return {"bin": int(idx), "midi": int(midi), "score": float(scores[idx])}

    return [_fmt(i) for i in top], [_fmt(i) for i in weak]


def _cross_profile_accuracy(X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray) -> tuple[float, int]:
    if len(y_train) == 0 or len(y_test) == 0:
        return 0.0, 0
    # Only evaluate overlapping labels for fair transfer reporting.
    overlap = set(y_train.tolist()) & set(y_test.tolist())
    if not overlap:
        return 0.0, 0
    mask = np.array([yy in overlap for yy in y_test], dtype=bool)
    Xq = X_test[mask]
    yq = y_test[mask]
    sims = cosine_similarity(Xq, X_train)
    nn = np.argmax(sims, axis=1)
    pred = y_train[nn]
    return float(np.mean(pred == yq)), int(len(yq))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze pitch profile quality")
    parser.add_argument("--profiles-dir", default="model_profiles")
    parser.add_argument("--profiles", nargs="+", default=["bass", "bass2", "bass3", "bass4", "Guitar", "Piano_Chords"])
    args = parser.parse_args()

    profiles_dir = Path(args.profiles_dir)
    payload = {}

    for name in args.profiles:
        path = profiles_dir / f"{name}.json"
        if not path.exists():
            print(f"[warn] missing: {path}")
            continue
        X, y = _load_pitch_rows(path)
        payload[name] = (X, y)

    print("\n" + "=" * 88)
    print("PITCH PROFILE ANALYSIS")
    print("=" * 88)

    print("\nData availability:")
    for name, (X, y) in payload.items():
        uniq = len(np.unique(y)) if len(y) else 0
        print(f"  {name:12s} rows={len(y):3d} unique_notes={uniq:2d}")

    print("\nPer-profile quality:")
    for name, (X, y) in payload.items():
        if len(y) == 0:
            print(f"\n{name}: no pitch rows")
            continue
        counts = Counter(y.tolist())
        same_sim, diff_sim = _same_diff_similarity(X, y)
        loo = _loo_1nn_accuracy(X, y)
        margin = same_sim - diff_sim
        print(f"\n{name}:")
        print(f"  samples={len(y)}, unique_notes={len(counts)}, loo_1nn_acc={loo:.3f}")
        print(f"  mean cosine same-note={same_sim:.4f}, diff-note={diff_sim:.4f}, margin={margin:.4f}")
        print("  note counts:", dict(sorted(counts.items())))

        top, weak = _top_discriminative_bins(X, y, k=8)
        if top:
            print("  top discriminative bins (bin->midi:score):")
            print("   ", ", ".join([f"{d['bin']}->{d['midi']}:{d['score']:.2f}" for d in top]))
            print("  weakest bins (bin->midi:score):")
            print("   ", ", ".join([f"{d['bin']}->{d['midi']}:{d['score']:.2f}" for d in weak]))

    # Cross-profile transfer matrix
    names = [n for n, (_, y) in payload.items() if len(y) > 0]
    if len(names) >= 2:
        print("\nCross-profile transfer accuracy (train -> test, overlap-only labels):")
        for a in names:
            Xa, ya = payload[a]
            for b in names:
                if a == b:
                    continue
                Xb, yb = payload[b]
                acc, n_eval = _cross_profile_accuracy(Xa, ya, Xb, yb)
                print(f"  {a:10s} -> {b:10s}: acc={acc:.3f} on n={n_eval}")


if __name__ == "__main__":
    main()
