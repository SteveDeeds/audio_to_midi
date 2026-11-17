"""Precision-weighted scoring: F-beta pooled per source song.

``F_beta = (1 + b^2) TP / ((1 + b^2) TP + b^2 FN + FP)``. With ``b = 0.5`` a wrong
note costs four times as much as a missed note in the denominator; the score
stays in [0, 1] and 1 still means every reference note found with nothing extra.
Counts are pooled over each source song's views before scoring, so an empty
reference stem scores 1 when left silent and is penalized in proportion to the
wrong notes added, and every song carries equal weight.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

BETA = 0.5


def f_beta(tp: float, fp: float, fn: float, beta: float = BETA) -> float:
    if tp + fp + fn == 0:
        return 1.0
    b2 = beta * beta
    return (1 + b2) * tp / ((1 + b2) * tp + b2 * fn + fp)


def f_beta_array(counts: np.ndarray, beta: float = BETA) -> np.ndarray:
    """Vectorized ``f_beta`` over a trailing [TP, FP, FN] axis."""
    counts = np.asarray(counts, dtype=float)
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    b2 = beta * beta
    denominator = (1 + b2) * tp + b2 * fn + fp
    return np.where(denominator > 0, (1 + b2) * tp / np.where(denominator > 0, denominator, 1), 1.0)


def song_scores(counts: np.ndarray, groups, beta: float = BETA) -> tuple[list, np.ndarray]:
    """Pool [..., views, 3] counts by source group; return (groups, scores[..., groups])."""
    names = sorted(set(groups))
    index = {name: i for i, name in enumerate(names)}
    counts = np.asarray(counts, dtype=float)
    pooled = np.zeros((*counts.shape[:-2], len(names), 3))
    for view, group in enumerate(groups):
        pooled[..., index[group], :] += counts[..., view, :]
    return names, f_beta_array(pooled, beta)


def tune_thresholds(view_counts: np.ndarray, groups, view_folds, beta: float = BETA) -> list[int]:
    """For each fold, the threshold index maximizing mean song F-beta on the other folds.

    ``view_counts`` is (views, thresholds, 3). Ties go to the lowest index.
    """
    view_folds = np.asarray(view_folds)
    groups = np.asarray(groups)
    chosen = []
    for fold in sorted(set(view_folds.tolist())):
        rest = view_folds != fold
        _, scores = song_scores(np.moveaxis(view_counts[rest], 1, 0), groups[rest].tolist(), beta)
        chosen.append(int(np.argmax(scores.mean(axis=1))))
    return chosen


def summarize(rows: list[dict], keys: tuple[str, ...], beta: float = BETA,
              baseline=None) -> list[dict]:
    """Song-pooled F-beta, precision and recall per ``keys`` group.

    ``baseline(key)`` returns {song: [TP, FP, FN]} used for gains: each song's gain
    is its pooled score minus the same song's pooled baseline score.
    """
    return summarize_counts(song_counts(rows, keys), keys, beta, baseline)


def summarize_counts(groups: dict, keys: tuple[str, ...], beta: float = BETA,
                     baseline=None) -> list[dict]:
    """``summarize`` from {key: {song: [TP, FP, FN]}} built by ``song_counts``."""
    summaries = []
    for key, by_song in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        songs = sorted(by_song)
        counts = np.array([by_song[song] for song in songs])
        scores = f_beta_array(counts, beta)
        total = counts.sum(axis=0)
        summary = {
            **dict(zip(keys, key)), "source_groups": len(songs), "f_beta": float(scores.mean()),
            "f1_song_pooled": float(f_beta_array(counts, 1.0).mean()),
            "precision": float(total[0] / (total[0] + total[1])) if total[0] + total[1] else 0.0,
            "recall": float(total[0] / (total[0] + total[2])) if total[0] + total[2] else 0.0,
            "wrong_per_correct": float(total[1] / total[0]) if total[0] else None,
            "tp": int(total[0]), "fp": int(total[1]), "fn": int(total[2]),
        }
        if baseline is not None:
            reference = baseline(key)
            gain = scores - f_beta_array(np.array([reference[song] for song in songs]), beta)
            boot = np.random.default_rng(2026).choice(gain, (1000, len(gain))).mean(axis=1)
            summary.update({
                "gain": float(gain.mean()), "gain_bootstrap95": np.quantile(boot, [0.025, 0.975]).tolist(),
                "regression_fraction": float(np.mean(gain < -1e-9)),
                "improved_fraction": float(np.mean(gain > 1e-9)),
                "p10_gain": float(np.quantile(gain, 0.1)), "worst_gain": float(gain.min()),
            })
        summaries.append(summary)
    return summaries


def song_counts(rows, keys: tuple[str, ...]) -> dict:
    """Pooled [TP, FP, FN] per song, keyed by ``keys`` then source group."""
    result = defaultdict(lambda: defaultdict(lambda: np.zeros(3)))
    for row in rows:
        result[tuple(row[key] for key in keys)][row["source_group"]] += (row["tp"], row["fp"], row["fn"])
    return result
