"""Architecture and edit-adaptation study on training source-group CV (F1 metric).

Why: the earlier edit updates trained only on the notes a user fixed (errors), so
they never saw confirmed-correct notes and often over-corrected. A user who fixes
the first k errors in time order has also *reviewed* everything up to the k-th
edit, so every canned candidate in that window gets an implicit label. This study
compares, for heads of several sizes (logistic, tiny/medium MLPs, a deeper DNN,
boosted trees):

* ``anchored_errors``: the previous all-layer update on explicit edits only;
* ``*_window``: updates on all candidates in the reviewed window;
* ``platt_window``: two-parameter recalibration of the frozen head's logit;
* ``gated_*``: a do-no-harm gate that fits on the earlier part of the reviewed
  window, keeps the update only if it lowers log loss on the later part, then
  refits on the whole window.

Settings are compared on the same four training folds as ``selection.json``.
Validation and test variants are never read. Run
``python -m experimental.benchmark.adaptation_study --out <run> --role bass``.

With ``--beta 0.5 --study adaptation_study_f05`` the keep threshold is tuned for
song-pooled F0.5 (wrong notes cost more than misses; see ``scoring``). Each
fold's threshold is chosen from the other folds' out-of-fold head probabilities
and is used both for what the user sees (so which edits they make) and for
scoring.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import logging
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from experimental.benchmark import fresh_experiment as fx
from experimental.benchmark import scoring
from experimental.benchmark.small_heads import SmallHead
from experimental.benchmark.threshold_sweep import HEAD_THRESHOLDS, counts, query, true_positives

LOG = logging.getLogger(__name__)
NETWORKS = {
    "linear": (), "mlp4": (4,), "mlp16": (16,), "mlp64x32": (64, 32),
    "dnn256x128x64": (256, 128, 64),
}
ARCHITECTURES = (*NETWORKS, "boosted_trees")
# (policy, regularization); "frozen" and the window policies need no truth beyond reviewed edits.
NETWORK_POLICIES = (
    ("anchored_errors", 100.0),
    ("anchored_window", 100.0), ("anchored_window", 1000.0),
    ("last_layer_window", 100.0),
    ("platt_window", 1.0), ("platt_window", 10.0),
    ("gated_anchored_window", 1000.0), ("gated_platt_window", 10.0),
)
TREE_POLICIES = (("platt_window", 1.0), ("platt_window", 10.0), ("gated_platt_window", 10.0))
BUDGETS = (1, 2, 5, 10, 20, 40)
MIN_GATE_LABELS = 6


class BoostedTrees:
    """Gradient-boosted trees exposing the same logit interface as ``SmallHead``."""

    def __init__(self, model):
        self.model = model
        self.converged = True

    def logits(self, X: np.ndarray) -> np.ndarray:
        if not len(X):
            return np.zeros(0)
        return self.model.decision_function(X)

    def probabilities(self, X: np.ndarray) -> np.ndarray:
        return expit(self.logits(X))

    def parameter_count(self) -> int:
        return int(sum(tree.get_n_leaf_nodes() for trees in self.model._predictors for tree in trees))


def source_weights(views: list) -> np.ndarray:
    """Equal weight per source group, as ``fresh_experiment.fit_head`` uses."""
    counts = defaultdict(int)
    for view in views:
        counts[view.record["source_group"]] += 1
    weights = np.concatenate([np.full(len(v.X), 1 / (len(v.X) * counts[v.record["source_group"]]))
                              for v in views])
    return weights * len(weights) / weights.sum()


def fit(views: list, architecture: str, iterations: int, seed: int = 2026):
    views = [view for view in views if len(view.X)]
    X = np.vstack([view.X for view in views])
    y = np.concatenate([view.y for view in views])
    weights = source_weights(views)
    if architecture == "boosted_trees":
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                               l2_regularization=1.0, early_stopping=False,
                                               random_state=seed)
        return BoostedTrees(model.fit(X, y, sample_weight=weights))
    rng = np.random.default_rng(seed)
    sizes = (X.shape[1], *NETWORKS[architecture], 1)
    head = SmallHead(X.mean(axis=0), np.maximum(X.std(axis=0), 0.05),
                     [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes, sizes[1:])],
                     [np.zeros(b) for b in sizes[1:]])
    parameters, head.converged = head.optimize(X, y, lam=1.0, iterations=iterations,
                                               anchor=np.zeros_like(head.pack()), row_weights=weights)
    head.weights, head.biases = head.unpack(parameters)
    return head


def parameter_count(head) -> int:
    return head.parameter_count() if isinstance(head, BoostedTrees) else len(head.pack())


def platt(frozen_logits: np.ndarray, window_logits: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Fit ``a*z + b`` regularized toward the identity; return adapted logits."""
    def loss(value):
        a, b = value
        z = a * window_logits + b
        residual = expit(z) - y
        return (float(np.sum(np.logaddexp(0, z) - y * z) + lam / 2 * ((a - 1) ** 2 + b ** 2)),
                np.array([residual @ window_logits + lam * (a - 1), residual.sum() + lam * b]))
    result = minimize(loss, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B")
    if not np.isfinite(result.x).all():
        raise RuntimeError(f"Platt recalibration failed: {result.message}")
    return result.x[0] * frozen_logits + result.x[1]


def adapted_logits(head, policy: str, lam: float, X: np.ndarray, train_X: np.ndarray,
                   train_y: np.ndarray) -> np.ndarray:
    base = head.logits(X)
    if not len(train_y):
        return base
    if policy.startswith("platt"):
        return platt(base, head.logits(train_X), train_y, lam)
    kind = "last_layer" if policy.startswith("last_layer") else "anchored"
    parameters, _ = head.optimize(train_X, train_y, lam=lam, iterations=60, anchor=head.pack(),
                                  last_layer=kind == "last_layer")
    return head.logits(X, parameters)


def log_loss(logits: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0, logits) - y * logits))


def reviewed_window(view, stream: list, budget: int) -> np.ndarray:
    """Support-region candidates whose truth the user has seen after ``budget`` edits.

    Edits are made in time order, so everything up to the onset of the last edit
    has been reviewed. When the user has fixed every support error, the whole
    support region is reviewed.
    """
    if not budget or not stream:
        return np.zeros(len(view.candidates), dtype=bool)
    end = fx.SUPPORT_SECONDS if budget >= len(stream) else stream[budget - 1][0][0]
    onsets = view.candidates[:, 0]
    return (onsets < fx.SUPPORT_SECONDS) & (onsets <= end)


def gated(head, policy: str, lam: float, X: np.ndarray, window_X: np.ndarray,
          window_y: np.ndarray, window_onsets: np.ndarray) -> tuple[np.ndarray, bool]:
    """Keep an update only if it beats the frozen head on the later part of the window."""
    base = head.logits(X)
    if len(window_y) < MIN_GATE_LABELS:
        return base, False
    order = np.argsort(window_onsets, kind="stable")
    cut = int(np.ceil(len(order) * 2 / 3))
    early, late = order[:cut], order[cut:]
    trial = adapted_logits(head, policy, lam, window_X[late], window_X[early], window_y[early])
    if log_loss(trial, window_y[late]) >= log_loss(head.logits(window_X[late]), window_y[late]):
        return base, False
    return adapted_logits(head, policy, lam, X, window_X, window_y), True


def score(view, keep: np.ndarray) -> tuple[int, int, int, float]:
    query = view.candidates[:, 0] >= fx.QUERY_START
    reference = np.asarray([row for row in view.reference if row[0] >= fx.QUERY_START],
                           dtype=float).reshape(-1, 4)
    kept = view.candidates[query & keep]
    tp = true_positives(reference, kept)
    fp, fn = len(kept) - tp, len(reference) - tp
    return tp, fp, fn, 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def edit_stream(view, head, threshold: float = 0.5) -> list[tuple[np.ndarray, float]]:
    """``fresh_experiment.edit_stream`` with a configurable keep threshold."""
    support_candidates = view.candidates[view.candidates[:, 0] < fx.SUPPORT_SECONDS]
    support_X = fx.features(support_candidates, view.channels)
    predicted = support_candidates[head.probabilities(support_X) >= threshold]
    reference = [row for row in view.reference if row[0] < fx.SUPPORT_SECONDS]
    matches = fx._match(fx.events(reference), fx.events(predicted.tolist()),
                        fx.TOLERANCE, False, fx.TOLERANCE, 0.2, True)
    ref_used = {left for left, _ in matches}
    pred_used = {right for _, right in matches}
    edits = [(row.copy(), 0.0) for i, row in enumerate(predicted) if i not in pred_used]
    edits.extend((np.array([*row[:3], row[3] / 127]), 1.0)
                 for i, row in enumerate(reference) if i not in ref_used)
    return sorted(edits, key=lambda edit: (edit[0][0], edit[0][2], edit[1]))


def threshold_counts(view, head) -> np.ndarray:
    """(len(HEAD_THRESHOLDS), [TP, FP, FN]) for the frozen head on the query region."""
    mask = view.candidates[:, 0] >= fx.QUERY_START
    probabilities = head.probabilities(view.X[mask]) if mask.any() else np.empty(0)
    return counts(query(view.reference), view.candidates[mask], probabilities, HEAD_THRESHOLDS)


def evaluate_view(view, head, architecture: str, threshold: float = 0.5) -> list[dict]:
    frozen_logits = head.logits(view.X)
    stream = edit_stream(view, head, threshold)
    frozen = score(view, expit(frozen_logits) >= threshold)
    base = {"clip": view.record["clip"], "source_group": view.record["source_group"],
            "role": view.role, "config": view.config, "architecture": architecture,
            "parameters": parameter_count(head), "support_errors": len(stream),
            "threshold": threshold}
    rows = [{**base, "policy": "frozen", "lam": 0.0, "budget": 0, "window_labels": 0,
             "accepted": False, "update_seconds": 0.0,
             **dict(zip(("tp", "fp", "fn", "f1"), frozen)), "gain": 0.0}]
    policies = TREE_POLICIES if isinstance(head, BoostedTrees) else NETWORK_POLICIES
    for budget in BUDGETS:
        edits = stream[:budget]
        window = reviewed_window(view, stream, budget)
        window_X, window_y = view.X[window], view.y[window]
        for policy, lam in policies:
            began = time.perf_counter()
            accepted = bool(edits)
            if policy == "anchored_errors":
                edits_X = fx.features(np.array([row for row, _ in edits]).reshape(-1, 4), view.channels)
                logits = adapted_logits(head, policy, lam, view.X, edits_X,
                                        np.array([label for _, label in edits]))
            elif policy.startswith("gated_"):
                logits, accepted = gated(head, policy.removeprefix("gated_"), lam, view.X,
                                         window_X, window_y, view.candidates[window, 0])
            else:
                logits = adapted_logits(head, policy, lam, view.X, window_X, window_y)
                accepted = bool(len(window_y))
            tp, fp, fn, f1 = score(view, expit(logits) >= threshold)
            rows.append({**base, "policy": policy, "lam": lam, "budget": budget,
                         "window_labels": int(window.sum()), "accepted": accepted,
                         "update_seconds": time.perf_counter() - began,
                         "tp": tp, "fp": fp, "fn": fn, "f1": f1, "gain": f1 - frozen[3]})
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["role"], row["architecture"], row["policy"], row["lam"], row["budget"])
        groups[key][row["source_group"]].append(row)
    summaries = []
    for key, by_source in sorted(groups.items()):
        f1 = np.array([np.mean([r["f1"] for r in group]) for group in by_source.values()])
        gain = np.array([np.mean([r["gain"] for r in group]) for group in by_source.values()])
        boot = np.random.default_rng(2026).choice(gain, (1000, len(gain))).mean(axis=1)
        flat = [row for group in by_source.values() for row in group]
        summaries.append({
            **dict(zip(("role", "architecture", "policy", "lam", "budget"), key)),
            "source_groups": len(by_source), "f1": float(f1.mean()), "gain": float(gain.mean()),
            "gain_bootstrap95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "regression_fraction": float(np.mean(gain < -1e-9)),
            "improved_fraction": float(np.mean(gain > 1e-9)),
            "p10_gain": float(np.quantile(gain, 0.1)), "worst_gain": float(gain.min()),
            "accepted_fraction": float(np.mean([r["accepted"] for r in flat])),
            "mean_update_seconds": float(np.mean([r["update_seconds"] for r in flat])),
            "parameters": int(np.median([r["parameters"] for r in flat])),
        })
    return summaries


def frozen_baseline(rows: list[dict]):
    """Per-song frozen counts for gains, keyed like ``scoring.summarize`` groups."""
    frozen = scoring.song_counts([row for row in rows if row["policy"] == "frozen"],
                                 ("role", "architecture"))
    return lambda key: frozen[key[:2]]


def run(output: Path, role: str, architectures: tuple[str, ...], iterations: int,
        resume: bool = False, beta: float | None = None, study: str = "adaptation_study") -> Path:
    experiment = fx.load_experiment(output)
    selection = fx.read_json(output / "selection.json")
    target = output / study / f"{role}.json"
    if target.exists():
        raise FileExistsError(f"{target} already exists; use a new run for another study")
    if set(architectures) - set(ARCHITECTURES):
        raise ValueError(f"Choose architectures from {ARCHITECTURES}")
    target.parent.mkdir(parents=True, exist_ok=True)
    fold_of = selection["fold_of_source_group"]
    folds = len(set(fold_of.values()))
    config = selection["selection"][role]["config"]
    records = [{**record, "notes": {role: record["notes"][role]}}
               for record in experiment["records"] if record["split"] == "train"]
    if {record["source_group"] for record in records} != set(fold_of):
        raise ValueError("Training records do not match the cross-validation folds")
    views = [view for view in fx.load_views(output, experiment, records) if view.config == config]
    view_folds = [fold_of[view.record["source_group"]] for view in views]
    rows, fits = [], []
    partial = target.with_suffix(".partial.json")
    if resume and partial.exists():
        saved = fx.read_json(partial)
        fits, rows = saved["fits"], saved["rows"]
        LOG.info("Resuming %s from %d completed fits", role, len(fits))
    done = {(fit_["architecture"], fit_["fold"]) for fit_ in fits}
    for architecture in architectures:
        todo = [fold for fold in range(folds) if (architecture, fold) not in done]
        if not todo:
            continue
        heads, seconds = {}, {}
        # Tuned thresholds need every fold's out-of-fold probabilities, so fit all folds first.
        for fold in (range(folds) if beta is not None else todo):
            began = time.perf_counter()
            heads[fold] = fit([v for v, f in zip(views, view_folds) if f != fold], architecture, iterations)
            seconds[fold] = time.perf_counter() - began
        thresholds = {fold: 0.5 for fold in heads}
        if beta is not None:
            grid = np.stack([threshold_counts(view, heads[f]) for view, f in zip(views, view_folds)])
            groups = [view.record["source_group"] for view in views]
            chosen = scoring.tune_thresholds(grid, groups, view_folds, beta)
            thresholds = {fold: HEAD_THRESHOLDS[index] for fold, index in enumerate(chosen)}
        for fold in todo:
            head = heads[fold]
            fits.append({"architecture": architecture, "fold": fold, "converged": bool(head.converged),
                         "parameters": parameter_count(head), "threshold": thresholds[fold],
                         "fit_seconds": seconds[fold]})
            for view, view_fold in zip(views, view_folds):
                if view_fold == fold:
                    rows.extend({**row, "fold": fold}
                                for row in evaluate_view(view, head, architecture, thresholds[fold]))
            LOG.info("%s %s %s fold %d/%d threshold %.2f", role, config, architecture, fold + 1,
                     folds, thresholds[fold])
            fx.write_json(partial, {"fits": fits, "rows": rows})
    result = {
        "scope": "Training source-group CV only; validation and test not read",
        "role": role, "config": config, "iterations": iterations,
        "fold_of_source_group": fold_of, "fits": fits, "rows": rows,
    }
    if beta is None:
        result.update(metric="F1; exact pitch/drum class, one-to-one onset match <=50 ms; query region only",
                      summary=summarize(rows))
    else:
        result.update(
            metric=f"Song-pooled F{beta:g} (wrong notes weigh 1/beta^2 times a miss); exact pitch/drum "
                   "class, one-to-one onset match <=50 ms; query region only",
            beta=beta, threshold_selection="Per fold: maximizes mean song F-beta of the frozen head "
                                           "on the other folds' out-of-fold probabilities",
            summary=scoring.summarize(rows, ("role", "architecture", "policy", "lam", "budget"),
                                      beta, frozen_baseline(rows)))
    fx.write_json(target, result)
    partial.unlink(missing_ok=True)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--architectures", default=",".join(ARCHITECTURES))
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--resume", action="store_true",
                        help="Skip architecture/fold fits already saved in <role>.partial.json")
    parser.add_argument("--beta", type=float, default=None,
                        help="Tune the keep threshold for song-pooled F-beta (e.g. 0.5)")
    parser.add_argument("--study", default="adaptation_study", help="Output subdirectory of --out")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        run(args.out, args.role, tuple(args.architectures.split(",")), args.iterations, args.resume,
            args.beta, args.study)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        LOG.error("Adaptation study stopped: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
