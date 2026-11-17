"""Precision-weighted operating points for canned models and fresh heads.

Each target is scored as ``TP - cost * FP`` (one-to-one exact-pitch/drum-class
onset matches within 50 ms, query region only; misses cost nothing extra).
Thresholds are chosen on three training source-group folds and reported on the
fourth. Validation and test variants are never read.

Run ``python -m experimental.benchmark.threshold_sweep --out <fresh run>``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import logging
from pathlib import Path

import numpy as np

from core.evaluation import _match
from experimental.benchmark import fresh_experiment as fx

LOG = logging.getLogger(__name__)
HEAD_THRESHOLDS = tuple(float(t) for t in np.round(np.arange(0.05, 0.951, 0.05), 2))
CANNED_QUANTILES = np.round(np.arange(0, 0.951, 0.05), 2)


def query(rows) -> np.ndarray:
    rows = np.asarray(rows, dtype=float)
    rows = rows.reshape(-1, rows.shape[1] if rows.ndim == 2 else 4)
    return rows[rows[:, 0] >= fx.QUERY_START]


def true_positives(reference: np.ndarray, predictions: np.ndarray) -> int:
    """Maximum one-to-one exact-pitch matching, solved independently per pitch."""
    by_pitch = defaultdict(list)
    for row in predictions:
        by_pitch[int(row[2])].append(row)
    references = defaultdict(list)
    for row in reference:
        references[int(row[2])].append(row)
    return sum(len(_match(fx.events(rows), fx.events(by_pitch[pitch]),
                          fx.TOLERANCE, False, fx.TOLERANCE, 0.2, True))
               for pitch, rows in references.items() if pitch in by_pitch)


def counts(reference: np.ndarray, predictions: np.ndarray, scores: np.ndarray,
           thresholds) -> np.ndarray:
    """(thresholds, [TP, FP, FN]) keeping predictions whose score >= threshold."""
    result = np.zeros((len(thresholds), 3), dtype=np.int64)
    for i, threshold in enumerate(thresholds):
        kept = predictions[scores >= threshold]
        tp = true_positives(reference, kept)
        result[i] = tp, len(kept) - tp, len(reference) - tp
    return result


def rates(tp: int, fp: int, fn: int, cost: float) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "true_positives": int(tp), "false_positives": int(fp), "false_negatives": int(fn),
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "wrong_per_correct": fp / tp if tp else None,
        "net": tp - cost * fp, "net_per_reference_note": (tp - cost * fp) / (tp + fn) if tp + fn else None,
    }


def cross_fold(totals: np.ndarray, thresholds, cost: float, default: int) -> dict:
    """Pick each held fold's threshold from the other folds only; pool held results."""
    chosen, held = [], np.zeros(3, dtype=np.int64)
    for fold in range(len(totals)):
        rest = np.delete(totals, fold, axis=0).sum(axis=0)
        index = int(np.argmax(rest[:, 0] - cost * rest[:, 1]))
        chosen.append(float(thresholds[index]))
        held += totals[fold, index]
    pooled = totals.sum(axis=0)
    return {
        "cost": cost, "fold_thresholds": chosen,
        "tuned": rates(*held, cost), "default_threshold": float(thresholds[default]),
        "default": rates(*pooled[default], cost),
    }


def curve(totals: np.ndarray, thresholds, cost: float) -> list[dict]:
    pooled = totals.sum(axis=0)
    return [{"threshold": float(t), **rates(*pooled[i], cost)} for i, t in enumerate(thresholds)]


def canned_totals(output: Path, records: list[dict], fold_of: dict, folds: int,
                  role: str, model: str) -> tuple[np.ndarray, np.ndarray]:
    items = []
    for record in records:
        reference = record["notes"][role]
        if role == "drums":
            reference = fx.drum_reference(reference)
        prediction = fx.read_json(fx.prediction_path(output, record, fx.target_stem(record, role), model))
        items.append((fold_of[record["source_group"]], query(reference), query(prediction["events"])))
    scores = np.concatenate([p[:, 3] for *_, p in items])
    # The model's native output is the lowest threshold; higher ones drop low-confidence notes.
    thresholds = np.unique(np.quantile(scores, CANNED_QUANTILES)) if len(scores) else np.zeros(1)
    totals = np.zeros((folds, len(thresholds), 3), dtype=np.int64)
    for fold, reference, predictions in items:
        totals[fold] += counts(reference, predictions, predictions[:, 3], thresholds)
    return thresholds, totals


def head_totals(output: Path, experiment: dict, records: list[dict], fold_of: dict,
                folds: int, role: str, choice: dict, iterations: int) -> tuple[np.ndarray, list[bool], dict]:
    subset = [{**record, "notes": {role: record["notes"][role]}} for record in records]
    views = [view for view in fx.load_views(output, experiment, subset) if view.config == choice["config"]]
    totals = np.zeros((folds, len(HEAD_THRESHOLDS), 3), dtype=np.int64)
    converged, default_rows = [], {}
    for fold in range(folds):
        head = fx.fit_head([v for v in views if fold_of[v.record["source_group"]] != fold],
                           choice["architecture"], iterations)
        converged.append(head.converged)
        for view in views:
            if fold_of[view.record["source_group"]] != fold:
                continue
            mask = view.candidates[:, 0] >= fx.QUERY_START
            candidates = view.candidates[mask]
            probabilities = head.probabilities(view.X[mask]) if mask.any() else np.empty(0)
            clip = counts(query(view.reference), candidates, probabilities, HEAD_THRESHOLDS)
            totals[fold] += clip
            default_rows[view.record["clip"]] = clip[HEAD_THRESHOLDS.index(0.5)].tolist()
        LOG.info("Head thresholds: %s %s %s fold %d/%d", role, choice["config"],
                 choice["architecture"], fold + 1, folds)
    return totals, converged, default_rows


def sweep(output: Path, costs: tuple[float, ...]) -> dict:
    experiment = fx.load_experiment(output)
    selection = fx.read_json(output / "selection.json")
    target = output / "precision_thresholds.json"
    if target.exists():
        raise FileExistsError("Precision thresholds were already swept for this run")
    fold_of = selection["fold_of_source_group"]
    folds = len(set(fold_of.values()))
    records = [record for record in experiment["records"] if record["split"] == "train"]
    if {record["source_group"] for record in records} != set(fold_of):
        raise ValueError("Training records do not match the cross-validation folds")
    report = {
        "scope": "Training source-group CV only; validation and test not read",
        "metric": "TP - cost*FP; exact pitch/drum class, one-to-one onset match <=50 ms; query region only",
        "selection": "Each fold's threshold maximizes pooled net on the other three folds",
        "limitations": [
            "Canned thresholds can only remove notes from the cached native output; lowering "
            "them needs new inference. Basic Pitch amplitude, Piano-HR velocity and ADTOF "
            "novelty are confidence proxies.",
            "Head thresholds for a fold are chosen from other folds' cross-fitted heads, whose "
            "training data included this fold; this leakage is limited to one scalar threshold.",
            "Settings were already selected by F1 on these folds; validation must confirm them.",
        ],
        "canned": {}, "heads": {},
    }
    for role in sorted({role for record in records for role in record["notes"]}):
        for config, models in fx.input_configs(role, experiment["models"]).items():
            if not config.startswith("single_"):
                continue
            thresholds, totals = canned_totals(output, records, fold_of, folds, role, models[0])
            report["canned"][f"{role}/{models[0]}"] = {
                "role": role, "system": models[0],
                "operating_points": [cross_fold(totals, thresholds, cost, 0) for cost in costs],
                "curve": curve(totals, thresholds, costs[0]),
            }
            LOG.info("Canned thresholds: %s %s", role, models[0])
        choice = selection["selection"][role]
        totals, converged, default_rows = head_totals(
            output, experiment, records, fold_of, folds, role, choice, selection["iterations"])
        report["heads"][role] = {
            "role": role, "config": choice["config"], "architecture": choice["architecture"],
            "fit_converged": converged, "default_clip_counts": default_rows,
            "operating_points": [cross_fold(totals, HEAD_THRESHOLDS, cost, HEAD_THRESHOLDS.index(0.5))
                                 for cost in costs],
            "curve": curve(totals, HEAD_THRESHOLDS, costs[0]),
        }
        fx.write_json(target.with_suffix(".partial.json"), report)
    fx.write_json(target, report)
    target.with_suffix(".partial.json").unlink(missing_ok=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--costs", default="2,1,3", help="Wrong-note costs; the first is the primary curve")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        costs = tuple(float(value) for value in args.costs.split(","))
        if not costs or any(not np.isfinite(c) or c < 0 for c in costs):
            raise ValueError("Wrong-note costs must be finite and non-negative")
        sweep(args.out, costs)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        LOG.error("Threshold sweep stopped: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
