"""Rescore the training-CV experiment with song-pooled F0.5 (see ``scoring``).

* Canned models: native output and a confidence threshold tuned for F0.5 on the
  other three folds (cached output can only be thinned).
* Heads: every input config and architecture from ``cv_rows.json`` at the
  original 0.5 keep threshold, before and after simulated edits.
* The earlier F1 adaptation study, rescored at its 0.5 threshold.

Everything is recomputed from stored TP/FP/FN counts or frozen predictions; no
model is refit. Validation and test variants are never read. Run
``python -m experimental.benchmark.fbeta_report --out <fresh run>``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from experimental.benchmark import fresh_experiment as fx
from experimental.benchmark import scoring
from experimental.benchmark.threshold_sweep import CANNED_QUANTILES, counts, query

LOG = logging.getLogger(__name__)
CV_FIELDS = ("role", "config", "architecture", "policy", "lam", "budget", "source_group", "fold")


def iter_json_array(path: Path, chunk: int = 1 << 24):
    """Yield the objects of a large top-level JSON array without loading it whole."""
    decoder = json.JSONDecoder()
    whitespace = " \t\r\n,"
    with open(path, encoding="utf-8") as handle:
        buffer = handle.read(chunk).lstrip()
        if not buffer.startswith("["):
            raise ValueError(f"{path} is not a JSON array")
        position, eof = 1, False
        while True:
            while position < len(buffer) and buffer[position] in whitespace:
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            try:
                if position >= len(buffer):
                    raise json.JSONDecodeError("need more data", buffer, position)
                item, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                more = handle.read(chunk)
                eof = not more
                buffer, position = buffer[position:] + more, 0
                continue
            yield item
            position = end


def count_row(row: dict, **extra) -> dict:
    return {**{key: row[key] for key in CV_FIELDS if key in row}, **extra,
            "tp": row["true_positives"], "fp": row["false_positives"], "fn": row["false_negatives"]}


def canned(output: Path, experiment: dict, records: list[dict], fold_of: dict, beta: float) -> list[dict]:
    rows = fx.read_json(output / "canned_training_baselines.json")["rows"]
    native = [count_row(row, system=row["system"]) for row in rows if row["split"] == "train"]
    summaries = [{**s, "operating_point": "native"}
                 for s in scoring.summarize(native, ("role", "system"), beta)]
    for role in sorted({row["role"] for row in native}):
        for system in sorted({row["system"] for row in native if row["role"] == role}):
            items = []
            for record in records:
                reference = record["notes"][role]
                if role == "drums":
                    reference = fx.drum_reference(reference)
                path = fx.prediction_path(output, record, fx.target_stem(record, role), system)
                items.append((record, query(reference), query(fx.read_json(path)["events"])))
            scores = np.concatenate([p[:, 3] for *_, p in items])
            thresholds = np.unique(np.quantile(scores, CANNED_QUANTILES)) if len(scores) else np.zeros(1)
            grid = np.stack([counts(reference, predictions, predictions[:, 3], thresholds)
                             for _, reference, predictions in items])
            groups = [record["source_group"] for record, *_ in items]
            folds = [fold_of[group] for group in groups]
            chosen = scoring.tune_thresholds(grid, groups, folds, beta)
            held = [{"role": role, "system": system, "source_group": group,
                     **dict(zip(("tp", "fp", "fn"), grid[i, chosen[fold]].tolist()))}
                    for i, (group, fold) in enumerate(zip(groups, folds))]
            summary = scoring.summarize(held, ("role", "system"), beta)[0]
            summaries.append({**summary, "operating_point": "tuned",
                              "fold_thresholds": [float(thresholds[i]) for i in chosen],
                              "fold_quantiles": [float(np.mean(scores < thresholds[i])) for i in chosen]})
            LOG.info("Canned %s %s tuned", role, system)
    return summaries


def heads(output: Path, beta: float) -> list[dict]:
    keys = ("role", "config", "architecture", "policy", "lam", "budget")
    rows = (count_row(row) for row in iter_json_array(output / "cv_rows.json") if row["split"] == "train")
    groups = scoring.song_counts(rows, keys)
    LOG.info("Aggregated %d head CV groups", len(groups))
    frozen = {key[:3]: songs for key, songs in groups.items() if key[3] == "frozen" and key[5] == 0}
    return scoring.summarize_counts(groups, keys, beta, lambda key: frozen[key[:3]])


def old_study(output: Path, beta: float) -> list[dict]:
    summaries = []
    for path in sorted((output / "adaptation_study").glob("*.json")):
        if path.name.endswith(".partial.json"):
            continue
        rows = fx.read_json(path)["rows"]
        frozen = scoring.song_counts([r for r in rows if r["policy"] == "frozen"], ("role", "architecture"))
        summaries += scoring.summarize(rows, ("role", "architecture", "policy", "lam", "budget"), beta,
                                       lambda key: frozen[key[:2]])
    return summaries


def report(output: Path, beta: float) -> Path:
    experiment = fx.load_experiment(output)
    selection = fx.read_json(output / "selection.json")
    target = output / f"fbeta{beta:g}_report.json"
    if target.exists():
        raise FileExistsError(f"{target} already exists")
    fold_of = selection["fold_of_source_group"]
    records = [record for record in experiment["records"] if record["split"] == "train"]
    if {record["source_group"] for record in records} != set(fold_of):
        raise ValueError("Training records do not match the cross-validation folds")
    result = {
        "scope": "Training source-group CV only; validation and test not read",
        "metric": f"Mean over source songs of F{beta:g} on pooled TP/FP/FN (a wrong note weighs "
                  f"{1 / beta ** 2:g}x a missed note; empty and silent scores 1); exact pitch/drum "
                  "class, one-to-one onset match <=50 ms; query region only",
        "beta": beta,
        "canned": canned(output, experiment, records, fold_of, beta),
        "old_adaptation_study": old_study(output, beta),
        "heads": heads(output, beta),
    }
    fx.write_json(target, result)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=scoring.BETA)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        report(args.out, args.beta)
    except (OSError, ValueError, KeyError) as exc:
        LOG.error("F-beta report stopped: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
