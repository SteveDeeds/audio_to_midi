"""Cross-song pre-training for the correction heads, on imported ``atm`` clips.

    python -m experimental.benchmark.crosssong extract [--workers 2]
    python -m experimental.benchmark.crosssong cv      [--lams 0.3,1,3,10] [--implicits 0,0.25]
    python -m experimental.benchmark.crosssong final   --lam 3 --implicit 0.25
    python -m experimental.benchmark.crosssong export  [--out experimental/benchmark/runs/priors.json]

The per-song heads in ``ensemble.py`` (onsets) and ``pitch_head.py`` (which pitches) start from a
hand-set prior that just reproduces the app, so the first clicks on every song re-learn what a
good onset or note looks like. Here the prior is learned instead: a logistic regression fitted
on every cell / candidate of the training songs (features are standardized per song, so weights
transfer between songs), then each song's clicks adapt it with the usual L2 pull of ``--lam``.

``extract`` builds the cached per-stem tables (the slow part: every model runs once per stem).
``cv`` splits the training songs into folds by composition, pre-trains on the other folds, and
simulates clicks on the held-out fold for each (lam, implicit) pair. ``final`` pre-trains on all
training songs and reports the validation and test songs. Clips are the ``atmx_*`` folders that
``import_atm.py`` writes, grouped by their ``split``; nothing from a song ever trains its own prior.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from experimental.benchmark import ensemble, pitch_head
from experimental.benchmark.harness import clip_truth, stem_path
from experimental.benchmark.prepare import CLIPS

ONSET_ROLES = ("bass", "piano", "guitar")
NOTE_ROLES = ("piano", "guitar")
ONSET_SETS = ("models", "all")
BUDGETS = [0, 5, 10, 20, 40]
PRE_LAMBDA = 1.0


def clips_by_split(prefix: str) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = defaultdict(list)
    for clip in sorted(CLIPS.glob(f"{prefix}*")):
        if (clip / "truth.json").exists():
            out[clip_truth(clip).get("split") or "train"].append(clip)
    return out


def stems(clip: Path, sources: list[str], roles: tuple[str, ...]) -> list[tuple[str, str]]:
    notes = clip_truth(clip)["notes"]
    return [(s, r) for s in sources for r in roles if r in notes and stem_path(clip, s, r)]


# -- extract -----------------------------------------------------------------------------------
def _limit_threads(threads: int) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(threads)
    try:
        import torch
        torch.set_num_threads(threads)
    except ImportError:
        pass


def extract_clip(clip: Path, sources: list[str]) -> str:
    began = time.perf_counter()
    done = []
    for source, role in stems(clip, sources, ONSET_ROLES):
        try:
            ensemble.onset_table(clip, source, role)
            if role in NOTE_ROLES:
                pitch_head.note_table(clip, source, role)
            done.append(f"{source}/{role}")
        except Exception:  # noqa: BLE001 - one bad stem shouldn't stop the batch
            done.append(f"{source}/{role} FAILED")
            print(f"{clip.name} {source}/{role} failed:\n{traceback.format_exc()}", flush=True)
    return f"{clip.name}: {', '.join(done)} in {time.perf_counter() - began:.0f}s"


def extract(clips: list[Path], sources: list[str], workers: int, threads: int) -> None:
    if workers <= 1:
        _limit_threads(threads)
        for clip in clips:
            print(extract_clip(clip, sources), flush=True)
        return
    with ProcessPoolExecutor(workers, initializer=_limit_threads, initargs=(threads,)) as pool:
        futures = [pool.submit(extract_clip, clip, sources) for clip in clips]
        for i, future in enumerate(as_completed(futures), 1):
            print(f"[{i}/{len(clips)}] {future.result()}", flush=True)


# -- pre-training ------------------------------------------------------------------------------
def pretrain(clips: list[Path], sources: list[str]) -> dict:
    """{"onset": {role: {set: weights}}, "note": {role: weights}} fitted on every cell of ``clips``."""
    onset: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    note: dict[str, list] = defaultdict(list)
    for clip in clips:
        for source, role in stems(clip, sources, ONSET_ROLES):
            path = clip / "cache" / f"{source}_{role}_onsets.npz"
            if path.exists():
                table = ensemble.onset_table(clip, source, role)
                for name in ONSET_SETS:
                    X = ensemble.design(ensemble.features_of(table, name), table["base"])
                    onset[role][name].append((X, table["truth"].astype(float)))
            if role in NOTE_ROLES and (clip / "cache" / f"{source}_{role}_notes.npz").exists():
                note[role].append(pitch_head.candidate_design(pitch_head.note_table(clip, source, role)))

    def fit(parts: list) -> np.ndarray:
        X = np.vstack([x for x, _ in parts])
        y = np.concatenate([t for _, t in parts])
        return ensemble.fit_prior_logistic(X, y, np.ones(len(y)), ensemble.hand_prior(X.shape[1]), PRE_LAMBDA)

    return {
        "onset": {role: {name: fit(parts) for name, parts in sets.items()} for role, sets in onset.items()},
        "note": {role: fit(parts) for role, parts in note.items() if parts},
    }


# -- evaluation --------------------------------------------------------------------------------
def evaluate_clip(clip: Path, sources: list[str], priors: dict, lam: float, implicit: float,
                  train_seconds: float, dense: bool) -> tuple[list[dict], list[dict]]:
    onset_rows, note_rows = [], []
    variants = ("current", "reg_models", "reg_all", "pre_models", "pre_all")
    for source, role in stems(clip, sources, ONSET_ROLES):
        if not (clip / "cache" / f"{source}_{role}_onsets.npz").exists():
            continue
        onset_rows += [asdict(r) for r in ensemble.evaluate(
            clip, source, role, BUDGETS, train_seconds, lam, implicit,
            priors["onset"].get(role), variants)]
        if role in NOTE_ROLES and (clip / "cache" / f"{source}_{role}_notes.npz").exists():
            note_rows += pitch_head.evaluate(clip, source, role, BUDGETS, train_seconds, lam, implicit, 0.25,
                                             priors["note"].get(role), dense)
    return onset_rows, note_rows


def evaluate(clips: list[Path], sources: list[str], priors: dict, lam: float, implicit: float,
             train_seconds: float, workers: int, dense: bool = True) -> tuple[list[dict], list[dict]]:
    onset_rows, note_rows = [], []
    with ProcessPoolExecutor(workers, initializer=_limit_threads, initargs=(1,)) as pool:
        futures = [pool.submit(evaluate_clip, c, sources, priors, lam, implicit, train_seconds, dense) for c in clips]
        for future in as_completed(futures):
            a, b = future.result()
            onset_rows += a
            note_rows += b
    return onset_rows, note_rows


def summarize(rows: list[dict], metric: str) -> dict[tuple, float]:
    """Mean ``metric`` per (role, source, variant, budget) over stems."""
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        groups[(r["role"], r["source"], r["variant"], r["budget"])].append(r[metric])
    return {k: float(np.mean(v)) for k, v in groups.items()}


def report(onset_rows: list[dict], note_rows: list[dict], title: str) -> str:
    lines = [f"\n=== {title} ==="]
    for rows, metric, label in ((onset_rows, "onset_f1", "onset F1"), (note_rows, "score", "musical score")):
        means = summarize(rows, metric)
        counts = defaultdict(set)
        for r in rows:
            counts[(r["role"], r["source"])].add(r["clip"])
        variants = list(dict.fromkeys(r["variant"] for r in rows))
        for role, source in sorted({(k[0], k[1]) for k in means}, key=lambda k: (k[1], k[0])):
            lines.append(f"\n{role} ({source}), mean held-out {label} over {len(counts[(role, source)])} songs")
            lines.append(f"  {'variant':17s}" + "".join(f"{b:>7d}" for b in BUDGETS) + "  dense/ceil")
            for v in variants:
                if v.endswith("_dense"):
                    continue
                cells = "".join(f"{means[(role, source, v, b)]:7.3f}" if (role, source, v, b) in means else " " * 7
                                for b in BUDGETS)
                tail = means.get((role, source, f"{v}_dense", -1), means.get((role, source, v, -1)))
                lines.append(f"  {v:17s}{cells}" + (f"  {tail:7.3f}" if tail is not None else ""))
    return "\n".join(lines)


def objective(onset_rows: list[dict], note_rows: list[dict], variant_onset: str, variant_note: str) -> float:
    """Mean over clicked budgets (5..40) of the pre-trained variants' onset F1 and musical score."""
    picked = [r["onset_f1"] for r in onset_rows if r["variant"] == variant_onset and r["budget"] > 0]
    picked += [r["score"] for r in note_rows if r["variant"] == variant_note and r["budget"] > 0]
    return float(np.mean(picked)) if picked else float("nan")


def folds(clips: list[Path], k: int) -> list[list[Path]]:
    order = sorted(clips, key=lambda c: c.name)
    rng = np.random.default_rng(2026)
    rng.shuffle(order)
    return [order[i::k] for i in range(k)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("extract", "cv", "final", "export"))
    parser.add_argument("--prefix", default="atmx_")
    parser.add_argument("--source", default="demucs,clean")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4, help="torch/BLAS threads per extract worker")
    parser.add_argument("--splits", default="train,validation,test", help="which splits to extract")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--lams", default="0.3,1,3,10")
    parser.add_argument("--implicits", default="0,0.25")
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--implicit", type=float, default=0.25)
    parser.add_argument("--train-seconds", type=float, default=20.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    sources = args.source.split(",")
    split = clips_by_split(args.prefix)
    runs = Path("experimental/benchmark/runs")
    runs.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()

    if args.mode == "extract":
        clips = [c for s in args.splits.split(",") for c in split.get(s, [])]
        print(f"extracting {len(clips)} clips with {args.workers} workers", flush=True)
        extract(clips, sources, args.workers, args.threads)

    elif args.mode == "cv":
        results = []
        lines = []
        for lam in [float(x) for x in args.lams.split(",")]:
            for implicit in [float(x) for x in args.implicits.split(",")]:
                onset_rows, note_rows = [], []
                for held in folds(split["train"], args.folds):
                    priors = pretrain([c for c in split["train"] if c not in held], sources)
                    a, b = evaluate(held, sources, priors, lam, implicit, args.train_seconds,
                                    max(1, os.cpu_count() - 2), dense=False)
                    onset_rows += a
                    note_rows += b
                score = {
                    "pre_models": objective(onset_rows, [], "pre_models", ""),
                    "reg_models": objective(onset_rows, [], "reg_models", ""),
                    "pitch_pre": objective([], note_rows, "", "pitch_pre"),
                    "pitch_head": objective([], note_rows, "", "pitch_head"),
                }
                results.append({"lam": lam, "implicit": implicit, **score})
                line = f"lam={lam:<5g} implicit={implicit:<5g} " + "  ".join(f"{k}={v:.4f}" for k, v in score.items())
                print(line, flush=True)
                lines.append(line)
                text = report(onset_rows, note_rows, f"CV over {len(split['train'])} train songs, lam={lam}, implicit={implicit}")
                (runs / f"cv_lam{lam:g}_imp{implicit:g}.txt").write_text(text, encoding="utf-8")
        out = args.out or runs / "crosssong_cv.json"
        out.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print("\n" + "\n".join(lines) + f"\n\nwrote {out}")

    elif args.mode == "final":
        priors = pretrain(split["train"], sources)
        record = {"lam": args.lam, "implicit": args.implicit}
        for name in ("validation", "test"):
            onset_rows, note_rows = evaluate(split.get(name, []), sources, priors, args.lam, args.implicit,
                                             args.train_seconds, max(1, os.cpu_count() - 2))
            text = report(onset_rows, note_rows, f"{name}: prior from {len(split['train'])} train songs, "
                                                 f"lam={args.lam}, implicit={args.implicit}")
            print(text, flush=True)
            (runs / f"crosssong_{name}.txt").write_text(text, encoding="utf-8")
            record[name] = {"onset": onset_rows, "note": note_rows}
        out = args.out or runs / "crosssong_final.json"
        out.write_text(json.dumps(record), encoding="utf-8")
        print(f"\nwrote {out}")

    else:
        priors = pretrain(split["train"] + split.get("validation", []), sources)
        out = args.out or runs / "priors.json"
        plain = {"onset": {r: {s: w.tolist() for s, w in sets.items()} for r, sets in priors["onset"].items()},
                 "note": {r: w.tolist() for r, w in priors["note"].items()}}
        out.write_text(json.dumps(plain, indent=1), encoding="utf-8")
        print(f"wrote {out}")

    print(f"elapsed {time.perf_counter() - began:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
