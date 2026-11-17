"""Isolated canned-model, offline-head and edit-adaptation experiment.

No previous head, prior, threshold selection or benchmark cache is loaded.
Run ``python -m experimental.benchmark.fresh_experiment --help``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import cached_property, lru_cache
import hashlib
import json
import logging
from pathlib import Path
import time

import numpy as np

from core.evaluation import NoteEvent, _match, score_notes
from experimental.benchmark.import_atm import FAMILY_ROUTING_POLICY, ROLES, balanced_index
from experimental.benchmark.small_heads import ARCHITECTURES, SmallHead

LOG = logging.getLogger(__name__)
SCHEMA = 2
FAMILY_ONLY_CODE_SHA256 = "fbad4ececd8fd9667eef4447f73fe851f3c12258bc603c3d051320e1e127aba3"
MODELS = ("basic_pitch", "piano_hr", "adtof")
STEMS = ("bass", "drums", "piano", "guitar", "other", "vocals")
POLICIES = ("frozen", "bias", "local", "last_layer", "anchored")
BUDGETS = (0, 1, 2, 5, 10, 20, 40)
TOLERANCE = 0.05
SUPPORT_SECONDS = 20.0
QUERY_START = SUPPORT_SECONDS + 0.1


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def code_hash() -> str:
    digest = hashlib.sha256()
    for name in ("fresh_experiment.py", "small_heads.py", "import_atm.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    for name in ("evaluation.py", "midi_export.py", "poly_pitch.py", "adtof_drums.py"):
        digest.update((Path(__file__).resolve().parents[2] / "core" / name).read_bytes())
    return digest.hexdigest()


def experiment_manifest(dataset: Path, models: tuple[str, ...]) -> dict:
    if not models or len(set(models)) != len(models) or set(models) - set(MODELS):
        raise ValueError(f"Choose distinct canned models from {MODELS}")
    progress = read_json(dataset / "progress.json")
    verification = read_json(dataset / "verification.json")
    if progress["phase"] != "complete" or verification.get("routing_policy") != FAMILY_ROUTING_POLICY:
        raise ValueError("The batch must finish whole-family routing and verification before import")
    records = balanced_index(dataset)
    if not records or {record["split"] for record in records} != {"train", "validation", "test"}:
        raise ValueError("The experiment needs nonempty train, validation and test partitions")
    if sum(len(record["notes"][role]) for record in records for role in ROLES
           if role in record["notes"]) != verification["note_count"]:
        raise ValueError("Imported family notes do not match the verified dataset note count")
    experiment = {
        "schema": SCHEMA, "code_sha256": code_hash(), "models": models,
        "dataset": str(dataset.resolve()), "records": records, "seed": 2026,
        "support_seconds": SUPPORT_SECONDS, "query_start_seconds": QUERY_START,
        "initialization": "All heads initialized from scratch; no previous fitted artifacts loaded",
        "routing": "Calibration-assisted whole-family assignment, not autonomous family recognition",
        "targets": "Original instrument families plus an independently scored, combined other-stem target",
        "dataset_hashes": {name: hashlib.sha256((dataset / name).read_bytes()).hexdigest()
                           for name in ("pairs.json", "training_pairs.json", "splits.json", "verification.json")},
        "limitations": [
            "The compositions were previously inspected; this is a reset benchmark, not a new independent test.",
            "Canned model pretraining overlap is not established by this experiment.",
            "Raw exact-pitch/class onset matching within 50 ms; durations are not scored.",
            "Heads select canned-model candidates; missing candidates cannot be invented in the query region.",
        ],
    }
    return experiment


def prepare(dataset: Path, output: Path, models: tuple[str, ...]) -> dict:
    if (output / "experiment.json").exists():
        raise FileExistsError("Use a new experiment directory; old learned state cannot be resumed")
    experiment = experiment_manifest(dataset, models)
    write_json(output / "experiment.json", experiment)
    return experiment


def validate_dataset(experiment: dict) -> None:
    dataset = Path(experiment["dataset"])
    for name, expected in experiment["dataset_hashes"].items():
        if hashlib.sha256((dataset / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Dataset {name} changed after experiment preparation")
    group_splits = {}
    for record in experiment["records"]:
        if group_splits.setdefault(record["source_group"], record["split"]) != record["split"]:
            raise ValueError("Source-group leakage in experiment manifest")


def load_experiment(output: Path) -> dict:
    experiment = read_json(output / "experiment.json")
    if experiment["schema"] != SCHEMA or experiment["code_sha256"] != code_hash():
        raise ValueError("Experiment code changed; prepare a new run instead of mixing result versions")
    validate_dataset(experiment)
    return experiment


def upgrade_other(output: Path) -> dict:
    """Revise only the known, untrained family-only run; keep frozen inference."""
    original = read_json(output / "experiment.json")
    if original["schema"] != 1 or original["code_sha256"] != FAMILY_ONLY_CODE_SHA256:
        raise ValueError("Only the registered family-only baseline run can be upgraded")
    if any((output / name).exists() for name in ("heads", "selection.json", "cv_rows.json", "cv")):
        raise ValueError("Cannot revise an experiment after head fitting has started")
    if list(output.glob("*_results.json")) or list(output.glob("*_baselines.json")):
        raise ValueError("Cannot revise an experiment after partition scoring has started")
    backup = output / "experiment.before-other.json"
    if backup.exists():
        raise FileExistsError("The original manifest backup already exists")
    validate_dataset(original)
    revised = experiment_manifest(Path(original["dataset"]), tuple(original["models"]))
    if revised["dataset_hashes"] != original["dataset_hashes"]:
        raise ValueError("Dataset changed; frozen predictions cannot be reused")
    old_records = {record["clip"]: record for record in original["records"]}
    new_records = {record["clip"]: record for record in revised["records"]}
    if set(old_records) != set(new_records):
        raise ValueError("Composition IDs changed during target revision")
    for clip, old in old_records.items():
        new = new_records[clip]
        for field in ("composition_id", "source_group", "source_composition_id", "split", "duration",
                      "family_destinations", "audio", "legacy_family_flagged", "routing_policy"):
            if old[field] != new[field]:
                raise ValueError(f"{clip}: {field} changed during target revision")
        for field in ("notes", "flagged"):
            if old[field] != {role: value for role, value in new[field].items() if role != "other"}:
                raise ValueError(f"{clip}: original family {field} changed")
    count = 0
    for record in revised["records"]:
        directory = Path(record["audio"]["mix"]).parent
        separation = read_json(directory / "separation.json")
        for stem in STEMS:
            for model in revised["models"]:
                path = prediction_path(output, record, stem, model)
                if path.exists():
                    cached = read_json(path)
                    if cached["model"] != model or cached["audio_sha256"] != separation["stem_sha256"][stem]:
                        raise ValueError(f"Frozen prediction provenance changed: {path}")
                    count += 1
    revised["target_revision"] = {
        "previous_schema": original["schema"], "previous_code_sha256": original["code_sha256"],
        "change": "Add the combined labelled other target; canned transcription is unchanged",
        "preserved_frozen_predictions": count, "learned_state_reused": False,
    }
    write_json(backup, original)
    write_json(output / "experiment.json", revised)
    return revised


def prediction_path(output: Path, record: dict, stem: str, model: str) -> Path:
    return output / "frozen" / record["clip"] / f"{stem}_{model}.json"


@lru_cache(maxsize=1)
def piano_model(device: str):
    from piano_transcription_inference import PianoTranscription
    return PianoTranscription(device=device, checkpoint_path=None)


def transcribe(path: Path, model: str) -> list[list[float]]:
    import librosa
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if model in ("basic_pitch", "piano_hr"):
        # The full piano range is intentional, including on misrouted bass/guitar stems.
        from core import poly_pitch
        if model == "piano_hr":
            from experimental.benchmark.ensemble import torch_device
            from piano_transcription_inference import sample_rate
            network = piano_model(torch_device())
            result = network.transcribe(librosa.resample(mono, orig_sr=rate, target_sr=sample_rate), None)
            return [[float(e["onset_time"]), float(e["offset_time"]), int(e["midi_note"]),
                     float(e["velocity"]) / 127] for e in result["est_note_events"]]
        return [list(note) for note in poly_pitch.transcribe_notes(mono, rate, "piano", model)]
    if model == "adtof":
        from core.adtof_drums import (
            CLASS_MIDI, DEFAULT_THRESHOLDS, compute_activations, peak_novelty, pick_peaks,
        )
        novelty = peak_novelty(compute_activations(librosa.resample(
            mono, orig_sr=rate, target_sr=44100)))
        return sorted([[int(frame) / 100, int(frame) / 100 + 0.18, pitch,
                        float(novelty[frame, c])]
                       for c, pitch in enumerate(CLASS_MIDI)
                       for frame in pick_peaks(novelty[:, c], DEFAULT_THRESHOLDS[c])])
    raise ValueError(f"Unknown canned model: {model}")


def extract(output: Path, experiment: dict, limit: int | None = None) -> None:
    records = experiment["records"] if limit is None else experiment["records"][:limit]
    for i, record in enumerate(records, 1):
        directory = Path(record["audio"]["mix"]).parent
        separation = read_json(directory / "separation.json")
        for stem in STEMS:
            for model in experiment["models"]:
                target = prediction_path(output, record, stem, model)
                expected_hash = separation["stem_sha256"][stem]
                if target.exists():
                    cached = read_json(target)
                    if cached["audio_sha256"] != expected_hash or cached["model"] != model:
                        raise ValueError(f"Frozen prediction provenance changed: {target}")
                    continue
                audio = Path(record["audio"]["stems"][stem])
                if hashlib.sha256(audio.read_bytes()).hexdigest() != expected_hash:
                    raise ValueError(f"Separated audio changed: {audio}")
                began = time.perf_counter()
                events = transcribe(audio, model)
                if any(len(row) != 4 or not np.isfinite(row).all() or row[1] < row[0]
                       for row in events):
                    raise ValueError(f"{model} returned invalid note events for {audio}")
                write_json(target, {
                    "model": model, "audio_sha256": expected_hash, "events": events,
                    "runtime_seconds": time.perf_counter() - began,
                })
        LOG.info("Frozen predictions ready: %d/%d %s", i, len(records), record["composition_id"])


def events(rows: list) -> list[NoteEvent]:
    return [NoteEvent(float(row[0]), float(row[1]), int(row[2])) for row in rows]


def matched_labels(candidates: np.ndarray, reference: list) -> np.ndarray:
    labels = np.zeros(len(candidates))
    for _, right in _match(events(reference), events(candidates.tolist()),
                           TOLERANCE, False, TOLERANCE, 0.2, True):
        labels[right] = 1
    return labels


def candidate_pool(channels: dict[str, np.ndarray]) -> np.ndarray:
    by_pitch = defaultdict(list)
    for rows in channels.values():
        for row in rows:
            by_pitch[int(row[2])].append(row)
    candidates = []
    for pitch, rows in sorted(by_pitch.items()):
        group = []
        for row in sorted(rows, key=lambda a: a[0]):
            if group and row[0] - group[0][0] > 0.02:
                candidates.append(max(group, key=lambda a: a[3]))
                group = []
            group.append(row)
        if group:
            candidates.append(max(group, key=lambda a: a[3]))
    return np.asarray(sorted(candidates, key=lambda a: (a[0], a[2])), dtype=float).reshape(-1, 4)


def features(candidates: np.ndarray, channels: dict[str, np.ndarray]) -> np.ndarray:
    columns = [candidates[:, 2] / 127, np.minimum(candidates[:, 1] - candidates[:, 0], 2) / 2]
    for rows in channels.values():
        rows = rows[np.argsort(rows[:, 0])]
        starts = np.searchsorted(rows[:, 0], candidates[:, 0] - 0.1)
        stops = np.searchsorted(rows[:, 0], candidates[:, 0] + 0.1, side="right")
        values = np.zeros((len(candidates), 5))
        for i, candidate in enumerate(candidates):
            if not len(rows):
                continue
            nearby = rows[starts[i]:stops[i]]
            if not len(nearby):
                continue
            same = nearby[nearby[:, 2] == candidate[2]]
            if len(same):
                best = same[np.argmin(np.abs(same[:, 0] - candidate[0]))]
                values[i, :3] = (1, np.clip(best[3], 0, 1), (best[0] - candidate[0]) / 0.1)
            values[i, 3] = np.max(nearby[:, 3])
            lower = nearby[np.isin(candidate[2] - nearby[:, 2], [12, 19, 24])]
            values[i, 4] = np.max(lower[:, 3]) if len(lower) else 0
        columns.extend(values.T)
    return np.column_stack(columns)


@dataclass
class View:
    record: dict
    role: str
    config: str
    channels: dict[str, np.ndarray]
    candidates: np.ndarray
    X: np.ndarray
    reference: list

    @cached_property
    def y(self) -> np.ndarray:
        return matched_labels(self.candidates, self.reference)


def input_configs(role: str, models: list[str]) -> dict[str, tuple[str, ...]]:
    relevant = tuple(model for model in models if (model == "adtof") == (role == "drums"))
    if not relevant:
        return {}
    return {**{f"single_{model}": (model,) for model in relevant},
            "multi_own": relevant, "multi_cross": relevant}


def target_stem(record: dict, role: str) -> str:
    return "other" if role == "other" else record["family_destinations"][role]


def load_views(output: Path, experiment: dict, records: list[dict]) -> list[View]:
    views = []
    for record in records:
        for role, reference in record["notes"].items():
            if role == "drums":
                reference = drum_reference(reference)
            if record["duration"] <= QUERY_START:
                raise ValueError(f"{record['clip']}: clip is too short for the fixed edit/query split")
            for config, models in input_configs(role, experiment["models"]).items():
                stems = STEMS if config == "multi_cross" else (target_stem(record, role),)
                channels = {}
                for stem in stems:
                    for model in models:
                        # Use positional stem names so feature meaning is stable between songs.
                        name = f"{stem if config == 'multi_cross' else 'own'}:{model}"
                        channels[name] = np.asarray(read_json(
                            prediction_path(output, record, stem, model))["events"], dtype=float).reshape(-1, 4)
                candidates = candidate_pool(channels)
                views.append(View(record, role, config, channels, candidates,
                                  features(candidates, channels), reference))
    return views


def drum_reference(reference: list) -> list:
    from core.adtof_drums import CLASS_MIDI, drum_class

    canonical = {}
    for row in reference:
        label = drum_class(int(row[2]))
        if label is not None:
            mapped = [*row[:2], CLASS_MIDI[label], row[3]]
            key = (float(row[0]), CLASS_MIDI[label])
            if key not in canonical or canonical[key][3] < row[3]:
                canonical[key] = mapped
    return sorted(canonical.values())


def score_view(view: View, predictions: np.ndarray) -> dict:
    reference = [row for row in view.reference if row[0] >= QUERY_START]
    predicted = [row for row in predictions if row[0] >= QUERY_START]
    result = asdict(score_notes(reference, predicted))
    result["candidate_recall"] = score_notes(
        reference, [row for row in view.candidates if row[0] >= QUERY_START]).recall
    return result


def result_row(view: View, predictions: np.ndarray, **metadata) -> dict:
    from core.adtof_drums import drum_class

    return {
        "clip": view.record["clip"], "source_group": view.record["source_group"],
        "split": view.record["split"], "role": view.role, "config": view.config,
        "target_kind": "routed_stem" if view.role == "other" else "instrument_family",
        "target_stem": target_stem(view.record, view.role),
        "target_families": view.record["other_families"] if view.role == "other" else [view.role],
        "flagged": view.record["flagged"][view.role], **metadata, **score_view(view, predictions),
        "unmapped_drum_reference_notes": sum(
            drum_class(int(row[2])) is None and row[0] >= QUERY_START
            for row in view.record["notes"][view.role]) if view.role == "drums" else 0,
    }


def edit_stream(view: View, head: SmallHead) -> list[tuple[np.ndarray, float]]:
    support_candidates = view.candidates[view.candidates[:, 0] < SUPPORT_SECONDS]
    support_X = features(support_candidates, view.channels)
    predicted = support_candidates[head.probabilities(support_X) >= 0.5]
    reference = [row for row in view.reference if row[0] < SUPPORT_SECONDS]
    matches = _match(events(reference), events(predicted.tolist()),
                     TOLERANCE, False, TOLERANCE, 0.2, True)
    ref_used = {left for left, _ in matches}
    pred_used = {right for _, right in matches}
    edits = [(row.copy(), 0.0) for i, row in enumerate(predicted) if i not in pred_used]
    edits.extend((np.array([*row[:3], row[3] / 127]), 1.0)
                 for i, row in enumerate(reference) if i not in ref_used)
    return sorted(edits, key=lambda edit: (edit[0][0], edit[0][2], edit[1]))


def apply_overrides(predictions: np.ndarray, edits: list[tuple[np.ndarray, float]]) -> np.ndarray:
    resolved = {(float(row[0]), int(row[2])): row for row in predictions}
    for note, keep in edits:
        key = (float(note[0]), int(note[2]))
        if keep:
            resolved[key] = note
        else:
            resolved.pop(key, None)
    return np.asarray(sorted(resolved.values(), key=lambda row: (row[0], row[2]))).reshape(-1, 4)


def evaluate_head(view: View, head: SmallHead, architecture: str, policies: tuple[str, ...],
                  lams: tuple[float, ...], budgets: tuple[int, ...] = BUDGETS) -> list[dict]:
    stream = edit_stream(view, head)
    zero = score_view(view, view.candidates[head.probabilities(view.X) >= 0.5])["f1"]
    rows = []
    for policy in policies:
        for lam in (lams if policy != "frozen" else (0.0,)):
            for budget in budgets:
                selected = stream[:budget]
                edits_X = features(np.array([row for row, _ in selected]).reshape(-1, 4), view.channels)
                edits_y = np.array([label for _, label in selected])
                began = time.perf_counter()
                probabilities = head.probabilities(view.X, edits_X, edits_y, policy=policy,
                                                   lam=lam if lam else 1)
                predictions = apply_overrides(view.candidates[probabilities >= 0.5], selected)
                resolved = {(float(note[0]), int(note[2])) for note in predictions}
                edits_preserved = all(
                    ((float(note[0]), int(note[2])) in resolved) == bool(keep) for note, keep in selected)
                if not edits_preserved:
                    raise ValueError("An adaptation policy overwrote an explicit user edit")
                row = result_row(
                    view, predictions, architecture=architecture, policy=policy, lam=lam, budget=budget,
                    clicks=len(selected), parameters=len(head.pack()),
                    fit_converged=head.converged,
                    edits_preserved=edits_preserved,
                    update_seconds=time.perf_counter() - began, zero_f1=zero,
                    # Author/reject overrides are kept separate from generated notes.
                    explicit_overrides=[{"onset": float(note[0]), "pitch": int(note[2]), "keep": bool(y)}
                                        for note, y in selected],
                )
                row["gain"] = row["f1"] - zero
                rows.append(row)
    return rows


def fit_head(views: list[View], architecture: str, iterations: int) -> SmallHead:
    if not views or not any(len(view.X) for view in views):
        raise ValueError("No frozen-model candidates in this training partition")
    parts = [view for view in views if len(view.X)]
    source_counts = defaultdict(int)
    for view in parts:
        source_counts[view.record["source_group"]] += 1
    weights = np.concatenate([
        np.full(len(view.X), 1 / (len(view.X) * source_counts[view.record["source_group"]]))
        for view in parts
    ])
    weights *= len(weights) / weights.sum()
    return SmallHead.fit(np.vstack([view.X for view in parts]),
                         np.concatenate([view.y for view in parts]), architecture,
                         iterations=iterations, row_weights=weights)


def summarize(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    summaries = []
    for key, selected in sorted(groups.items()):
        by_source = defaultdict(list)
        for row in selected:
            by_source[row["source_group"]].append(row)
        source_f1 = np.array([np.mean([row["f1"] for row in group])
                              for group in by_source.values()])
        gains = np.array([np.mean([row.get("gain", 0) for row in group])
                          for group in by_source.values()])
        rng = np.random.default_rng(2026)
        bootstrap = rng.choice(gains, (1000, len(gains))).mean(axis=1)
        summaries.append({
            **dict(zip(keys, key)), "source_groups": len(by_source), "views": len(selected),
            "f1": float(source_f1.mean()), "gain": float(gains.mean()),
            "precision": float(np.mean([
                np.mean([row["precision"] for row in group]) for group in by_source.values()])),
            "recall": float(np.mean([
                np.mean([row["recall"] for row in group]) for group in by_source.values()])),
            "fit_converged_fraction": float(np.mean([row.get("fit_converged", True) for row in selected])),
            "gain_bootstrap95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "regression_fraction": float(np.mean(gains < -1e-9)),
            "p10_gain": float(np.quantile(gains, 0.1)), "worst_gain": float(gains.min()),
            "candidate_recall": float(np.mean([
                np.mean([row["candidate_recall"] for row in group]) for group in by_source.values()])),
            "false_positive_notes": sum(row["false_positives"] for row in selected),
            "empty_reference_views": sum(row["true_positives"] + row["false_negatives"] == 0
                                         for row in selected),
            "empty_reference_false_positives": sum(
                row["false_positives"] for row in selected
                if row["true_positives"] + row["false_negatives"] == 0),
            "mean_update_seconds": float(np.mean([row.get("update_seconds", 0) for row in selected])),
        })
    return summaries


def baselines(output: Path, experiment: dict, records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        for role, reference in record["notes"].items():
            if record["duration"] <= QUERY_START:
                raise ValueError(f"{record['clip']}: clip is too short for the fixed edit/query split")
            for config, models in input_configs(role, experiment["models"]).items():
                if not config.startswith("single_"):
                    continue
                model = models[0]
                prediction = read_json(prediction_path(
                    output, record, target_stem(record, role), model))
                raw = np.asarray(prediction["events"], dtype=float).reshape(-1, 4)
                channels = {f"own:{model}": raw}
                view = View(record, role, config, channels, candidate_pool(channels),
                            np.empty((0, 0)), drum_reference(reference) if role == "drums" else reference)
                rows.append(result_row(view, raw, system=model,
                                       inference_seconds=prediction["runtime_seconds"]))
    return rows


def baseline_report(output: Path, experiment: dict, split: str, allow_test: bool) -> None:
    if split == "test" and not allow_test:
        raise ValueError("Test evaluation is locked; pass --unlock-test after freezing the experiment")
    target = output / f"{split}_canned_baselines.json"
    if target.exists():
        raise FileExistsError("Canned baselines were already evaluated for this partition")
    records = [record for record in experiment["records"] if record["split"] == split]
    rows = baselines(output, experiment, records)
    write_json(target, {
        "split": split, "rows": rows, "summary": summarize(rows, ("role", "system")),
        "metric": "Exact pitch/drum class, one-to-one raw onset matching <=50 ms; offsets ignored",
    })


def select_configurations(cv_rows: list[dict]) -> dict:
    summaries = summarize(cv_rows, ("role", "config", "architecture", "policy", "lam", "budget"))
    selected = {}
    for role in sorted({row["role"] for row in cv_rows}):
        frozen = [row for row in summaries if row["role"] == role
                  and row["policy"] == "frozen" and row["budget"] == 0]
        best = max(frozen, key=lambda row: (row["f1"], row["config"], row["architecture"]))
        variants = defaultdict(list)
        for row in summaries:
            if (row["role"] == role and row["config"] == best["config"]
                    and row["architecture"] == best["architecture"] and row["budget"] in (1, 2, 5, 10)):
                objective = row["gain"] - 0.02 * row["regression_fraction"] + 0.1 * row["p10_gain"]
                variants[(row["policy"], row["lam"])].append(objective)
        policy, lam = max(variants, key=lambda key: (np.mean(variants[key]), key[0] == "frozen"))
        selected[role] = {
            "config": best["config"], "architecture": best["architecture"],
            "policy": policy, "lam": lam, "cv_zero_f1": best["f1"],
            "adaptation_objective": float(np.mean(variants[(policy, lam)])),
        }
    return {"selection": selected, "summaries": summaries}


def baseline_selection(rows: list[dict]) -> dict[str, str]:
    summaries = summarize(rows, ("role", "system"))
    return {
        role: max((row for row in summaries if row["role"] == role),
                  key=lambda row: (row["f1"], row["system"]))["system"]
        for role in sorted({row["role"] for row in rows})
    }


def head_improvements(head_rows: list[dict], baseline_rows: list[dict],
                      systems: dict[str, str]) -> list[dict]:
    baseline = {(row["clip"], row["role"]): row for row in baseline_rows
                if systems[row["role"]] == row["system"]}
    paired = []
    for row in head_rows:
        if row["policy"] == "frozen" and row["budget"] == 0:
            compared = baseline[(row["clip"], row["role"])]
            paired.append({**row, "gain": row["f1"] - compared["f1"],
                           "canned_system": compared["system"], "canned_f1": compared["f1"]})
    return paired


def train(output: Path, experiment: dict, architectures: tuple[str, ...],
          iterations: int, folds: int, lams: tuple[float, ...]) -> None:
    if (output / "selection.json").exists():
        raise FileExistsError("Training already completed; prepare a new run for another experiment")
    records = [record for record in experiment["records"] if record["split"] == "train"]
    groups = sorted({record["source_group"] for record in records},
                    key=lambda group: hashlib.sha256(f"2026:{group}".encode()).digest())
    if folds < 2 or len(groups) < folds:
        raise ValueError("Need at least one source group per composition fold and two folds")
    fold_of = {group: i % folds for i, group in enumerate(groups)}
    views = load_views(output, experiment, records)
    baseline_rows = baselines(output, experiment, records)
    write_json(output / "canned_training_baselines.json", {
        "rows": baseline_rows, "summary": summarize(baseline_rows, ("role", "system")),
    })
    rows = []
    for role, config in sorted({(view.role, view.config) for view in views}):
        selected_views = [view for view in views if (view.role, view.config) == (role, config)]
        for architecture in architectures:
            if architecture not in ARCHITECTURES:
                raise ValueError(f"Unknown architecture: {architecture}")
            for fold in range(folds):
                fit_views = [view for view in selected_views
                             if fold_of[view.record["source_group"]] != fold]
                held_views = [view for view in selected_views
                              if fold_of[view.record["source_group"]] == fold]
                if not fit_views or not held_views:
                    raise ValueError(f"{role}/{config}: missing family examples in a composition fold")
                head = fit_head(fit_views, architecture, iterations)
                for view in held_views:
                    rows.extend({**row, "fold": fold, "fit_converged": head.converged}
                                for row in evaluate_head(view, head, architecture, POLICIES, lams))
                write_json(output / "cv_rows.json", rows)
                LOG.info("CV complete: %s %s %s fold %d/%d", role, config, architecture, fold + 1, folds)
    selection = select_configurations(rows)
    selection.update({
        "canned_baseline_by_role": baseline_selection(baseline_rows),
        "fold_of_source_group": fold_of, "initialization": experiment["initialization"],
        "selection_scope": "Training source-group CV only; validation and test not read during fitting",
        "architectures": architectures, "iterations": iterations, "adaptation_lambdas": lams,
        "criterion": "Best zero-edit raw F1, then mean gain - .02 regression_fraction + .1 p10_gain at 1/2/5/10 edits",
    })
    write_json(output / "selection.json", selection)


def evaluate(output: Path, experiment: dict, split: str, allow_test: bool) -> None:
    if split not in ("validation", "test"):
        raise ValueError("Head evaluation must use validation or test, not the fitting partition")
    if split == "test" and not allow_test:
        raise ValueError("Test evaluation is locked; pass --unlock-test after freezing the experiment")
    target = output / f"{split}_results.json"
    if target.exists():
        raise FileExistsError("This partition was already evaluated; do not overwrite inspected results")
    selection = read_json(output / "selection.json")
    train_records = [record for record in experiment["records"] if record["split"] == "train"]
    held_records = [record for record in experiment["records"] if record["split"] == split]
    train_views = load_views(output, experiment, train_records)
    held_views = load_views(output, experiment, held_records)
    baseline_rows = baselines(output, experiment, held_records)
    rows = []
    for role, config in selection["selection"].items():
        head = fit_head([view for view in train_views if view.role == role
                         and view.config == config["config"]],
                        config["architecture"], selection["iterations"])
        write_json(output / "heads" / f"{role}.json", {
            **head.state(), "fit_source_groups": sorted({r["source_group"] for r in train_records}),
            "architecture": config["architecture"], "config": config["config"],
        })
        for view in held_views:
            if view.role == role and view.config == config["config"]:
                policies = tuple(dict.fromkeys(("frozen", config["policy"])))
                rows.extend(evaluate_head(view, head, config["architecture"], policies,
                                          (config["lam"],) if config["lam"] else (1.0,)))
    write_json(target, {
        "scope": f"{split}; reset-model evaluation on previously seen composition collection",
        "baselines": baseline_rows, "head_rows": rows,
        "baseline_summary": summarize(baseline_rows, ("role", "system")),
        "baseline_summary_by_quality": summarize(baseline_rows, ("role", "system", "flagged")),
        "head_vs_canned": summarize(
            head_improvements(rows, baseline_rows, selection["canned_baseline_by_role"]),
            ("role", "config", "architecture", "canned_system")),
        "head_summary": summarize(rows, ("role", "config", "architecture", "policy", "lam", "budget")),
        "head_summary_by_quality": summarize(
            rows, ("role", "config", "architecture", "policy", "lam", "budget", "flagged")),
        "metric": "Exact pitch/drum class, one-to-one raw onset matching <=50 ms; offsets ignored",
    })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "upgrade-other", "extract", "baseline", "train", "evaluate"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--architectures", default=",".join(ARCHITECTURES))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--adaptation-lambdas", default="1,10,100")
    parser.add_argument("--limit", type=int, help="Extract only the first N clips; incomplete runs cannot train")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--unlock-test", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.command == "prepare":
            if args.dataset is None:
                raise ValueError("--dataset is required for prepare")
            experiment = prepare(args.dataset, args.out, tuple(args.models.split(",")))
            LOG.info("Prepared %d variants in a new, isolated experiment", len(experiment["records"]))
        elif args.command == "upgrade-other":
            experiment = upgrade_other(args.out)
            LOG.info("Added other targets; preserved %d frozen predictions",
                     experiment["target_revision"]["preserved_frozen_predictions"])
        else:
            experiment = load_experiment(args.out)
            if args.command == "extract":
                if args.limit is not None and args.limit < 1:
                    raise ValueError("--limit must be positive")
                extract(args.out, experiment, args.limit)
            elif args.command == "train":
                lams = tuple(float(value) for value in args.adaptation_lambdas.split(","))
                if not lams or any(not np.isfinite(value) or value <= 0 for value in lams):
                    raise ValueError("Adaptation regularizers must be finite and positive")
                train(args.out, experiment, tuple(args.architectures.split(",")),
                      args.iterations, args.folds, lams)
            elif args.command == "baseline":
                baseline_report(args.out, experiment, args.split, args.unlock_test)
            else:
                evaluate(args.out, experiment, args.split, args.unlock_test)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        LOG.error("Experiment stopped: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
