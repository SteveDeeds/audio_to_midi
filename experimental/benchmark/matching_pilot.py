"""Train-only, review-gated MIDI-to-separated-stem matching on existing audio."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import inspect
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
from pathlib import Path
import secrets
import threading
from urllib.parse import unquote, urlparse

import numpy as np
from scipy.optimize import nnls
from scipy.signal import stft
import soundfile as sf

from core.adtof_drums import CLASS_MIDI, drum_class
from core.evaluation import NoteEvent, _match
from core.midi_export import load_midi_notes
from experimental.benchmark.fresh_experiment import (
    prediction_path, read_json, write_json,
)
from experimental.benchmark.import_atm import ROLES, balanced_index

LOG = logging.getLogger(__name__)
STEMS = ("bass", "drums", "piano", "guitar", "other", "vocals")
SCHEMA = 1
WINDOW_SECONDS = 0.25
NFFT = 4096
MIN_SCORE = 0.30
MIN_MARGIN = 0.15
ASSETS = Path(__file__).with_name("matching_review")
LEGACY_CODE_SHA256 = "9ed7606d00802eb850dc95bd180b03c374655ca8bc737495bfe49098b12808f7"
LEGACY_EVIDENCE_SHA256 = "f52cc8e524c4f17e20d09ceff8a581d27d42509aaf36524542af9af55d8f4c8e"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def code_hash() -> str:
    return digest(Path(__file__))


def evidence_hash() -> str:
    functions = (select_records, spectrum, spectral_evidence, propose, original_notes,
                 cached_support, analyze_song)
    source = "".join(inspect.getsource(function) for function in functions)
    constants = json.dumps([SCHEMA, STEMS, WINDOW_SECONDS, NFFT, MIN_SCORE, MIN_MARGIN],
                           separators=(",", ":"))
    return hashlib.sha256((constants + source).encode()).hexdigest()


def report_hash(report: dict) -> str:
    return hashlib.sha256(json.dumps(
        report, sort_keys=True, allow_nan=False, separators=(",", ":"),
    ).encode()).hexdigest()


def select_records(records: list[dict], songs: int) -> list[dict]:
    if songs < 1:
        raise ValueError("--songs must be positive")
    groups: dict[str, str] = {}
    by_group: dict[str, list[dict]] = {}
    for record in records:
        if groups.setdefault(record["source_group"], record["split"]) != record["split"]:
            raise ValueError("Source-group leakage in dataset")
        if record["split"] == "train":
            by_group.setdefault(record["source_group"], []).append(record)
    originals = []
    for items in by_group.values():
        original = [item for item in items if item["composition_id"].endswith("_base")]
        if len(original) != 1:
            raise ValueError("Pilot requires exactly one original-pitch _base variant per group")
        originals.append(original[0])
    if songs > len(originals):
        raise ValueError(f"Only {len(originals)} training source groups available")
    originals.sort(key=lambda item: (
        sum(item["flagged"].values()),
        hashlib.sha256(item["source_group"].encode()).hexdigest(),
    ))
    selected = []
    while len(selected) < songs:
        selected.append(originals.pop())
        if originals and len(selected) < songs:
            selected.append(originals.pop(0))
    return selected


def spectrum(audio: np.ndarray, rate: int) -> np.ndarray:
    if audio.ndim != 2 or audio.shape[0] == 0 or not np.isfinite(audio).all():
        raise ValueError("Audio window must be nonempty, finite and frames-by-channels")
    padded = np.pad(audio, ((0, max(0, NFFT - len(audio))), (0, 0)))
    _, _, values = stft(
        padded.T, fs=rate, nperseg=NFFT, noverlap=NFFT // 2,
        boundary=None, padded=False,
    )
    return np.sqrt(np.mean(np.abs(values) ** 2, axis=(0, 2)))


def spectral_evidence(clean: dict[str, np.ndarray], stems: dict[str, np.ndarray],
                      family: str, pitch: int, rate: int) -> dict:
    """Local nonnegative spectral fit; never a probability or note attribution."""
    spectra = {name: spectrum(audio, rate) for name, audio in clean.items()}
    target = spectra[family]
    frequencies = np.fft.rfftfreq(NFFT, 1 / rate)
    if family == "drums":
        mask = (frequencies >= 80) & (frequencies <= min(12000, rate / 2))
    else:
        fundamental = 440 * 2 ** ((pitch - 69) / 12)
        mask = np.zeros(len(frequencies), dtype=bool)
        for harmonic in range(1, 9):
            centre = fundamental * harmonic
            width = max(rate / NFFT, centre * (2 ** (0.5 / 12) - 1))
            mask |= np.abs(frequencies - centre) <= width
        mask &= frequencies > 0
    total = sum(spectra.values())
    whitening = 1 / np.sqrt(np.maximum(total[mask], 1e-8))
    names = list(clean)
    design = np.stack([spectra[name][mask] for name in names], axis=1) * whitening[:, None]
    norms = np.linalg.norm(design, axis=0)
    active = norms > 1e-10
    template = design[:, names.index(family)]
    template_norm = float(np.linalg.norm(template))
    rms = float(np.sqrt(np.mean(clean[family] ** 2)))
    rms_db = 20 * math.log10(max(rms, 1e-12))
    competitors = {
        name: float(np.dot(template, design[:, i]) / (template_norm * norms[i]))
        for i, name in enumerate(names)
        if name != family and template_norm > 0 and norms[i] > 1e-10
    }
    result = {}
    for stem, audio in stems.items():
        measured = spectrum(audio, rate)[mask] * whitening
        norm = float(np.linalg.norm(measured))
        similarity = (float(np.dot(template, measured) / (template_norm * norm))
                      if norm > 1e-10 and template_norm > 1e-10 else 0.0)
        coefficients = np.zeros(len(names))
        residual = 1.0
        if active.any() and norm > 1e-10:
            fit, error = nnls(design[:, active], measured, maxiter=100)
            coefficients[active] = fit
            residual = float(error / norm)
        retention = float(coefficients[names.index(family)])
        result[stem] = {
            "score": float(np.clip(min(retention, 1.0) * similarity, 0, 1)),
            "spectral_similarity": float(np.clip(similarity, 0, 1)),
            "estimated_retention": retention,
            "fit_residual": residual,
            "window_rms_dbfs": 20 * math.log10(max(
                float(np.sqrt(np.mean(audio ** 2))), 1e-12)),
        }
    return {"stems": result, "clean_rms_dbfs": rms_db,
            "competing_family_similarity": competitors}


def propose(evidence: dict, overlap: bool) -> tuple[list[str], list[str]]:
    ranked = sorted(STEMS, key=lambda stem: -evidence["stems"][stem]["score"])
    best = evidence["stems"][ranked[0]]["score"]
    candidates = [stem for stem in ranked
                  if evidence["stems"][stem]["score"] >= MIN_SCORE
                  and evidence["stems"][stem]["score"] >= best * 0.5]
    reasons = []
    if evidence["clean_rms_dbfs"] < -60:
        reasons.append("quiet_clean_reference")
    if best < MIN_SCORE:
        reasons.append("weak_spectral_evidence")
    if best - evidence["stems"][ranked[1]]["score"] < MIN_MARGIN:
        reasons.append("close_destinations")
    if len(candidates) > 1:
        reasons.append("possible_multi_stem_leakage")
    if overlap:
        reasons.append("overlapping_same_family_notes_or_release")
    if max(evidence["competing_family_similarity"].values(), default=0) >= 0.85:
        reasons.append("confusable_clean_families")
    return candidates, reasons


def original_notes(dataset: Path, record: dict, pairs: dict) -> list[dict]:
    notes = []
    for family in ROLES:
        if family not in record["notes"]:
            continue
        pair = pairs[(record["composition_id"], family)]
        rows = load_midi_notes(str(dataset / pair["midi"]))
        if len(rows) != pair["note_count"]:
            raise ValueError(f"{record['composition_id']}: MIDI note count mismatch")
        for i, (start, end, pitch, velocity, channel) in enumerate(rows):
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                raise ValueError("Invalid source MIDI note")
            notes.append({
                "id": f"{record['clip']}:{family}:{i}",
                "family": family, "onset": start, "offset": end,
                "pitch": pitch, "velocity": velocity, "channel": channel,
                "legacy_stem": record["family_destinations"][family],
            })
    return sorted(notes, key=lambda note: (note["onset"], note["family"], note["id"]))


def cached_support(output: Path | None, record: dict, notes: list[dict],
                   hashes: dict[str, str]) -> tuple[dict, list[dict]]:
    support = {note["id"]: {stem: [] for stem in STEMS} for note in notes}
    provenance = []
    if output is None:
        return support, [{"status": "not_requested"}]
    for stem in STEMS:
        for model in ("basic_pitch", "piano_hr", "adtof"):
            path = prediction_path(output, record, stem, model)
            if not path.exists():
                LOG.warning("No frozen evidence: %s", path)
                provenance.append({"stem": stem, "model": model, "status": "missing"})
                continue
            cached = read_json(path)
            if cached["model"] != model or cached["audio_sha256"] != hashes[stem]:
                raise ValueError(f"Cached prediction/audio provenance mismatch: {path}")
            events = np.asarray(cached["events"], dtype=float).reshape(-1, 4)
            if not np.isfinite(events).all() or np.any(events[:, 1] < events[:, 0]):
                raise ValueError(f"Invalid cached events: {path}")
            eligible, reference = [], []
            for note in notes:
                if (note["family"] == "drums") != (model == "adtof"):
                    continue
                pitch = note["pitch"]
                if model == "adtof":
                    cls = drum_class(pitch)
                    if cls is None:
                        raise ValueError(f"Unsupported reference drum pitch {pitch}")
                    pitch = CLASS_MIDI[cls]
                eligible.append(note)
                reference.append(NoteEvent(note["onset"], note["offset"], pitch))
            predictions = [NoteEvent(float(row[0]), float(row[1]), int(row[2]))
                           for row in events]
            for left, right in _match(reference, predictions, .05, False, .05, .2, True):
                note = eligible[left]
                support[note["id"]][stem].append({
                    "model": model, "onset_error_seconds": predictions[right].onset - note["onset"],
                    "strength": float(events[right, 3]),
                })
            provenance.append({"stem": stem, "model": model, "status": "available",
                               "path": str(path.resolve()), "sha256": digest(path)})
    return support, provenance


def analyze_song(dataset: Path, record: dict, pairs: dict, predictions: Path | None) -> dict:
    directory = dataset / record["composition_id"]
    clean_render = read_json(directory / "clean-render.json")
    separation = read_json(directory / "separation.json")
    paths = {"mix": directory / "mix.wav"}
    expected = {"mix": separation["mix_sha256"]}
    for family, path in record["audio"]["clean"].items():
        paths[f"clean_{family}"] = Path(path)
        expected[f"clean_{family}"] = clean_render["audio_sha256"][family]
        pair = pairs[(record["composition_id"], family)]
        if digest(dataset / pair["midi"]) != clean_render["midi_sha256"][family]:
            raise ValueError(f"{record['composition_id']}: source MIDI changed")
    for stem in STEMS:
        paths[stem] = Path(record["audio"]["stems"][stem])
        expected[stem] = separation["stem_sha256"][stem]
    for name, path in paths.items():
        if digest(path) != expected[name]:
            raise ValueError(f"Audio hash mismatch: {path}")
    notes = original_notes(dataset, record, pairs)
    support, caches = cached_support(predictions, record, notes, separation["stem_sha256"])
    with ExitStack() as stack:
        files = {name: stack.enter_context(sf.SoundFile(path)) for name, path in paths.items()}
        rate, frames = files["mix"].samplerate, len(files["mix"])
        if any(file.samplerate != rate or len(file) != frames for file in files.values()):
            raise ValueError(f"{record['composition_id']}: audio alignment mismatch")
        duration = frames / rate
        for note in notes:
            if note["onset"] >= duration:
                raise ValueError("Source MIDI onset lies outside audio")
            start = max(0, round((note["onset"] - .01) * rate))
            stop = min(frames, round((note["onset"] + WINDOW_SECONDS) * rate))
            windows = {}
            for name, file in files.items():
                if name == "mix":
                    continue
                file.seek(start)
                windows[name] = file.read(stop - start, dtype="float64", always_2d=True)
            clean = {family: windows[f"clean_{family}"] for family in record["audio"]["clean"]}
            evidence = spectral_evidence(clean, {stem: windows[stem] for stem in STEMS},
                                         note["family"], note["pitch"], rate)
            overlap = any(
                other["id"] != note["id"] and other["family"] == note["family"]
                and other["onset"] < stop / rate and other["offset"] + .15 > start / rate
                for other in notes
            )
            candidates, reasons = propose(evidence, overlap)
            note.update({
                "window": [start / rate, stop / rate],
                "evidence": evidence, "model_support": support[note["id"]],
                "suggested_stems": candidates, "review_reasons": reasons,
                "status": "unreviewed",
            })
    rights = [pairs[(record["composition_id"], family)].get("license")
              for family in record["notes"] if family in ROLES]
    return {
        "clip": record["clip"], "composition_id": record["composition_id"],
        "source_group": record["source_group"], "split": record["split"],
        "duration": duration, "sample_rate": rate,
        "audio": {name: str(path.resolve()) for name, path in paths.items()},
        "audio_sha256": expected, "cached_evidence": caches, "notes": notes,
        "license": rights, "rights_review_required": True,
        "provenance": sorted({
            pairs[(record["composition_id"], family)]["provenance"]
            for family in record["notes"] if family in ROLES
        }),
        "source_midi": {
            family: str((dataset / pairs[(record["composition_id"], family)]["midi"]).resolve())
            for family in record["notes"] if family in ROLES
        },
        "source_midi_sha256": {
            family: clean_render["midi_sha256"][family]
            for family in record["notes"] if family in ROLES
        },
    }


def prepare(dataset: Path, output: Path, songs: int, predictions: Path | None = None) -> dict:
    dataset, output = dataset.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty pilot directory; previous evidence is immutable")
    verification = read_json(dataset / "verification.json")
    if (verification["status"] != "complete"
            or read_json(dataset / "progress.json")["phase"] != "complete"
            or read_json(dataset / "failures.json")):
        raise ValueError("Dataset must have completed verification and zero pipeline failures")
    records = select_records(balanced_index(dataset), songs)
    pairs = {(pair["composition_id"], pair["instrument"]): pair
             for pair in read_json(dataset / "pairs.json")}
    if predictions is not None:
        previous = read_json(predictions / "experiment.json")
        if Path(previous["dataset"]).resolve() != dataset:
            raise ValueError("Frozen predictions come from a different dataset")
        for name, expected in previous["dataset_hashes"].items():
            if digest(dataset / name) != expected:
                raise ValueError(f"Frozen prediction dataset metadata changed: {name}")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": SCHEMA, "code_sha256": code_hash(), "dataset": str(dataset),
        "dataset_sha256": {name: digest(dataset / name) for name in (
            "pairs.json", "training_pairs.json", "splits.json", "verification.json")},
        "status": "in_progress", "split": "train", "records": [],
        "method": {
            "window_seconds": WINDOW_SECONDS, "nfft": NFFT,
            "score_threshold": MIN_SCORE, "margin_threshold": MIN_MARGIN,
            "description": "Stereo-energy spectra; local nonnegative fit against all clean families. "
                           "Melodic fits use eight pitch harmonics; drum fits use 80Hz-12kHz.",
            "scores_are_probabilities": False,
            "cached_models": "Optional frozen canned predictions, one-to-one pitch/class onset matches within 50ms. "
                             "No learned heads or tuned thresholds are loaded.",
        },
        "limitations": [
            "Family references contain chords, overlapping notes and release tails, not isolated note truth.",
            "Magnitude spectral fits cannot prove source attribution; octave/unison collisions and model mistakes remain.",
            "Original MIDI offsets/velocities are synthesis metadata, not measured separated-audio releases/loudness.",
            "Pilot selection alternates more and less flagged original-pitch training sources; not an accuracy benchmark.",
            "All source notes remain preserved; zero or multiple destinations are allowed only after review.",
            "Underlying composition/arrangement rights still require review; no new renders or training are performed.",
        ],
    }
    write_json(output / "report.json", report)
    for i, record in enumerate(records, 1):
        LOG.info("Analyzing %d/%d: %s", i, len(records), record["composition_id"])
        report["records"].append(analyze_song(dataset, record, pairs, predictions))
        write_json(output / "report.json", report)
    notes = [note for record in report["records"] for note in record["notes"]]
    report["status"] = "ready_for_review"
    report["summary"] = {
        "source_groups": len(records), "notes": len(notes),
        "reviewed_notes": 0, "hard_training_labels": 0,
        "no_suggested_destination": sum(not note["suggested_stems"] for note in notes),
        "multiple_suggested_destinations": sum(len(note["suggested_stems"]) > 1 for note in notes),
        "suggestion_differs_from_legacy": sum(
            bool(note["suggested_stems"]) and note["suggested_stems"][0] != note["legacy_stem"]
            for note in notes),
        "review_reasons": dict(Counter(reason for note in notes for reason in note["review_reasons"])),
    }
    write_json(output / "report.json", report)
    for name in ("index.html", "review.js"):
        (output / name).write_bytes((ASSETS / name).read_bytes())
    write_json(output / "reviews.json", empty_reviews(report))
    LOG.info("Ready for review: %s", report["summary"])
    return report


def empty_reviews(report: dict) -> dict:
    return {"schema": SCHEMA, "report_sha256": report_hash(report), "revision": 0, "decisions": {}}


def validate_reviews(report: dict, reviews: dict) -> dict:
    if not isinstance(reviews, dict):
        raise ValueError("Review file must be an object")
    if reviews.get("schema") != SCHEMA or reviews.get("report_sha256") != report_hash(report):
        raise ValueError("Review file belongs to a different report/schema")
    if type(reviews.get("revision")) is not int or reviews["revision"] < 0:
        raise ValueError("Review revision must be a nonnegative integer")
    decisions = reviews.get("decisions")
    if not isinstance(decisions, dict):
        raise ValueError("Review decisions must be a mapping")
    valid_ids = {note["id"] for record in report["records"] for note in record["notes"]}
    for note_id, decision in decisions.items():
        if note_id not in valid_ids or not isinstance(decision, dict):
            raise ValueError("Unknown note ID or invalid review decision")
        status, stems = decision.get("status"), decision.get("stems")
        if status not in ("confirmed", "absent", "uncertain") or not isinstance(stems, list):
            raise ValueError("Review needs confirmed/absent/uncertain status and a stem list")
        if (any(not isinstance(stem, str) or stem not in STEMS for stem in stems)
                or len(set(stems)) != len(stems)):
            raise ValueError("Invalid or duplicated reviewed stem")
        if (status == "confirmed") != bool(stems):
            raise ValueError("Only confirmed notes may have stems; confirmation requires at least one")
        if decision.get("scope") not in (None, "exact_note"):
            raise ValueError("Review scope must be exact_note or an unscoped legacy decision")
    comments = reviews.get("comments", {})
    if not isinstance(comments, dict):
        raise ValueError("Comments must be a mapping")
    for note_id, comment in comments.items():
        if (note_id not in valid_ids or not isinstance(comment, dict)
                or not isinstance(comment.get("text"), str)
                or not 0 < len(comment["text"]) <= 4000):
            raise ValueError("Comments need a known note ID and 1-4000 characters of text")
    return decisions


def load_report(output: Path, verify_audio: bool = False) -> dict:
    report = read_json(output / "report.json")
    # Preserve old review hashes only when the matching calculations are byte-for-byte unchanged.
    compatible = (report["code_sha256"] == code_hash()
                  or report["code_sha256"] == LEGACY_CODE_SHA256
                  and evidence_hash() == LEGACY_EVIDENCE_SHA256)
    if (report["schema"] != SCHEMA or report["status"] != "ready_for_review"
            or not compatible or report["split"] != "train"):
        raise ValueError("Pilot incomplete or algorithm/schema changed; use a fresh output")
    dataset = Path(report["dataset"])
    for name, expected in report["dataset_sha256"].items():
        if digest(dataset / name) != expected:
            raise ValueError(f"Pilot dataset metadata changed: {name}")
    groups = set()
    ids = set()
    for record in report["records"]:
        if record["split"] != "train" or record["source_group"] in groups:
            raise ValueError("Pilot must contain distinct training source groups")
        groups.add(record["source_group"])
        for note in record["notes"]:
            if note["id"] in ids:
                raise ValueError("Duplicated note ID")
            ids.add(note["id"])
        if verify_audio:
            for name, path in record["audio"].items():
                if digest(Path(path)) != record["audio_sha256"][name]:
                    raise ValueError(f"Pilot audio changed: {path}")
            for family, path in record["source_midi"].items():
                if digest(Path(path)) != record["source_midi_sha256"][family]:
                    raise ValueError(f"Pilot source MIDI changed: {path}")
    return report


def export_supervision(output: Path, destination: Path, reviews_path: Path | None = None) -> dict:
    if destination.exists():
        raise FileExistsError("Refusing to overwrite a supervision export")
    report = load_report(output, verify_audio=True)
    reviews_path = reviews_path or output / "reviews.json"
    reviews = read_json(reviews_path)
    decisions = validate_reviews(report, reviews)
    records, confirmed, assignments = [], 0, 0
    for record in report["records"]:
        targets = {}
        for note in record["notes"]:
            decision = decisions.get(note["id"])
            if (decision is None or decision["status"] != "confirmed"
                    or decision.get("scope") != "exact_note"):
                continue
            confirmed += 1
            for stem in decision["stems"]:
                target = targets.setdefault(stem, {"pitched_onsets": [], "drum_onsets": []})
                label = {
                    "note_id": note["id"], "onset": note["onset"], "family": note["family"],
                    "source_pitch": note["pitch"], "source_channel": note["channel"],
                    "source_velocity": note["velocity"], "source_offset": note["offset"],
                }
                if note["family"] == "drums":
                    cls = drum_class(note["pitch"])
                    if cls is None:
                        raise ValueError("Cannot export an unsupported five-class drum")
                    label["class_index"] = cls
                    target["drum_onsets"].append(label)
                else:
                    label["pitch"] = note["pitch"]
                    target["pitched_onsets"].append(label)
                assignments += 1
        records.append({
            "composition_id": record["composition_id"], "source_group": record["source_group"],
            "split": "train", "duration": record["duration"],
            "audio": record["audio"], "audio_sha256": record["audio_sha256"],
            "clean_baseline_midi": record["source_midi"],
            "source_midi_sha256": record["source_midi_sha256"], "targets": targets,
            "license": record["license"], "provenance": record["provenance"],
            "rights_review_required": True,
            "unsupervised_stems": [stem for stem in STEMS if stem not in targets],
            "negative_supervision_allowed": False,
        })
    if not confirmed:
        raise ValueError("No human-confirmed notes with exact-note scope; legacy decisions require rechecking")
    result = {
        "schema": SCHEMA, "report_sha256": report_hash(report),
        "reviews_sha256": digest(reviews_path), "records": records,
        "review_comments": reviews.get("comments", {}),
        "label_policy": "Human-confirmed positive onsets only; all other labels/times are unknown, not negative.",
        "negative_supervision_allowed": False, "rights_review_required": True,
        "review_scope": "exact_note",
        "legacy_decisions_requiring_recheck": sum(
            decision.get("scope") != "exact_note" for decision in decisions.values()),
        "drum_head_order": ["kick", "snare", "toms", "hi_hat", "cymbals_ride"],
        "source_offsets_are_acoustic_targets": False,
        "confirmed_notes": confirmed, "positive_stem_assignments": assignments,
        "review_counts": dict(Counter(decision["status"] for decision in decisions.values())),
        "unreviewed_notes": report["summary"]["notes"] - len(decisions),
    }
    write_json(destination, result)
    return result


def load_positive_supervision(path: Path, *, rights_cleared: bool = False) -> dict:
    """Consumer contract: positives only, never implicit empty-stem negatives."""
    data = read_json(path)
    if not rights_cleared:
        raise ValueError("Confirm composition/arrangement rights before training")
    if data["schema"] != SCHEMA or data["negative_supervision_allowed"] is not False:
        raise ValueError("Unsupported positive-only supervision policy")
    if data.get("review_scope") != "exact_note":
        raise ValueError("Supervision needs explicit exact-note review scope")
    if not data["confirmed_notes"]:
        raise ValueError("No reviewed positives available")
    groups, confirmed_ids, assignments = set(), set(), 0
    for record in data["records"]:
        if record["split"] != "train" or record["negative_supervision_allowed"] is not False:
            raise ValueError("Non-training or implicit-negative supervision")
        if record["source_group"] in groups:
            raise ValueError("Duplicated source group in matching supervision")
        groups.add(record["source_group"])
        for name, audio in record["audio"].items():
            if digest(Path(audio)) != record["audio_sha256"][name]:
                raise ValueError(f"Supervision audio changed: {audio}")
        for family, midi in record["clean_baseline_midi"].items():
            if digest(Path(midi)) != record["source_midi_sha256"][family]:
                raise ValueError(f"Supervision source MIDI changed: {midi}")
        for stem, target in record["targets"].items():
            if stem not in STEMS:
                raise ValueError("Unknown target stem")
            ids = set()
            for lane in ("pitched_onsets", "drum_onsets"):
                for label in target[lane]:
                    if (label["note_id"] in ids or not math.isfinite(label["onset"])
                            or not 0 <= label["onset"] < record["duration"]):
                        raise ValueError("Duplicated or invalid positive onset")
                    ids.add(label["note_id"])
                    confirmed_ids.add(label["note_id"])
                    assignments += 1
                    if lane == "drum_onsets":
                        if (label["family"] != "drums"
                                or drum_class(label["source_pitch"]) != label["class_index"]
                                or label["class_index"] is None):
                            raise ValueError("Invalid five-class drum label")
                    elif (label["family"] not in ("bass", "piano", "guitar")
                          or type(label["pitch"]) is not int or not 0 <= label["pitch"] <= 127):
                        raise ValueError("Invalid melodic pitch label")
    if (len(confirmed_ids) != data["confirmed_notes"]
            or assignments != data["positive_stem_assignments"]):
        raise ValueError("Reviewed positive counts do not match supervision")
    return data


def serve(output: Path, port: int, fluidsynth: str | None = None) -> None:
    from experimental.benchmark.matching_cues import CueRenderer, click_timing

    report = load_report(output, verify_audio=True)
    validate_reviews(report, read_json(output / "reviews.json"))
    audio_paths = {
        f"/audio/{record['clip']}/{name}.wav": Path(path)
        for record in report["records"] for name, path in record["audio"].items()
    }
    token = secrets.token_urlsafe(32)
    lock = threading.Lock()
    cues = CueRenderer(output, report, fluidsynth)
    records = {record["clip"]: record for record in report["records"]}

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, content: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

        def send_json(self, value: dict, status: int = 200) -> None:
            self.send_bytes(json.dumps(value, allow_nan=False).encode(), "application/json", status)

        def allowed_host(self) -> bool:
            return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

        def do_GET(self) -> None:
            if not self.allowed_host():
                self.send_error(403, "Loopback host required")
                return
            path = urlparse(self.path).path
            if path == "/report.json":
                self.send_json({"report": report, "token": token,
                                "reviews": read_json(output / "reviews.json")})
            elif path == "/health":
                saved = read_json(output / "reviews.json")
                self.send_json({"status": "ok", "revision": saved["revision"],
                                "updated_at": saved.get("updated_at")})
            elif path in ("/", "/index.html", "/review.js"):
                name = "index.html" if path == "/" else path[1:]
                self.send_bytes((ASSETS / name).read_bytes(),
                                "text/html; charset=utf-8" if name.endswith("html") else "text/javascript")
            elif path.startswith("/click/") and path.endswith(".json"):
                record = records.get(unquote(path[len("/click/"):-5]))
                if record is None:
                    self.send_error(404, "Unknown song")
                    return
                try:
                    family = next(iter(record["source_midi"]))
                    source = Path(record["source_midi"][family])
                    if digest(source) != record["source_midi_sha256"][family]:
                        raise ValueError("Source MIDI changed before click audition")
                    self.send_json({"beats": click_timing(source, record["duration"])})
                except (OSError, ValueError, KeyError, EOFError) as error:
                    LOG.exception("Click-context timing failed")
                    self.send_json({"error": str(error)}, 500)
                return
            elif path.startswith("/cue/") and path.endswith(".wav"):
                if self.headers.get("X-Review-Token") != token:
                    self.send_error(403, "Review token required for cue rendering")
                    return
                note_id = unquote(path[len("/cue/"):-4])
                if note_id not in cues.notes:
                    self.send_error(404, "Unknown note")
                    return
                try:
                    cue_path = cues.render(note_id)
                except (OSError, ValueError, RuntimeError, KeyError) as error:
                    LOG.exception("Isolated-note cue failed")
                    self.send_json({"error": str(error)}, 500)
                    return
                self.send_audio(cue_path)
            elif path in audio_paths:
                self.send_audio(audio_paths[path])
            else:
                self.send_error(404)

        def send_audio(self, path: Path) -> None:
            size = path.stat().st_size
            start, stop = 0, size - 1
            range_header = self.headers.get("Range")
            if range_header:
                try:
                    unit, values = range_header.split("=")
                    first, last = values.split("-")
                    if unit != "bytes" or not first or "," in values:
                        raise ValueError("Unsupported range")
                    start = int(first)
                    stop = min(int(last), size - 1) if last else size - 1
                    if start < 0 or start >= size or stop < start:
                        raise ValueError("Invalid range")
                except ValueError:
                    self.send_error(416, "Invalid byte range")
                    return
            self.send_response(206 if range_header else 200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(stop - start + 1))
            if range_header:
                self.send_header("Content-Range", f"bytes {start}-{stop}/{size}")
            self.end_headers()
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = stop - start + 1
                try:
                    while remaining:
                        chunk = stream.read(min(65536, remaining))
                        if not chunk:
                            raise IOError("Audio file changed during playback")
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    LOG.debug("Browser stopped audio playback")

        def do_POST(self) -> None:
            origin = f"http://127.0.0.1:{self.server.server_port}"
            if (not self.allowed_host() or self.headers.get("Origin") not in (None, origin)
                    or self.headers.get("X-Review-Token") != token):
                self.send_error(403, "Same-origin review token required")
                return
            if urlparse(self.path).path != "/reviews":
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 32768:
                    raise ValueError("Review request must be 1-32768 bytes")
                request = json.loads(self.rfile.read(size))
                if not isinstance(request, dict):
                    raise ValueError("Review request must be an object")
                with lock:
                    current = read_json(output / "reviews.json")
                    if request.get("revision") != current["revision"]:
                        self.send_json({"error": "Review changed in another window; reload before saving"}, 409)
                        return
                    note_id = request["note_id"]
                    if note_id not in {note["id"] for record in report["records"]
                                       for note in record["notes"]}:
                        raise ValueError("Unknown note ID")
                    if "decision" not in request and "comment" not in request:
                        raise ValueError("Include a review decision or comment")
                    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    if "decision" in request:
                        if request["decision"] is None:
                            current["decisions"].pop(note_id, None)
                        elif isinstance(request["decision"], dict):
                            current["decisions"][note_id] = {**request["decision"], "saved_at": now}
                        else:
                            raise ValueError("Decision must be an object or null")
                    if "comment" in request:
                        text = request["comment"]
                        if not isinstance(text, str) or len(text) > 4000:
                            raise ValueError("Comment must be text with at most 4000 characters")
                        comments = current.setdefault("comments", {})
                        if text:
                            comments[note_id] = {"text": text, "saved_at": now}
                        else:
                            comments.pop(note_id, None)
                    validate_reviews(report, current)
                    current["revision"] += 1
                    current["updated_at"] = now
                    write_json(output / "reviews.json", current)
                self.send_json(current)
            except (ValueError, KeyError, TypeError) as error:
                LOG.warning("Invalid review request: %s", error)
                self.send_json({"error": str(error)}, 400)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    LOG.info("Matching review: http://127.0.0.1:%d", server.server_port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare", help="Analyze existing train audio; no rendering or model inference")
    build.add_argument("--dataset", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--songs", type=int, default=12)
    build.add_argument("--predictions", type=Path, help="Optional existing canned-prediction run")
    review = commands.add_parser("serve", help="Local review UI; decisions persist in reviews.json")
    review.add_argument("--out", type=Path, required=True)
    review.add_argument("--port", type=int, default=8767)
    review.add_argument("--fluidsynth", help="Override the recorded local FluidSynth executable")
    export = commands.add_parser("export", help="Export only manually confirmed positive onset targets")
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--destination", type=Path, required=True)
    export.add_argument("--reviews", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "prepare":
        prepare(args.dataset, args.out, args.songs, args.predictions)
    elif args.command == "serve":
        serve(args.out, args.port, args.fluidsynth)
    else:
        result = export_supervision(args.out, args.destination, args.reviews)
        LOG.info("Exported %d confirmed notes / %d stem positives",
                 result["confirmed_notes"], result["positive_stem_assignments"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
