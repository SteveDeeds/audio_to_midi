"""Cached evidence and tuning for the local experimental GUI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from experimental.vocal_midi import (
    HOP, METHODS, SR, SOURCE, Detector, Note, NoteSettings, coalesce, custom_detector, energy_attacks,
    fixed_window, flux_detector, make_notes, neural_detectors, stable_changes, voice_regions,
)

KEYS = ["Off", *[n for root in ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
                 for n in (root, root + "m")]]
CONTROL_SPECS = [
    ("neural_threshold", "Neural onset threshold", 0.01, 0.99, 0.01, "Onsets", "neural",
     "Lower finds more attacks. This changes peak picking, not trained model weights."),
    ("flux_delta", "Flux onset threshold", 0.01, 0.8, 0.01, "Onsets", "flux",
     "Lower finds softer attacks in librosa/SuperFlux; may also detect breath or vibrato."),
    ("onset_wait", "Minimum onset spacing (s)", 0.02, 0.4, 0.01, "Onsets", "all",
     "Combine nearby onset proposals. Separate from minimum exported note length."),
    ("min_note", "Minimum note length (s)", 0.04, 0.4, 0.01, "Splitting", "all",
     "Shorter allows faster syllables and ornaments; longer removes short notes."),
    ("pitch_hold", "Pitch-change hold (s)", 0.03, 0.4, 0.01, "Splitting", "all",
     "How long a new pitch must persist. Raise to reject brief vibrato/slide excursions."),
    ("pitch_hysteresis", "Pitch-change threshold (semitones)", 0.25, 2.0, 0.05, "Splitting", "all",
     "Movement from the held semitone required before proposing a pitch-change split."),
    ("periodicity", "CREPE periodicity gate", 0.05, 0.95, 0.01, "Certainty", "all",
     "Actual median-filtered CREPE periodicity. Not a calibrated probability of correct pitch."),
    ("voice_db", "Voice energy gate (relative dB)", -60.0, -10.0, 1.0, "Certainty", "all",
     "Exclude quiet regions from notes. The measured pitch contour remains visible as evidence."),
    ("min_voiced_fraction", "Minimum note voiced fraction", 0.0, 1.0, 0.05, "Certainty", "all",
     "Reject notes whose pitch body contains too few voiced frames."),
    ("min_pitch_agreement", "Minimum note pitch agreement", 0.0, 1.0, 0.05, "Certainty", "all",
     "Fraction of voiced frames within half a semitone of the measured note. Not accuracy."),
    ("release_gap", "Phrase release gap (s)", 0.02, 0.3, 0.01, "Articulation", "all",
     "Unvoiced gaps shorter than this are bridged; longer gaps end phrases."),
    ("articulation_gap", "Minimum articulation gap (s)", 0.01, 0.1, 0.01, "Articulation", "all",
     "Pre-attack unvoiced or energy-dip duration needed to shorten the previous note."),
    ("articulation_dip", "Articulation dip below body (dB)", 2.0, 15.0, 0.5, "Articulation", "all",
     "Required pre-attack energy dip relative to the preceding vowel."),
    ("articulation_recovery", "Articulation recovery (dB)", 1.0, 12.0, 0.5, "Articulation", "all",
     "Required rise from the energy valley to the next vowel."),
    ("energy_valley", "Hybrid energy valley (dB)", 1.0, 12.0, 0.5, "Hybrid", "hybrid",
     "Energy dip required to accept a SuperFlux retrigger in the pitch-aware method."),
    ("energy_recovery", "Hybrid energy recovery (dB)", 1.0, 12.0, 0.5, "Hybrid", "hybrid",
     "Recovery required to accept a SuperFlux retrigger, including same-pitch syllables."),
    ("key_limit", "Key correction limit (semitones)", 0.0, 2.0, 1.0, "Key", "key",
     "Maximum semitone correction for an out-of-key note. Zero leaves notes unchanged."),
]
GUI_CONTROL_SPECS = [
    CONTROL_SPECS[0], CONTROL_SPECS[1], CONTROL_SPECS[7],
    ("onset_wait", "Minimum syllable spacing (s)", 0.02, 0.4, 0.01, "Syllables", "all",
     "Try 0.10-0.16 s. Higher removes close attack duplicates, not within-syllable pitch changes."),
    ("pitch_hold", "Slur pitch-change hold (s)", 0.03, 0.4, 0.01, "Slurs", "all",
     "Try 0.06-0.16 s. Lower splits short pitch movements; higher rejects brief vibrato and slides."),
    ("min_note", "Minimum split length (s)", 0.04, 0.4, 0.01, "Splitting", "all",
     "Try 0.05-0.10 s. Lower accepts shorter proposed syllables/slur notes; it does not invent syllables."),
]
VERSIONS = ("numpy", "scipy", "librosa", "torch", "torchcrepe", "madmom")
CACHE_SCHEMA = 1


@dataclass(frozen=True)
class TuningParameters(NoteSettings):
    method: str = "madmom_cnn"
    neural_threshold: float = 0.30
    flux_delta: float = 0.12
    onset_wait: float = 0.08
    periodicity: float = 0.35

    @classmethod
    def from_dict(cls, values: dict) -> TuningParameters:
        if not isinstance(values, dict):
            raise ValueError("Parameters must be a JSON object.")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown parameters: {', '.join(sorted(unknown))}")
        result = cls(**values)
        if (not isinstance(result.method, str) or result.method not in METHODS or
                not isinstance(result.key, str) or result.key not in KEYS):
            raise ValueError("Unknown detector or musical key.")
        if type(result.split_pitch) is not bool or type(result.syllabic) is not bool:
            raise ValueError("Pitch splitting and syllabic mode must be true or false.")
        for name, _, low, high, _, _, _, _ in CONTROL_SPECS:
            value = getattr(result, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric.")
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be between {low:g} and {high:g}.")
        return result


def tuning_defaults() -> TuningParameters:
    return TuningParameters(split_pitch=True, syllabic=True, onset_wait=0.12, energy_valley=6.0)


@dataclass
class Evidence:
    metadata: dict
    arrays: dict[str, np.ndarray]
    directory: Path

    @property
    def duration(self) -> float:
        return self.metadata["duration_seconds"]


def crepe_evidence(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same full CREPE configuration as the experiment, retaining real periodicity."""
    import torch
    import torchcrepe

    torch.set_num_threads(4)
    # torchcrepe's frequency conversion uses SciPy/NumPy random pitch dither.
    np.random.seed(0)
    audio = torch.from_numpy(librosa.resample(y, orig_sr=SR, target_sr=16000))[None]
    with torch.inference_mode():
        f0, periodicity = torchcrepe.predict(
            audio, 16000, hop_length=160, fmin=70.0, fmax=1100.0, model="full",
            decoder=torchcrepe.decode.viterbi, return_periodicity=True,
            batch_size=1024, device="cpu",
        )
        confidence = torchcrepe.filter.median(periodicity, 3)[0].numpy()
        midi = librosa.hz_to_midi(np.maximum(np.nan_to_num(f0[0].numpy(), nan=1.0), 1.0))
    return np.arange(len(midi)) * 0.01, midi, confidence


def prepare_evidence(directory: Path, source: Path = SOURCE, start: float = 39,
                     seconds: float = 26.6, baseline: Path | None = None) -> Evidence:
    directory.mkdir(parents=True, exist_ok=True)
    source = source.resolve(strict=True)
    audio, sr = sf.read(source, dtype="float32", always_2d=True)
    a, b, _ = fixed_window(audio, sr, seconds, start)
    pad_a, pad_b = max(0, a - sr), min(len(audio), b + sr)
    shift = (a - pad_a) / sr
    context = audio[pad_a:pad_b].mean(axis=1)
    analysis = librosa.resample(context, orig_sr=sr, target_sr=SR)
    print("Preparing cache: CREPE full pitch + actual periodicity...", flush=True)
    times, pitch, confidence = crepe_evidence(analysis)
    rms = librosa.feature.rms(y=analysis, frame_length=1024, hop_length=HOP)[0]
    db = librosa.amplitude_to_db(rms + 1e-9, ref=np.max(rms) + 1e-9)
    interpolated_db = np.interp(times, librosa.times_like(rms, sr=SR, hop_length=HOP), db)
    # Preserve the original pitch tracker's additional 2048-frame/-45dB gate.
    tracker_rms = librosa.feature.rms(y=analysis, frame_length=2048, hop_length=256)[0]
    tracker_db = librosa.amplitude_to_db(tracker_rms + 1e-9, ref=np.max(tracker_rms) + 1e-9)
    tracker_gate = np.interp(
        times, librosa.times_like(tracker_rms, sr=SR, hop_length=256), tracker_db,
    ) > -45.0
    keep = (times >= shift) & (times < shift + (b - a) / sr)
    arrays = {"times": times[keep] - shift, "pitch": pitch[keep],
              "periodicity": confidence[keep], "db": interpolated_db[keep],
              "tracker_gate": tracker_gate[keep], "vocal": audio[a:b].mean(axis=1)}
    print("Preparing cache: librosa/SuperFlux and pretrained madmom activations...", flush=True)
    detectors = {"librosa": flux_detector(analysis, False),
                 "superflux": flux_detector(analysis, True)}
    detectors.update(neural_detectors(context, sr))
    parameters = {}
    for name, detector in detectors.items():
        arrays[f"{name}_times"] = detector.envelope_times - shift
        arrays[f"{name}_envelope"] = detector.envelope
        parameters[name] = detector.parameters
    cqt = np.abs(librosa.cqt(analysis, sr=SR, hop_length=HOP,
                            fmin=librosa.midi_to_hz(36), n_bins=144, bins_per_octave=36))
    arrays["cqt_db"] = librosa.amplitude_to_db(cqt, ref=np.max(cqt))
    arrays["cqt_times"] = librosa.times_like(cqt, sr=SR, hop_length=HOP) - shift
    voiced = pitch[keep][(confidence[keep] > 0.35) & tracker_gate[keep]]
    if len(voiced) < 10:
        raise ValueError("Too few confident pitch frames for a tuning session.")
    metadata = {
        "schema": CACHE_SCHEMA, "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sample_rate": sr, "start_sample": a, "end_sample": b,
        "start_seconds": a / sr, "end_seconds": b / sr, "duration_seconds": (b - a) / sr,
        "pitch_tracker": "CREPE full, Viterbi, 10ms, median periodicity 3, CPU threads 4",
        "new_pitch_dither_seed": 0,
        "versions": {name: importlib.metadata.version(name) for name in VERSIONS},
        "detector_parameters": parameters,
        "pitch_low": max(36, int(np.floor(np.percentile(voiced, 2))) - 5),
        "pitch_high": min(83, int(np.ceil(np.percentile(voiced, 98))) + 5),
    }
    evidence = Evidence(metadata, arrays, directory)
    if baseline is not None:
        anchor_baseline(evidence, baseline)
    save_evidence(evidence)
    render_background(evidence)
    print(f"Analysis cached: {directory}", flush=True)
    return evidence


def anchor_baseline(evidence: Evidence, report_path: Path) -> None:
    """Reuse the reviewed contour, avoiding new random dither at semitone boundaries."""
    report_path = report_path.resolve(strict=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report["source"]["sha256"] != evidence.metadata["source_sha256"] or
            report["window"]["start_sample"] != evidence.metadata["start_sample"] or
            report["window"]["end_sample"] != evidence.metadata["end_sample"]):
        raise ValueError("Baseline review source/interval does not match the tuning cache.")
    csv_path = report_path.parent / "pitch.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    times = np.array([float(row["crop_seconds"]) for row in rows])
    pitch = np.array([float(row["midi"]) if row["midi"] else np.nan for row in rows])
    if not np.array_equal(times, evidence.arrays["times"]):
        raise ValueError("Baseline pitch frame grid does not match the tuning cache.")
    voiced = (evidence.arrays["periodicity"] > 0.35) & evidence.arrays["tracker_gate"]
    if not np.array_equal(voiced, np.isfinite(pitch)):
        raise ValueError("Baseline voicing does not match fresh CREPE evidence.")
    evidence.arrays["pitch"][voiced] = pitch[voiced]
    evidence.metadata["baseline_pitch_csv"] = str(csv_path)
    evidence.metadata["baseline_pitch_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    evidence.metadata["pitch_contour_source"] = (
        "Reviewed pitch CSV for previously voiced frames; fresh CREPE for previously unvoiced frames."
    )


def save_evidence(evidence: Evidence) -> None:
    np.savez_compressed(evidence.directory / "evidence.npz", **evidence.arrays)
    (evidence.directory / "evidence.json").write_text(
        json.dumps(evidence.metadata, indent=2), encoding="utf-8",
    )


def load_evidence(directory: Path) -> Evidence:
    metadata = json.loads((directory / "evidence.json").read_text(encoding="utf-8"))
    if metadata["schema"] != CACHE_SCHEMA:
        raise ValueError("Cache schema changed. Rebuild the experimental cache.")
    source = Path(metadata["source"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != metadata["source_sha256"]:
        raise ValueError("Source audio changed. Rebuild the experimental cache.")
    if metadata["versions"] != {name: importlib.metadata.version(name) for name in VERSIONS}:
        raise ValueError("Analysis library versions changed. Rebuild the experimental cache.")
    if "baseline_pitch_csv" in metadata:
        if hashlib.sha256(Path(metadata["baseline_pitch_csv"]).read_bytes()).hexdigest() != metadata["baseline_pitch_sha256"]:
            raise ValueError("Reviewed pitch contour changed. Rebuild the experimental cache.")
    with np.load(directory / "evidence.npz", allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    if not (directory / "cqt_background.png").is_file():
        raise ValueError("CQT background missing. Rebuild the experimental cache.")
    return Evidence(metadata, arrays, directory)


def render_background(evidence: Evidence) -> None:
    fig = plt.figure(figsize=(20, 5), dpi=120)
    axis = fig.add_axes((0, 0, 1, 1))
    axis.pcolormesh(evidence.arrays["cqt_times"], np.arange(144) / 3 + 36,
                    evidence.arrays["cqt_db"], shading="nearest", cmap="magma", vmin=-65, vmax=0)
    axis.set(xlim=(0, evidence.duration),
             ylim=(evidence.metadata["pitch_low"], evidence.metadata["pitch_high"]))
    axis.set_axis_off()
    fig.savefig(evidence.directory / "cqt_background.png", dpi=120)
    plt.close(fig)


def pick_cached(evidence: Evidence, method: str, parameters: TuningParameters) -> Detector:
    from madmom.features.onsets import OnsetPeakPickingProcessor

    name = "superflux" if method == "pitch_aware" else method
    times = evidence.arrays[f"{name}_times"]
    env = evidence.arrays[f"{name}_envelope"]
    if name.startswith("madmom"):
        onsets = OnsetPeakPickingProcessor(
            fps=100, threshold=parameters.neural_threshold, smooth=0,
            pre_max=0.03, post_max=0.03, combine=parameters.onset_wait,
        )(env)
        onsets += float(times[0])
    else:
        onsets = librosa.onset.onset_detect(
            onset_envelope=env, sr=SR, hop_length=HOP, units="time",
            normalize=False, delta=parameters.flux_delta,
            wait=round(parameters.onset_wait * SR / HOP),
            pre_max=round(0.03 * SR / HOP), post_max=round(0.03 * SR / HOP),
            pre_avg=round(0.10 * SR / HOP), post_avg=round(0.10 * SR / HOP),
        ) + float(times[0])
    return Detector(onsets[(onsets >= 0) & (onsets < evidence.duration)],
                    times, env, asdict(parameters))


def retune(evidence: Evidence, parameters: TuningParameters) -> tuple[dict, list[Note], Detector]:
    started = time.perf_counter()
    arrays = evidence.arrays
    times, db = arrays["times"], arrays["db"]
    pitch = np.where((arrays["periodicity"] > parameters.periodicity) &
                     arrays["tracker_gate"], arrays["pitch"], np.nan)
    regions = voice_regions(times, pitch, db, evidence.duration, parameters)
    detector = pick_cached(evidence, parameters.method, parameters)
    raw_onsets = len(detector.times)
    extras = np.array([], dtype=float)
    if parameters.syllabic:
        superflux = pick_cached(evidence, "superflux", parameters)
        candidates = energy_attacks(superflux, times, pitch, db,
                                    replace(parameters, energy_recovery=parameters.energy_valley))
        extras = coalesce(np.asarray([c for c in candidates
                                      if any(a <= c < b for a, b in regions) and
                                      not np.any(np.abs(detector.times - c) < parameters.onset_wait - 1e-9)]),
                          parameters.onset_wait)
        detector = Detector(np.unique(np.concatenate((detector.times, extras))),
                            detector.envelope_times, detector.envelope, detector.parameters, extras)
    elif parameters.method == "pitch_aware":
        detector = custom_detector(detector, times, pitch, db, regions, parameters)
        raw_onsets = len(detector.times)
    notes = make_notes(detector, times, pitch, db, regions, parameters)
    changes = stable_changes(times, pitch, parameters) if parameters.split_pitch or parameters.syllabic else np.array([])
    result = {
        "parameters": asdict(parameters),
        "parameter_id": hashlib.sha256(json.dumps(asdict(parameters), sort_keys=True).encode()).hexdigest()[:16],
        "notes": [{**asdict(n), "name": librosa.midi_to_note(n.midi, unicode=False)} for n in notes],
        "onsets": detector.times.tolist(), "pitch_changes": changes.tolist(), "energy_onsets": extras.tolist(),
        "times": times.tolist(), "pitch": [float(v) if np.isfinite(v) else None for v in pitch],
        "periodicity": arrays["periodicity"].tolist(), "db": db.tolist(),
        "envelope_times": detector.envelope_times.tolist(), "envelope": detector.envelope.tolist(),
        "stats": {"notes": len(notes), "raw_onsets": raw_onsets, "extra_syllable_proposals": len(extras),
                  "syllable_starts": sum(n.start_evidence != "pitch change" for n in notes),
                  "slur_pitch_starts": sum(n.start_evidence == "pitch change" for n in notes),
                  "pitch_change_proposals": len(changes),
                  "key_adjusted": sum(n.key_adjusted for n in notes),
                  "links": {label: sum(n.link_to_next == label for n in notes)
                            for label in ("slurred", "tied", "articulated", "phrase_end")},
                  "voiced_fraction": float(np.isfinite(pitch).mean())},
        "runtime_ms": (time.perf_counter() - started) * 1000,
    }
    return result, notes, detector
