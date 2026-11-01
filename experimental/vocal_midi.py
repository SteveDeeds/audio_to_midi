"""Compare vocal onset detectors with one pitch track and evidence-based releases."""

from __future__ import annotations

import argparse
import base64
import csv
from dataclasses import asdict, dataclass
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mido
import numpy as np
import soundfile as sf

from core.mono_pitch import track_pitch
from core.chord_analysis import key_to_scale_pcs
from core.vocal_boundaries import BoundarySettings, coalesce, energy_retriggers, runs, stable_changes, voice_regions

SR = 22050
HOP = 220
MIN_NOTE = 0.08
RELEASE_GAP = 0.06
SYLLABLE_RENDER_GAP = 0.01
NEURAL_PEAK_PARAMETERS = dict(
    fps=100, threshold=0.30, pre_max=0.03, post_max=0.03, smooth=0.0, combine=MIN_NOTE,
)
SOURCE = Path(
    r"C:\Users\steve\Music\How We Go On From Here Project"
    r"\1_How We Go On From Here 150bpm exact core_(Vocals).mp3"
)
METHODS = {
    "librosa": "librosa spectral flux",
    "superflux": "SuperFlux (librosa lag=2, max_size=3)",
    "madmom_cnn": "madmom Schlueter/Boeck CNN",
    "madmom_brnn": "madmom bidirectional RNN ensemble",
    "pitch_aware": "Pitch-aware: stable changes + energy valleys + SuperFlux",
}


@dataclass
class Detector:
    times: np.ndarray
    envelope_times: np.ndarray
    envelope: np.ndarray
    parameters: dict
    extra_onsets: np.ndarray | None = None


@dataclass(frozen=True)
class NoteSettings(BoundarySettings):
    split_pitch: bool = False
    articulation_gap: float = 0.02
    articulation_dip: float = 6.0
    articulation_recovery: float = 4.0
    energy_valley: float = 3.0
    energy_recovery: float = 3.0
    min_voiced_fraction: float = 0.0
    min_pitch_agreement: float = 0.0
    key: str = "Off"
    key_limit: float = 1.0
    syllabic: bool = False


@dataclass
class Note:
    start: float
    end: float
    midi: int
    velocity: int
    start_evidence: str
    end_evidence: str
    link_to_next: str = "phrase_end"
    gap_to_next_ms: float | None = None
    voiced_fraction: float = 0.0
    pitch_agreement: float = 0.0
    raw_midi: float = 0.0
    key_adjusted: bool = False


def loudest_window(audio: np.ndarray, sr: int, seconds: float) -> tuple[int, int, float]:
    """Maximize full-window mean channel power at original-sample resolution."""
    if audio.ndim not in (1, 2) or not len(audio) or sr <= 0 or seconds <= 0:
        raise ValueError("Expected nonempty mono/stereo audio, positive rate and duration.")
    if not np.isfinite(audio).all():
        raise ValueError("Source audio contains non-finite samples.")
    size = round(seconds * sr)
    if size < 1 or len(audio) < size:
        raise ValueError(f"Source must contain at least {seconds:g} seconds.")
    power = np.square(audio, dtype=np.float64)
    if power.ndim == 2:
        power = power.mean(axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(power)))
    sums = cumulative[size:] - cumulative[:-size]
    start = int(np.argmax(sums))
    if sums[start] <= 0:
        raise ValueError("Source is silent; no vocal experiment is possible.")
    return start, start + size, float(np.sqrt(sums[start] / size))


def fixed_window(
    audio: np.ndarray, sr: int, seconds: float, start_seconds: float,
) -> tuple[int, int, float]:
    """Select an explicit source interval without silently clipping its duration."""
    if audio.ndim not in (1, 2) or not len(audio) or sr <= 0:
        raise ValueError("Expected nonempty mono/stereo audio and a positive sample rate.")
    if not np.isfinite(seconds) or seconds <= 0 or not np.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("Duration must be finite and positive; start must be finite and nonnegative.")
    start, size = round(start_seconds * sr), round(seconds * sr)
    end = start + size
    if size < 1 or end > len(audio):
        raise ValueError("Requested interval must fit entirely inside the source audio.")
    segment = audio[start:end]
    if not np.isfinite(segment).all():
        raise ValueError("Selected audio contains non-finite samples.")
    rms = float(np.sqrt(np.mean(np.square(segment, dtype=np.float64))))
    if rms <= 0:
        raise ValueError("Selected interval is silent; no vocal experiment is possible.")
    return start, end, rms


def flux_detector(y: np.ndarray, superflux: bool) -> Detector:
    parameters = {
        "hop_length": HOP, "lag": 2 if superflux else 1,
        "max_size": 3 if superflux else 1, "delta": 0.12,
        "wait_seconds": MIN_NOTE, "backtrack": False,
    }
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, hop_length=HOP, n_fft=2048,
        n_mels=138 if superflux else 128, fmin=27.5, fmax=10000,
    )
    env = librosa.onset.onset_strength(
        S=librosa.power_to_db(mel, ref=np.max), sr=SR, hop_length=HOP,
        lag=parameters["lag"], max_size=parameters["max_size"],
    )
    # Explicit normalization matches librosa's default detection normalization.
    env = env - env.min()
    env /= max(float(env.max()), 1e-12)
    detected = librosa.onset.onset_detect(
        onset_envelope=env, sr=SR, hop_length=HOP, units="time",
        normalize=False, delta=parameters["delta"], wait=round(MIN_NOTE * SR / HOP),
        pre_max=round(0.03 * SR / HOP), post_max=round(0.03 * SR / HOP),
        pre_avg=round(0.10 * SR / HOP), post_avg=round(0.10 * SR / HOP),
    )
    parameters.update(pre_max_seconds=0.03, post_max_seconds=0.03,
                      pre_avg_seconds=0.10, post_avg_seconds=0.10,
                      n_mels=138 if superflux else 128, n_fft=2048)
    return Detector(detected, librosa.times_like(env, sr=SR, hop_length=HOP), env, parameters)


def neural_detectors(y: np.ndarray, sr: int) -> dict[str, Detector]:
    from madmom.audio.signal import Signal
    from madmom.features.onsets import (
        CNNOnsetProcessor, OnsetPeakPickingProcessor, RNNOnsetProcessor,
    )
    from madmom.models import ONSETS_BRNN, ONSETS_CNN

    signal = Signal(librosa.resample(y, orig_sr=sr, target_sr=44100), sample_rate=44100)
    result = {}
    for name, processor, models in (
        ("madmom_cnn", CNNOnsetProcessor(), ONSETS_CNN),
        ("madmom_brnn", RNNOnsetProcessor(online=False), ONSETS_BRNN),
    ):
        started = time.perf_counter()
        env = np.asarray(processor(signal)).reshape(-1)
        if not len(env) or not np.isfinite(env).all():
            raise RuntimeError(f"{name} returned empty or non-finite activations.")
        parameters = NEURAL_PEAK_PARAMETERS.copy()
        detected = OnsetPeakPickingProcessor(**parameters)(env)
        parameters["models"] = [
            {"name": Path(p).name, "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()}
            for p in models
        ]
        parameters["runtime_seconds"] = time.perf_counter() - started
        result[name] = Detector(detected, np.arange(len(env)) / 100.0, env, parameters)
        print(f"{name}: {len(detected)} context onsets in {parameters['runtime_seconds']:.1f}s",
              flush=True)
    return result


def custom_detector(
    superflux: Detector, times: np.ndarray, pitch: np.ndarray, db: np.ndarray,
    regions: list[tuple[float, float]],
    settings: NoteSettings = NoteSettings(),
) -> Detector:
    events = list(stable_changes(times, pitch, settings))
    events.extend(a for a, _ in regions)
    events.extend(energy_attacks(superflux, times, pitch, db, settings))
    changes = stable_changes(times, pitch, settings)
    env = superflux.envelope.copy()
    for event in changes:
        index = int(np.argmin(np.abs(superflux.envelope_times - event)))
        env[index] = 1.0
    return Detector(coalesce(np.asarray(events), settings.min_note), superflux.envelope_times, env, {
        "stable_pitch_hold_seconds": settings.pitch_hold,
        "pitch_hysteresis_semitones": settings.pitch_hysteresis,
        "energy_valley_db": settings.energy_valley, "energy_recovery_db": settings.energy_recovery,
        "voiced_neighborhood_fraction": 0.4,
        "source": "SuperFlux attacks + voiced phrase entries + sustained pitch changes",
        "activation_plot": "SuperFlux with stable-pitch proposals marked at 1.0",
    })


def energy_attacks(
    superflux: Detector, times: np.ndarray, pitch: np.ndarray, db: np.ndarray,
    settings: NoteSettings,
) -> np.ndarray:
    return energy_retriggers(superflux.times, times, pitch, db, settings,
                             settings.energy_valley, settings.energy_recovery)


def articulation_release(
    start: float, boundary: float, times: np.ndarray, pitch: np.ndarray, db: np.ndarray,
    settings: NoteSettings = NoteSettings(),
) -> tuple[float, str]:
    """Only shorten a note when pre-attack voicing/energy evidence supports a gap."""
    step = float(np.median(np.diff(times)))
    window = (times >= max(start + settings.min_note, boundary - 0.13)) & (times < boundary)
    indices = np.flatnonzero(window)
    if len(indices) < 2:
        return boundary, "connected"
    for a, b in reversed(runs(~np.isfinite(pitch[indices]))):
        release = float(times[indices[a]])
        if ((b - a) * step >= settings.articulation_gap - 1e-9 and
                boundary - release >= settings.articulation_gap - 1e-9):
            return release, "pre-attack voicing gap"
    body = (times >= max(start, boundary - 0.20)) & (times < boundary - 0.06)
    after = (times >= boundary) & (times < boundary + 0.07)
    if not body.any() or not after.any():
        return boundary, "connected"
    level = float(np.percentile(db[body], 75))
    recovery = float(np.percentile(db[after], 75))
    dips = ((db[indices] < level - settings.articulation_dip) &
            (db[indices] < recovery - settings.articulation_recovery))
    for a, b in reversed(runs(dips)):
        release = float(times[indices[a]])
        if ((b - a) * step >= settings.articulation_gap - 1e-9 and
                boundary - release >= settings.articulation_gap - 1e-9):
            return release, "pre-attack energy dip"
    return boundary, "connected"


def segment_note(
    a: float, release: float, times: np.ndarray, pitch: np.ndarray, db: np.ndarray,
    onset_evidence: str, reason: str, settings: NoteSettings, scale: set[int] | None,
) -> Note | None:
    body = (times >= a + min(0.04, (release - a) / 4)) & (times < release)
    values = pitch[body & np.isfinite(pitch)]
    if len(values) < 3:
        return None
    raw = float(np.median(values))
    midi = int(np.clip(round(raw), 0, 127))
    voiced_fraction = len(values) / max(1, int(np.count_nonzero(body)))
    agreement = float(np.mean(np.abs(values - midi) <= 0.5))
    if (voiced_fraction < settings.min_voiced_fraction or
            agreement < settings.min_pitch_agreement):
        return None
    original_midi = midi
    if scale is not None and midi % 12 not in scale:
        candidates = [p for p in range(max(0, midi - 2), min(127, midi + 2) + 1)
                      if p % 12 in scale and abs(p - midi) <= settings.key_limit]
        if candidates:
            midi = min(candidates, key=lambda p: (abs(p - raw), p))
    velocity = int(np.clip(100 + 1.2 * float(np.median(db[body])), 45, 110))
    return Note(a, release, midi, velocity, onset_evidence, reason,
                voiced_fraction=voiced_fraction, pitch_agreement=agreement,
                raw_midi=raw, key_adjusted=midi != original_midi)


def make_notes(
    detector: Detector, times: np.ndarray, pitch: np.ndarray, db: np.ndarray,
    regions: list[tuple[float, float]],
    settings: NoteSettings = NoteSettings(),
) -> list[Note]:
    notes = []
    changes = stable_changes(times, pitch, settings) if settings.split_pitch or settings.syllabic else np.array([])
    attacks = detector.times
    if settings.syllabic:
        attacks = coalesce(attacks, settings.min_note + SYLLABLE_RENDER_GAP)
        # Syllable attacks take priority over nearby pitch-only proposals.
        changes = np.asarray([c for c in changes
                              if not np.any(np.abs(attacks - c) < settings.min_note + SYLLABLE_RENDER_GAP - 1e-9)])
    proposals = np.unique(np.concatenate((attacks, changes)))
    scale = key_to_scale_pcs(settings.key) if settings.key != "Off" else None
    for begin, end in regions:
        near = attacks[np.abs(attacks - begin) <= 0.10]
        first = float(near[np.argmin(np.abs(near - begin))]) if len(near) else begin
        first = max(0.0, first)
        evidence = "detector" if len(near) else "voiced phrase entry"
        inner = proposals[(proposals >= first + settings.min_note) &
                          (proposals > begin) & (proposals < end - settings.min_note)]
        boundaries = [first, *coalesce(inner, settings.min_note).tolist(), end]
        for index, (a, b) in enumerate(zip(boundaries, boundaries[1:])):
            if b - a < settings.min_note - 1e-9:
                continue
            pitch_only_end = settings.syllabic and np.any(np.abs(changes - b) < 1e-8)
            release, reason = (
                articulation_release(a, b, times, pitch, db, settings)
                if index < len(boundaries) - 2 and not pitch_only_end
                else (b, "within-syllable pitch change" if pitch_only_end else "voiced phrase release")
            )
            if settings.syllabic and index < len(boundaries) - 2 and not pitch_only_end:
                if release > b - SYLLABLE_RENDER_GAP:
                    release, reason = b - SYLLABLE_RENDER_GAP, "syllable reattack: 10ms rendering gap"
            onset_evidence = evidence if index == 0 else (
                "pitch change" if np.any(np.abs(changes - a) < 1e-8) else "detector"
            )
            if (detector.extra_onsets is not None and
                    np.any(np.abs(detector.extra_onsets - a) < 1e-8)):
                onset_evidence = "SuperFlux + energy recovery"
            note = segment_note(a, release, times, pitch, db, onset_evidence, reason, settings, scale)
            if note is not None:
                if (settings.syllabic and notes and onset_evidence == "pitch change" and
                        abs(notes[-1].end - a) < 1e-9 and notes[-1].midi == note.midi):
                    previous = notes.pop()
                    note = segment_note(previous.start, release, times, pitch, db,
                                        previous.start_evidence, reason, settings, scale)
                if note is not None:
                    notes.append(note)
    # Adjacent phrases can begin on a consonant before voicing resumes.
    for previous, following in zip(notes, notes[1:]):
        if previous.end > following.start:
            previous.end = following.start
            previous.end_evidence = "next attack clips previous release"
        if (settings.syllabic and following.start_evidence != "pitch change" and
                following.start - previous.end < SYLLABLE_RENDER_GAP):
            previous.end = following.start - SYLLABLE_RENDER_GAP
            previous.end_evidence = "syllable reattack: 10ms rendering gap"
        gap = following.start - previous.end
        previous.gap_to_next_ms = round(gap * 1000, 3)
        previous.link_to_next = (
            ("tied" if previous.midi == following.midi else "slurred")
            if abs(gap) < 1e-9 else "articulated"
        )
    return notes


def synthesize(notes: list[Note], sr: int, samples: int) -> np.ndarray:
    """Continuous oscillator/envelope over ties/slurs, retriggered only after gaps."""
    result = np.zeros(samples, dtype=np.float64)
    phase = 0.0
    for i, note in enumerate(notes):
        a, b = max(0, round(note.start * sr)), min(samples, round(note.end * sr))
        if b <= a:
            raise ValueError("Synth note must have positive duration inside the crop.")
        connected_before = i > 0 and abs(notes[i - 1].end - note.start) < 1e-9
        connected_after = i + 1 < len(notes) and abs(note.end - notes[i + 1].start) < 1e-9
        if not connected_before:
            phase = 0.0
        omega = 2 * np.pi * float(librosa.midi_to_hz(note.midi)) / sr
        phases = phase + omega * np.arange(b - a)
        wave = np.sin(phases) + 0.25 * np.sin(2 * phases) + 0.10 * np.sin(3 * phases)
        phase = float((phase + omega * (b - a)) % (2 * np.pi))
        envelope = np.ones(b - a)
        attack = min(round(0.008 * sr), (b - a) // 2)
        release = min(round(0.015 * sr), (b - a) // 2)
        if not connected_before and attack:
            envelope[:attack] = np.linspace(0, 1, attack)
        if not connected_after and release:
            envelope[-release:] = np.linspace(1, 0, release)
        result[a:b] += wave * envelope * note.velocity / 127
    return result


def stereo_comparison(vocal: np.ndarray, synth: np.ndarray) -> tuple[np.ndarray, dict]:
    if vocal.shape != synth.shape or not len(vocal):
        raise ValueError("Vocal and synth must have identical nonempty mono sample grids.")
    if not np.isfinite(vocal).all() or not np.isfinite(synth).all() or not np.any(synth):
        raise ValueError("Comparison requires finite audio and audible synthesized notes.")
    vocal_gain = 0.85 / max(float(np.max(np.abs(vocal))), 1e-12)
    vocal_rms = float(np.sqrt(np.mean((vocal * vocal_gain) ** 2)))
    synth_rms = float(np.sqrt(np.mean(synth ** 2)))
    synth_gain = min(vocal_rms / synth_rms, 0.85 / float(np.max(np.abs(synth))))
    return np.column_stack((vocal * vocal_gain, synth * synth_gain)), {
        "vocal_gain": vocal_gain, "synth_gain": synth_gain,
        "left": "mono vocal", "right": "additive synth",
        "alignment": "sample-aligned; no padding or latency compensation",
    }


def export_midi(notes: list[Note], path: Path, bpm: float) -> None:
    # A high PPQN preserves short articulation gaps without beat quantization.
    midi = mido.MidiFile(ticks_per_beat=9600)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    tempo = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo))
    track.append(mido.Message("program_change", program=80, time=0))
    events = []
    for note in notes:
        for t, kind in ((note.start, "note_on"), (note.end, "note_off")):
            tick = round(mido.second2tick(t, midi.ticks_per_beat, tempo))
            events.append((tick, 0 if kind == "note_off" else 1, kind, note))
    previous_tick = 0
    for tick, _, kind, note in sorted(events, key=lambda e: (e[0], e[1], e[3].midi)):
        track.append(mido.Message(kind, note=note.midi,
                                  velocity=note.velocity if kind == "note_on" else 0,
                                  time=tick - previous_tick))
        previous_tick = tick
    midi.save(path)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def draw_cqt(
    axis: plt.Axes, cqt_db: np.ndarray, cqt_times: np.ndarray,
    pitch_times: np.ndarray, pitch: np.ndarray, detector: Detector,
    notes: list[Note], lo: float, hi: float, label: bool = True,
) -> None:
    axis.pcolormesh(cqt_times, np.arange(cqt_db.shape[0]) / 3 + 36,
                    cqt_db, shading="nearest", cmap="magma", vmin=-65, vmax=0,
                    rasterized=True)
    axis.plot(pitch_times, pitch, color="#35e9ff", linewidth=1.15, alpha=0.9)
    for onset in detector.times[(detector.times >= lo) & (detector.times < hi)]:
        axis.axvline(onset, color="#ffb02e", linestyle="--", linewidth=0.8, alpha=0.8)
    if detector.extra_onsets is not None:
        for onset in detector.extra_onsets[(detector.extra_onsets >= lo) & (detector.extra_onsets < hi)]:
            axis.axvline(onset, color="#74acff", linestyle="--", linewidth=0.8, alpha=0.9)
    colors = {"slurred": "#73ff93", "tied": "#ffd166",
              "articulated": "#f783ff", "phrase_end": "#ffffff"}
    for note in notes:
        if note.end <= lo or note.start >= hi:
            continue
        color = colors[note.link_to_next]
        axis.add_patch(Rectangle((note.start, note.midi - 0.35),
                                 note.end - note.start, 0.7,
                                 fill=False, edgecolor=color, linewidth=1.8))
        if label:
            tag = {"slurred": "S", "tied": "T", "articulated": "A", "phrase_end": "end"}
            axis.text(max(lo, note.start) + 0.015, note.midi + 0.65,
                      f"{librosa.midi_to_note(note.midi, unicode=False)} {tag[note.link_to_next]}",
                      color=color, fontsize=7, clip_on=True)
    finite = pitch[np.isfinite(pitch)]
    low, high = (float(np.percentile(finite, 2)), float(np.percentile(finite, 98)))
    axis.set(xlim=(lo, hi), ylim=(max(36, np.floor(low) - 5), min(83, np.ceil(high) + 5)),
             ylabel="MIDI pitch")
    axis.grid(alpha=0.12)


def render_plots(
    output: Path, cqt_db: np.ndarray, cqt_times: np.ndarray,
    times: np.ndarray, pitch: np.ndarray, db: np.ndarray,
    detectors: dict[str, Detector], all_notes: dict[str, list[Note]],
    duration: float, source_start: float,
) -> None:
    fig, axes = plt.subplots(len(detectors), 1, figsize=(22, max(4, 3.4 * len(detectors))),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, (name, detector) in zip(axes, detectors.items()):
        draw_cqt(axis, cqt_db, cqt_times, times, pitch, detector, all_notes[name],
                 0, duration, label=False)
        onset_label = f"{len(detector.times)} raw onsets"
        if detector.extra_onsets is not None:
            onset_label = (f"{len(detector.times) - len(detector.extra_onsets)} detector + "
                           f"{len(detector.extra_onsets)} energy-backed onsets")
        axis.set_title(f"{METHODS[name]} - {onset_label} / "
                       f"{len(all_notes[name])} notes", fontsize=11)
        axis.set_xlabel("Crop seconds")
    policy_caption = ("\nSyllable mode: minimum 10ms articulation rendering gap, not measured silence."
                      if any(d.parameters.get("syllabic") for d in detectors.values()) else "")
    fig.suptitle(f"Same CQT and CREPE pitch for all methods | source starts {source_start:.3f}s\n"
                 "Cyan: pitch | orange dashed: onset | blue dashed: energy-backed onset | boxes: exported notes "
                 "(green slurred / yellow tied / pink articulated / white phrase end)" + policy_caption,
                 fontsize=13)
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)
    segments = int(np.ceil(duration / 10))
    for name, detector in detectors.items():
        fig, axes = plt.subplots(segments * 2, 1, figsize=(19, 5 * segments),
                                 gridspec_kw={"height_ratios": [4, 1] * segments},
                                 constrained_layout=True)
        axes = np.atleast_1d(axes)
        for section in range(segments):
            lo, hi = section * 10, min(duration, (section + 1) * 10)
            axis, evidence = axes[section * 2:section * 2 + 2]
            draw_cqt(axis, cqt_db, cqt_times, times, pitch, detector, all_notes[name], lo, hi)
            axis.set_title(f"{METHODS[name]} | crop {lo:g}-{hi:g}s | "
                           f"source {source_start + lo:.3f}-{source_start + hi:.3f}s")
            evidence.plot(times, db, color="#333333", linewidth=1, label="RMS relative dB")
            evidence.set(xlim=(lo, hi), ylim=(-60, 3), ylabel="RMS dB",
                         xlabel="Crop seconds (not beat-quantized)")
            right = evidence.twinx()
            right.plot(detector.envelope_times, detector.envelope,
                       color="#c77700", linewidth=0.8)
            right.set(ylim=(0, 1.05), ylabel="Onset activation")
            evidence.grid(alpha=0.2)
        gap_caption = ("Syllable mode: minimum 10ms rendering gap, not measured silence; pitch-only changes connect."
                       if detector.parameters.get("syllabic") else "Gaps are acoustic estimates, not ground truth.")
        fig.suptitle("Cyan pitch / orange onset / blue energy-backed onset / note boxes: "
                     "S=slurred, T=tied, A=articulated, end=phrase end\n"
                     "Box color describes its connection TO THE NEXT note. "
                     + gap_caption, fontsize=12)
        fig.savefig(output / f"{name}_cqt.png", dpi=150)
        plt.close(fig)


def write_review(output: Path, report: dict) -> None:
    cards = []
    for name, stats in report["methods"].items():
        counts = stats["links"]
        embedded_audio = base64.b64encode((output / f"{name}_AB.flac").read_bytes()).decode("ascii")
        cards.append(f"""
        <section id="{name}"><h2>{html.escape(METHODS[name])}</h2>
        <p>{stats['raw_onsets']} raw onsets; {stats['notes']} notes.
        Connections: {counts['slurred']} slurred, {counts['tied']} tied,
        {counts['articulated']} articulated. Median note: {stats['median_duration_seconds']:.3f}s.</p>
        <p><strong>Left vocal / right synth:</strong>
        <audio controls preload="none" data-audio-file="{name}_AB.flac"
        src="data:audio/flac;base64,{embedded_audio}"></audio></p>
        <p><a href="{name}_AB.wav">Download stereo WAV</a> &middot;
        <a href="{name}_AB.flac">Download lossless FLAC</a></p>
        <p><a href="{name}.mid">MIDI</a> &middot; <a href="{name}_notes.csv">Note table</a> &middot;
        <a href="{name}_onsets.csv">Raw onsets</a> &middot;
        <a href="{name}_cqt.png">Full-resolution plot (zoom in)</a></p>
        <a href="{name}_cqt.png"><img loading="lazy" src="{name}_cqt.png"
        alt="{html.escape(METHODS[name])} CQT, pitch, onset and note overlays"></a></section>""")
    start = report["window"]["start_seconds"]
    end = report["window"]["end_seconds"]
    fixed = report["window"].get("mode") == "fixed"
    selection_title = "Selected excerpt" if fixed else "Loudest continuous"
    selection_description = (
        "selected by user-specified source time, rounded to the original sample grid."
        if fixed else "selected by mean channel power over every original-sample-aligned window."
    )
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Vocal MIDI {start:.3f}-{end:.3f}s</title>
    <style>body{{font:16px system-ui;background:#17191d;color:#eee;max-width:1500px;
    margin:auto;padding:24px}}a{{color:#72c8ff}}img{{width:100%;height:auto}}
    section{{border-top:1px solid #555;margin-top:36px;padding-top:12px}}
    audio{{vertical-align:middle;width:min(500px,100%)}}code{{color:#ffd166}}</style>
    <h1>Vocal MIDI onset and release experiment</h1>
    <p><strong>{selection_title} {end-start:g}s:</strong> source {start:.6f}-{end:.6f}s,
    {selection_description}
    Original file: {html.escape(report['source']['path'])}.</p>
    <p>All methods share the same uncorrected CREPE full pitch, voice regions and release policy.
    No beat-grid snapping, key correction, manual labels, or accuracy claims.
    Stereo synth plays the exported note intervals, not the continuous pitch contour.
    Use headphones and pan left/right to compare. Audio stays local.
    Players use embedded lossless FLAC so they also work in local-file browser previews.</p>
    <p><strong>Legend:</strong> cyan = measured pitch; orange dashed = raw detector onset;
    rectangles = exported notes. Green S = different-pitch slur; yellow T = same-pitch tie;
    pink A = gap/articulation; white end = phrase end.
    Colors describe the connection <em>to the next note</em>.
    T/S share an exact MIDI tick; A has a positive gap. Not every dashed line becomes a note.</p>
    <p>Long voicing gaps end phrases; short pre-attack voicing gaps (20ms) or energy dips
    (6dB below the prior body and 4dB below recovery for 20ms) shorten note ends.
    Otherwise adjacent notes remain connected. Consonants, breath, vibrato, reverb and
    overlapping singers can fool these estimates. A gap does not prove expressive intent.</p>
    <p>madmom CNN is the pretrained Schlueter/Boeck multi-resolution CNN; BRNN is a separate
    eight-model recurrent ensemble, <em>not another CNN</em>. SuperFlux uses librosa's
    documented maximum-filter recipe, not an assertion of bit-identical original code.
    The pitch-aware variant adds sustained pitch changes and requires energy dip/recovery
    for SuperFlux same-pitch retriggers.</p>
    <p><a href="vocal_excerpt.wav">Original excerpt</a> &middot; <a href="pitch.csv">Pitch frames</a> &middot;
    <a href="report.json">Parameters and provenance</a> &middot;
    <a href="validation.json">Persisted output checks</a> &middot;
    <a href="comparison.png">All-method overview</a></p>
    <p>madmom model weights are
    <a href="https://github.com/CPJKU/madmom_models">CC BY-NC-SA 4.0</a>.
    Do not incorporate them into a commercial app without appropriate permission.
    This experiment does not modify the production app.</p>
    <a href="comparison.png"><img src="comparison.png" alt="All onset methods over the same CQT"></a>
    {''.join(cards)}</html>"""
    (output / "review.html").write_text(page, encoding="utf-8")


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs" / "vocals_30s")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--start", type=float,
                        help="Start in original-source seconds; omit to select the loudest window.")
    parser.add_argument("--bpm", type=float, default=150)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or not np.isfinite(args.bpm) or args.bpm <= 0:
        parser.error("Duration and tempo must be finite; tempo must be positive.")
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    audio, native_sr = sf.read(source, dtype="float32", always_2d=True)
    fixed = args.start is not None
    a, b, rms = (
        fixed_window(audio, native_sr, args.seconds, args.start)
        if fixed else loudest_window(audio, native_sr, args.seconds)
    )
    duration, source_start = (b - a) / native_sr, a / native_sr
    vocal = audio[a:b].mean(axis=1)
    pad_a, pad_b = max(0, a - native_sr), min(len(audio), b + native_sr)
    context = audio[pad_a:pad_b].mean(axis=1)
    shift = (a - pad_a) / native_sr
    analysis = librosa.resample(context, orig_sr=native_sr, target_sr=SR)
    selection_title = "Selected window" if fixed else "Loudest window"
    print(f"{selection_title}: {source_start:.6f}-{b/native_sr:.6f}s; RMS {rms:.6f}", flush=True)
    print("Tracking CREPE full pitch with one second context...", flush=True)
    torch.set_num_threads(4)
    pitch_started = time.perf_counter()
    pitch_times, pitch = track_pitch(analysis, SR, "vocals", "crepe")
    pitch_runtime = time.perf_counter() - pitch_started
    rms_frames = librosa.feature.rms(y=analysis, frame_length=1024, hop_length=HOP)[0]
    energy_times = librosa.times_like(rms_frames, sr=SR, hop_length=HOP)
    db_frames = librosa.amplitude_to_db(rms_frames + 1e-9, ref=np.max(rms_frames) + 1e-9)
    db = np.interp(pitch_times, energy_times, db_frames)
    keep = (pitch_times >= shift) & (pitch_times < shift + duration)
    pitch_times, pitch, db = pitch_times[keep] - shift, pitch[keep], db[keep]
    if np.isfinite(pitch).sum() < 10:
        raise RuntimeError("CREPE found too few voiced frames for a meaningful comparison.")
    regions = voice_regions(pitch_times, pitch, db, duration)
    detectors = {"librosa": flux_detector(analysis, False),
                 "superflux": flux_detector(analysis, True)}
    detectors.update(neural_detectors(context, native_sr))
    for detector in detectors.values():
        detector.times = detector.times - shift
        detector.times = detector.times[(detector.times >= 0) & (detector.times < duration)]
        detector.envelope_times = detector.envelope_times - shift
    detectors["pitch_aware"] = custom_detector(
        detectors["superflux"], pitch_times, pitch, db, regions,
    )
    cqt = np.abs(librosa.cqt(
        analysis, sr=SR, hop_length=HOP, fmin=librosa.midi_to_hz(36),
        n_bins=144, bins_per_octave=36,
    ))
    cqt_times = librosa.times_like(cqt, sr=SR, hop_length=HOP) - shift
    cqt_db = librosa.amplitude_to_db(cqt, ref=np.max(cqt))
    report = {
        "source": {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                   "sample_rate": native_sr, "channels": audio.shape[1],
                   "duration_seconds": len(audio) / native_sr},
        "window": {"start_sample": a, "end_sample": b, "start_seconds": source_start,
                   "end_seconds": b / native_sr, "duration_seconds": duration, "rms": rms,
                   "mode": "fixed" if fixed else "loudest",
                   "requested_start_seconds": args.start,
                   "selection": (
                       "user-specified source start and duration; original sample grid"
                       if fixed else "exact maximum sliding mean channel power; original sample grid"
                   )},
        "pitch": {"tracker": "existing core.mono_pitch.track_pitch CREPE full",
                  "runtime_seconds": pitch_runtime, "hop_seconds": 0.01,
                  "torch_cpu_threads": torch.get_num_threads(),
                  "context_seconds": 1, "key_correction": False,
                  "voiced_frames": int(np.isfinite(pitch).sum()),
                  "voiced_fraction": float(np.isfinite(pitch).mean())},
        "note_policy": {"min_note_seconds": MIN_NOTE, "voice_gate_relative_db": -35,
                        "phrase_release_gap_seconds": RELEASE_GAP,
                        "articulation_voicing_gap_seconds": 0.02,
                        "articulation_dip_db": 6, "articulation_recovery_db": 4,
                        "bpm": args.bpm, "ticks_per_beat": 9600,
                        "beat_quantization": False,
                        "connection": "end==next start: different pitch slurred, same pitch tied"},
        "voice_regions": regions, "methods": {},
        "versions": {p: importlib.metadata.version(p) for p in (
            "numpy", "scipy", "librosa", "madmom", "torch", "torchcrepe", "soundfile", "mido",
        )},
        "python": platform.python_version(),
        "limitations": ["No labeled truth: counts are descriptive, not accuracy.",
                        "Dominant monophonic line only; overlapping singers may confuse pitch.",
                        "Acoustic gap/connection is not proof of musical articulation intent.",
                        "madmom weights are CC BY-NC-SA 4.0; commercial integration needs review."],
    }
    sf.write(output / "vocal_excerpt.wav", audio[a:b], native_sr, subtype="PCM_24")
    write_csv(output / "pitch.csv", ["crop_seconds", "source_seconds", "midi", "rms_db"],
              [{"crop_seconds": float(t), "source_seconds": float(t + source_start),
                "midi": float(p) if np.isfinite(p) else "", "rms_db": float(d)}
               for t, p, d in zip(pitch_times, pitch, db)])
    all_notes = {}
    for name, detector in detectors.items():
        notes = make_notes(detector, pitch_times, pitch, db, regions)
        if not notes:
            raise RuntimeError(f"{name} produced no playable notes.")
        all_notes[name] = notes
        synth = synthesize(notes, native_sr, len(vocal))
        stereo, gains = stereo_comparison(vocal, synth)
        sf.write(output / f"{name}_AB.wav", stereo, native_sr, subtype="PCM_24")
        sf.write(output / f"{name}_AB.flac", stereo, native_sr, subtype="PCM_24")
        export_midi(notes, output / f"{name}.mid", args.bpm)
        write_csv(output / f"{name}_notes.csv",
                  [*asdict(notes[0]), "note_name", "source_start_seconds", "source_end_seconds"],
                  [{**asdict(n), "note_name": librosa.midi_to_note(n.midi, unicode=False),
                    "source_start_seconds": n.start + source_start,
                    "source_end_seconds": n.end + source_start} for n in notes])
        write_csv(output / f"{name}_onsets.csv", ["crop_seconds", "source_seconds"],
                  [{"crop_seconds": float(t), "source_seconds": float(t + source_start)}
                   for t in detector.times])
        links = {label: sum(n.link_to_next == label for n in notes)
                 for label in ("slurred", "tied", "articulated", "phrase_end")}
        report["methods"][name] = {
            "raw_onsets": len(detector.times), "notes": len(notes), "links": links,
            "median_duration_seconds": float(np.median([n.end - n.start for n in notes])),
            "parameters": detector.parameters, "comparison_audio": gains,
            "notes_detail": [asdict(n) for n in notes],
        }
        print(f"{name}: {len(detector.times)} onsets / {len(notes)} notes; {links}", flush=True)
    render_plots(output, cqt_db, cqt_times, pitch_times, pitch, db,
                 detectors, all_notes, duration, source_start)
    report["runtime_seconds"] = time.perf_counter() - started
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False),
                                       encoding="utf-8")
    write_review(output, report)
    print(f"Review: {output / 'review.html'}", flush=True)


if __name__ == "__main__":
    main()
