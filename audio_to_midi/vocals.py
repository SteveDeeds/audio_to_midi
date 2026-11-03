"""Neural syllable attacks and connected pitch changes, bounded to a requested view."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
import math

import librosa
import numpy as np

from audio_to_midi.domain import Note
from audio_to_midi.timebase import Grid
from audio_to_midi.vocal_settings import VOCAL_MODELS, VocalParameters
from core.chord_analysis import key_to_scale_pcs
from core.mono_pitch import pitch_agreement, track_pitch
from core.vocal_boundaries import BoundarySettings, coalesce, energy_retriggers, stable_changes, voice_regions

SR, HOP = 22050, 220


def check_cancel(cancelled: Callable[[], bool]) -> None:
    if cancelled():
        from audio_to_midi.adapters import Cancelled
        raise Cancelled()


@dataclass(frozen=True)
class VocalEvidence:
    times: np.ndarray
    pitch: np.ndarray
    db: np.ndarray
    activation_times: np.ndarray
    activation: np.ndarray
    flux_times: np.ndarray
    flux: np.ndarray


def grid_notes(
    evidence: VocalEvidence, attacks: np.ndarray, grid: Grid, start: Fraction, end: Fraction,
    settings: BoundarySettings, track_id: str, key: str = "",
    labels: dict[Fraction, bool] | None = None,
) -> tuple[list[Note], list[Fraction]]:
    """Only gaps between articulated syllables are half a grid cell; slurs touch."""
    if len(evidence.times) < 2:
        return [], []
    times, pitch, db = evidence.times, evidence.pitch, evidence.db
    feedback = labels or {}
    regions = voice_regions(times, pitch, db, grid.beat_to_seconds(end), settings)
    syllables = {
        grid.snap(grid.seconds_to_beat(float(t))) for t in attacks
        if grid.beat_to_seconds(start) <= t < grid.beat_to_seconds(end)
    }
    syllables.update(b for b, keep in feedback.items() if keep and start <= b < end)
    changes = {
        grid.snap(grid.seconds_to_beat(float(t))) for t in stable_changes(times, pitch, settings)
    }
    vetoes = {b for b, keep in feedback.items() if not keep}
    syllables.difference_update(vetoes)
    changes.difference_update(vetoes | syllables)
    spans: list[list[Fraction]] = []
    for a, b in regions:
        nearby = attacks[np.abs(attacks - a) <= 0.10]
        first = float(nearby[np.argmin(np.abs(nearby - a))]) if len(nearby) else a
        begin = max(start, grid.snap(grid.seconds_to_beat(first)))
        stop = min(end, grid.snap_end(grid.seconds_to_beat(b)))
        if begin >= stop:
            continue
        if begin not in vetoes:
            syllables.add(begin)
        if spans and begin <= spans[-1][1]:
            spans[-1][1] = max(stop, spans[-1][1])
        else:
            spans.append([begin, stop])
    scale = key_to_scale_pcs(key) if key else None

    def measure(a: Fraction, b: Fraction) -> Note | None:
        t0, t1 = grid.beat_to_seconds(a), grid.beat_to_seconds(b)
        body = (times >= t0 + min(0.04, (t1 - t0) / 4)) & (times < t1)
        values = pitch[body & np.isfinite(pitch)]
        if len(values) < 3:
            return None
        raw = float(np.median(values))
        midi = int(np.clip(round(raw), 0, 127))
        if scale is not None and midi % 12 not in scale:
            candidates = [p for p in range(max(0, midi - 1), min(127, midi + 1) + 1)
                          if p % 12 in scale]
            if candidates:
                midi = min(candidates, key=lambda p: (abs(p - raw), p))
        velocity = int(np.clip(100 + 1.2 * float(np.median(db[body])), 45, 110))
        return Note(f"{track_id}:vocal:{a}:{midi}", a, b, midi, velocity,
                    confidence=pitch_agreement(pitch[body], midi))

    notes: list[Note] = []
    half = grid.subdivision / 2
    for begin, stop in spans:
        boundaries = sorted({begin, stop} | {b for b in syllables | changes if begin < b < stop})
        for a, b in zip(boundaries, boundaries[1:]):
            if grid.beat_to_seconds(b) - grid.beat_to_seconds(a) < settings.min_note - 1e-9:
                continue
            release = b - half if b in syllables and b != stop else b
            if release <= a:
                continue
            note = measure(a, release)
            if note is None:
                continue
            if (notes and a in changes and notes[-1].end == a and notes[-1].pitch == note.pitch):
                previous = notes.pop()
                note = measure(previous.start, release)
            if note is not None:
                notes.append(note)
    bounded = []
    for index, note in enumerate(notes):
        if index + 1 < len(notes):
            following = notes[index + 1]
            stop = following.start - half if following.start in syllables else following.start
            note = replace(note, end=min(note.end, stop)) if stop > note.start else note
        if note.end > note.start:
            bounded.append(note)
    return bounded, sorted({n.start for n in bounded if n.start in syllables})


class VocalTranscriber:
    """Cache only visible-window evidence; retuning does not rerun neural models."""

    def __init__(self, audio: np.ndarray, sr: int, model: str, key: str):
        if model not in VOCAL_MODELS:
            raise ValueError(f"Unknown neural vocal model: {model}")
        self.audio, self.sr, self.model, self.key = audio, sr, model, key
        self._cache: dict[tuple[int, int], VocalEvidence] = {}
        self._processor = None

    def _pitch(self, audio: np.ndarray, cancelled: Callable[[], bool]) -> tuple[np.ndarray, np.ndarray]:
        import torch

        torch.set_num_threads(4)
        times, pitches = [], []
        for first in range(0, len(audio), SR * 2):
            check_cancel(cancelled)
            last = min(len(audio), first + SR * 2)
            a, b = max(0, first - SR // 4), min(len(audio), last + SR // 4)
            t, p = track_pitch(audio[a:b], SR, "vocals", "crepe_full")
            t = t + a / SR
            keep = (t >= first / SR - 1e-9) & (t < last / SR - 1e-9)
            times.append(t[keep])
            pitches.append(p[keep])
        check_cancel(cancelled)
        return np.concatenate(times), np.concatenate(pitches)

    def _activation(self, audio: np.ndarray) -> np.ndarray:
        from audio_to_midi.model_evidence import onset_evidence
        return onset_evidence(audio, SR, "vocals", self.model + "_full")["onset"]

    def evidence(self, t0: float, t1: float, cancelled: Callable[[], bool]) -> VocalEvidence:
        a, b = max(0, math.ceil(t0 * self.sr)), min(len(self.audio), math.ceil(t1 * self.sr))
        key = a, b
        if key in self._cache:
            return self._cache[key]
        check_cancel(cancelled)
        if b <= a:
            raise ValueError("Visible vocal interval has no audio.")
        y = librosa.resample(self.audio[a:b], orig_sr=self.sr, target_sr=SR)
        if (b - a) / self.sr < 0.03 or not np.any(y):
            empty = np.array([], dtype=float)
            return VocalEvidence(empty, empty, empty, empty, empty, empty, empty)
        times, pitch = self._pitch(y, cancelled)
        check_cancel(cancelled)
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]
        db = librosa.amplitude_to_db(rms + 1e-9, ref=np.max(rms) + 1e-9)
        db = np.interp(times, librosa.times_like(rms, sr=SR, hop_length=HOP), db)
        activation = self._activation(y)
        check_cancel(cancelled)
        mel = librosa.feature.melspectrogram(y=y, sr=SR, hop_length=HOP, n_fft=2048,
                                            n_mels=138, fmin=27.5, fmax=10000)
        flux = librosa.onset.onset_strength(S=librosa.power_to_db(mel, ref=np.max),
                                           sr=SR, hop_length=HOP, lag=2, max_size=3)
        flux = flux - flux.min()
        flux /= max(float(flux.max()), 1e-12)
        check_cancel(cancelled)
        offset = a / self.sr
        result = VocalEvidence(times + offset, pitch, db, np.arange(len(activation)) / 100 + offset,
                               activation, librosa.times_like(flux, sr=SR, hop_length=HOP) + offset, flux)
        self._cache[key] = result
        return result

    def transcribe(
        self, track_id: str, grid: Grid, start: Fraction, end: Fraction, parameters: VocalParameters,
        labels: dict[Fraction, bool], cancelled: Callable[[], bool],
    ) -> tuple[list[Note], list[Fraction]]:
        from madmom.features.onsets import OnsetPeakPickingProcessor

        check_cancel(cancelled)
        evidence = self.evidence(max(0, grid.beat_to_seconds(start)), grid.beat_to_seconds(end), cancelled)
        if len(evidence.times) < 2:
            return [], []
        attacks = OnsetPeakPickingProcessor(fps=100, threshold=parameters.neural_threshold, smooth=0,
                                           pre_max=0.03, post_max=0.03, combine=parameters.onset_wait)(evidence.activation)
        attacks += float(evidence.activation_times[0])
        flux_onsets = librosa.onset.onset_detect(
            onset_envelope=evidence.flux, sr=SR, hop_length=HOP, units="time", normalize=False,
            delta=0.12, wait=round(parameters.onset_wait * SR / HOP),
            pre_max=round(0.03 * SR / HOP), post_max=round(0.03 * SR / HOP),
            pre_avg=round(0.10 * SR / HOP), post_avg=round(0.10 * SR / HOP),
        ) + float(evidence.flux_times[0])
        settings = BoundarySettings(min_note=parameters.min_note, voice_db=parameters.voice_db,
                                    pitch_hold=parameters.pitch_hold)
        extra = energy_retriggers(flux_onsets, evidence.times, evidence.pitch, evidence.db, settings)
        extra = coalesce(np.asarray([t for t in extra if not np.any(np.abs(attacks - t) < parameters.onset_wait)]),
                         parameters.onset_wait)
        attacks = np.unique(np.concatenate((attacks, extra)))
        check_cancel(cancelled)
        return grid_notes(evidence, attacks, grid, start, end, settings, track_id, self.key, labels)
