"""Monophonic pitch provider for the onset-grid note model.

The app's candidate onsets own note boundaries. This module only supplies, for
the visible region, a cached key-cleaned f0 contour (CREPE or pYIN), a whole-stem
attack detector used as extra onset evidence, per-block pitch, and legato
pitch-change times.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import re

import librosa
import numpy as np
from scipy import ndimage

from core.analysis import detect_key_from_chroma
from core.chord_analysis import key_to_scale_pcs


def clean_key_contour(
    quantized: np.ndarray, raw: np.ndarray, allowed: np.ndarray, hop: float,
) -> np.ndarray:
    """Fold short unharmonised passing runs into adjacent stable pitches."""
    q = quantized.copy()
    runs: list[tuple[int, int, int, bool]] = []
    i = 0
    while i < len(q):
        if np.isnan(q[i]):
            i += 1
            continue
        j = i + 1
        while j < len(q) and q[j] == q[i]:
            j += 1
        pitch = int(q[i])
        keep = allowed[pitch % 12, i:j].mean() >= 0.5 or (j - i) * hop >= 0.25
        runs.append((i, j, pitch, bool(keep)))
        i = j
    for k, (start, end, pitch, keep) in enumerate(runs):
        if keep:
            continue
        target = float(np.nanmedian(raw[start:end]))
        candidates: list[int] = []
        for step in (1, -1):
            m = k + step
            while 0 <= m < len(runs):
                contiguous = (
                    runs[m][0] == runs[m - 1][1] if step == 1
                    else runs[m + 1][0] == runs[m][1]
                )
                if not contiguous:
                    break
                if runs[m][3]:
                    candidates.append(runs[m][2])
                    break
                m += step
        if not candidates:
            frame_ok = allowed[:, start:end].mean(axis=1) >= 0.5
            candidates = [
                pitch + d for d in (-1, 1, -2, 2) if frame_ok[(pitch + d) % 12]
            ]
        if candidates:
            q[start:end] = min(candidates, key=lambda p: abs(p - target))
    return q


def track_pitch(y: np.ndarray, sr: int, stem: str, tracker: str) -> tuple[np.ndarray, np.ndarray]:
    fmin, fmax = (30.0, 420.0) if stem == "bass" else (70.0, 1100.0)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=256)[0]
    rms_times = librosa.times_like(rms, sr=sr, hop_length=256)
    db = librosa.amplitude_to_db(rms + 1e-9, ref=np.max(rms) + 1e-9)
    if tracker == "pyin":
        f0, voiced, _ = librosa.pyin(
            y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=256,
        )
        times = librosa.times_like(f0, sr=sr, hop_length=256)
    else:
        try:
            import torch
            import torchcrepe
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "CREPE requires torchcrepe. Install the app requirements in its Python environment."
            ) from exc
        audio = torch.from_numpy(librosa.resample(y, orig_sr=sr, target_sr=16000))[None]
        with torch.inference_mode():
            f0_tensor, periodicity = torchcrepe.predict(
                audio, 16000, hop_length=160, fmin=max(fmin, 32.71), fmax=fmax,
                model="tiny" if tracker == "crepe_tiny" else "full",
                decoder=torchcrepe.decode.viterbi, return_periodicity=True,
                batch_size=1024, device="cpu",
            )
            voiced = torchcrepe.filter.median(periodicity, 3)[0].numpy() > 0.35
            f0 = f0_tensor[0].numpy()
        times = np.arange(len(f0)) * 0.01
    voiced = voiced & (np.interp(times, rms_times, db) > -45.0)
    midi = librosa.hz_to_midi(np.maximum(np.nan_to_num(f0, nan=1.0), 1.0))
    return times, np.where(voiced, midi, np.nan)


class MonoTranscriber:
    """Analyze missing visible audio in interruptible batches; queries read the cache."""

    CHUNK_SECONDS = 2.0
    CONTEXT_SECONDS = 0.25
    SAMPLE_RATE = 22050
    ONSET_DELTA = 0.2
    ATTACK_SECONDS = 0.06
    RELEASE_GAP_SECONDS = 0.1
    MIN_VOICED_SECONDS = 0.05
    MIN_VOICED_FRACTION = 0.3
    LEGATO_HOLD_SECONDS = 0.12

    def __init__(self, audio: np.ndarray, sr: int, wav_path: str):
        self.audio = audio
        self.sr = sr
        self.wav_path = Path(wav_path)
        self.duration = len(audio) / sr
        self._contours: dict[
            tuple[str, str, tuple[float, float]], tuple[np.ndarray, np.ndarray, np.ndarray],
        ] = {}
        self._clean: dict[tuple[str, str], tuple[tuple[np.ndarray, np.ndarray], ...]] = {}
        self._onsets: dict[str, np.ndarray] = {}
        self._context: tuple[set[int], np.ndarray, np.ndarray] | None = None
        self._context_window: tuple[float, float] | None = None

    def _key_context(self, start: float, end: float) -> tuple[set[int], np.ndarray, np.ndarray]:
        if self._context is not None and (
            self._context_window is None or self._context_window == (start, end)
        ):
            return self._context
        mix = librosa.resample(
            self.audio[int(start * self.sr):int(end * self.sr)],
            orig_sr=self.sr, target_sr=self.SAMPLE_RATE,
        ).copy()
        harmony_audio = None
        for path in sorted(self.wav_path.parent.glob("*.wav")):
            if path.resolve() == self.wav_path.resolve():
                continue
            name = path.stem.lower()
            stem = next((
                s for s in ("bass", "vocals", "other", "piano", "guitar")
                if name == s or re.search(rf"\({s}\)$", name)
            ), None)
            if stem is None:
                continue
            y, _ = librosa.load(
                path, sr=self.SAMPLE_RATE, mono=True, offset=start, duration=end - start,
            )
            n = min(len(y), len(mix))
            mix[:n] += y[:n]
            if stem == "other":
                harmony_audio = y
        chroma = librosa.feature.chroma_cqt(y=mix, sr=self.SAMPLE_RATE, hop_length=2048)
        key_text = detect_key_from_chroma(chroma.mean(axis=1))
        scale = key_to_scale_pcs(key_text)
        if key_text.split()[0].endswith("m"):
            tonic = librosa.note_to_midi(key_text.split()[0][:-1] + "4") % 12
            scale.add((tonic + 11) % 12)
        if harmony_audio is None:
            harmony_times, harmony = np.array([]), np.zeros((12, 0))
        else:
            harmony = librosa.feature.chroma_cqt(
                y=harmony_audio, sr=self.SAMPLE_RATE, hop_length=2048, norm=None,
            )
            harmony = ndimage.uniform_filter1d(
                harmony, size=max(1, int(0.8 * self.SAMPLE_RATE / 2048)), axis=1,
            )
            level = harmony.sum(axis=0)
            harmony /= harmony.max(axis=0, keepdims=True) + 1e-9
            harmony[:, level < 0.05 * np.percentile(level, 95)] = 0.0
            harmony_times = librosa.times_like(harmony, sr=self.SAMPLE_RATE, hop_length=2048) + start
        self._context = scale, harmony_times, harmony
        self._context_window = start, end
        return self._context

    def onset_times(self, stem: str, tracker: str | None = None) -> np.ndarray:
        """Attack times over the whole stem with one global normalisation.

        Normalising per visible chunk turned quiet bleed into onsets, so the
        detector runs once per stem and is cached.
        """
        cached = self._onsets.get(stem)
        if cached is not None:
            return cached
        sr = self.SAMPLE_RATE
        y = librosa.resample(self.audio, orig_sr=self.sr, target_sr=sr)
        if stem == "bass":
            hop = 128
            mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop, n_mels=128, fmax=8000)
            env = librosa.onset.onset_strength(
                S=librosa.power_to_db(mel, ref=np.max), sr=sr, hop_length=hop, lag=2, max_size=3,
            )
            wait = int(0.1 * sr / hop)
        else:
            hop = 256
            env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
            wait = 4
        env = env / (np.percentile(env, 99) + 1e-9) if len(env) else env
        frames = librosa.onset.onset_detect(
            onset_envelope=env, sr=sr, hop_length=hop, delta=self.ONSET_DELTA,
            wait=wait, normalize=False,
        )
        times = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
        self._onsets[stem] = np.asarray(times, dtype=float)
        return self._onsets[stem]

    def prepare(
        self, t_start: float, t_end: float, stem: str, tracker: str,
        should_cancel: Callable[[], bool] | None = None, max_batches: int | None = None,
    ) -> bool:
        """Track uncached parts of the window; True when the window is fully covered.

        Cancellation is only honoured between batches so a finished batch is
        always kept, otherwise continuous scrolling could starve slow trackers.
        max_batches lets callers publish progress between slow batches.
        """
        cancelled = should_cancel or (lambda: False)
        start = max(0.0, round(t_start * self.sr) / self.sr)
        end = min(self.duration, round(t_end * self.sr) / self.sr)
        if end <= start:
            return True
        if cancelled():
            return False
        self.onset_times(stem)
        missing = self._missing(start, end, stem, tracker)
        if not missing:
            return True
        scale, harmony_times, harmony = self._key_context(start, end)
        sr = self.SAMPLE_RATE
        batches = []
        for a, b in missing:
            while a < b:
                next_end = min(a + self.CHUNK_SECONDS, b)
                batches.append((a, next_end))
                a = next_end
        changed = False
        if max_batches is not None:
            batches = batches[:max(1, int(max_batches))]
        for a, b in batches:
            if cancelled():
                break
            padded_start = max(0.0, a - self.CONTEXT_SECONDS)
            padded_end = min(self.duration, b + self.CONTEXT_SECONDS)
            y = librosa.resample(
                self.audio[int(padded_start * self.sr):int(padded_end * self.sr)],
                orig_sr=self.sr, target_sr=sr,
            )
            if not np.any(y):
                times = np.arange(a, b, 0.01)
                midi = np.full(len(times), np.nan)
            else:
                times, midi = track_pitch(y, sr, stem, tracker)
                times = times + padded_start
                core = (times >= a) & (times < b)
                times, midi = times[core], midi[core]
            allowed = np.zeros((12, len(times)), dtype=bool)
            allowed[sorted(scale)] = True
            if len(harmony_times):
                indices = np.clip(np.searchsorted(harmony_times, times), 0, len(harmony_times) - 1)
                allowed |= harmony[:, indices] >= 0.8
            self._contours[(stem, tracker, (a, b))] = times, midi, allowed
            changed = True
        if changed:
            self._clean_cached(stem, tracker)
        return not self._missing(start, end, stem, tracker)

    def _missing(self, start: float, end: float, stem: str, tracker: str) -> list[tuple[float, float]]:
        covered = sorted(bounds for s, t, bounds in self._contours if (s, t) == (stem, tracker))
        missing = []
        cursor = start
        for a, b in covered:
            if a > cursor:
                missing.append((cursor, min(a, end)))
            cursor = max(cursor, b)
            if cursor >= end:
                break
        if cursor < end:
            missing.append((cursor, end))
        return [(a, b) for a, b in missing if b - a > 1.0 / self.sr]

    def covers(self, t_start: float, t_end: float, stem: str, tracker: str) -> bool:
        start = max(0.0, float(t_start))
        end = min(self.duration, float(t_end))
        return end <= start or not self._missing(start, end, stem, tracker)

    def _clean_cached(self, stem: str, tracker: str) -> None:
        intervals = sorted(bounds for s, t, bounds in self._contours if (s, t) == (stem, tracker))
        groups: list[list[tuple[float, float]]] = []
        for bounds in intervals:
            if groups and abs(bounds[0] - groups[-1][-1][1]) < 1.0 / self.sr:
                groups[-1].append(bounds)
            else:
                groups.append([bounds])
        cleaned_groups = []
        for group in groups:
            contours = [self._contours[(stem, tracker, bounds)] for bounds in group]
            times = np.concatenate([c[0] for c in contours])
            midi = np.concatenate([c[1] for c in contours])
            allowed = np.concatenate([c[2] for c in contours], axis=1)
            if not len(times):
                continue
            hop = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
            voiced = np.isfinite(midi)
            smoothed = ndimage.median_filter(np.where(voiced, midi, 0.0), size=5, mode="nearest")
            q = np.where(voiced, np.round(smoothed), np.nan)
            cleaned_groups.append((times, clean_key_contour(q, midi, allowed, hop)))
        self._clean[(stem, tracker)] = tuple(cleaned_groups)

    def contour(self, t_start: float, t_end: float, stem: str, tracker: str) -> tuple[np.ndarray, np.ndarray]:
        """Key-cleaned semitone contour (NaN = unvoiced) for cached frames in [t_start, t_end)."""
        parts = [
            (times[mask], pitch[mask])
            for times, pitch in self._clean.get((stem, tracker), ())
            for mask in [(times >= t_start) & (times < t_end)]
            if mask.any()
        ]
        if not parts:
            return np.array([]), np.array([])
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

    def sounding_fraction(self, t0: float, t1: float, stem: str, tracker: str) -> float:
        """Fraction of cached frames in [t0, t1) with a voiced pitch."""
        _, pitch = self.contour(t0, t1, stem, tracker)
        return float(np.isfinite(pitch).mean()) if len(pitch) else 0.0

    def block_pitch(
        self, t0: float, t1: float, stem: str, tracker: str, attack_start: float | None = None,
    ) -> tuple[list[int], float] | None:
        """Ranked pitches for one onset block and where its sound ends.

        The attack (from the precise onset when given) is skipped, the stable
        body votes for the pitch, and the release moves earlier when the tracker
        loses the pitch for longer than RELEASE_GAP_SECONDS.
        """
        times, pitch = self.contour(t0, t1, stem, tracker)
        if not len(times):
            return None
        hop = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
        voiced = np.isfinite(pitch)
        voiced_idx = np.flatnonzero(voiced)
        if not len(voiced_idx):
            return None
        release = float(t1)
        gap_frames = max(1, int(round(self.RELEASE_GAP_SECONDS / hop)))
        run = 0
        for i in range(int(voiced_idx[0]), len(voiced)):
            run = 0 if voiced[i] else run + 1
            if run >= gap_frames:
                release = float(times[i - run + 1])
                break
        body_start = max(float(t0), float(attack_start if attack_start is not None else t0))
        attack = min(self.ATTACK_SECONDS, 0.25 * (release - body_start))
        body = (times >= body_start + attack) & (times < release)
        if not np.any(body & voiced):
            body = times < release
        values = pitch[body & voiced]
        voiced_seconds = len(values) * hop
        if voiced_seconds < min(self.MIN_VOICED_SECONDS, 0.5 * max(float(t1) - float(t0), hop)) or (
            voiced_seconds < self.MIN_VOICED_FRACTION * (np.count_nonzero(body) * hop)
        ):
            return None
        counts: dict[int, int] = {}
        for value in values.astype(int):
            counts[int(value)] = counts.get(int(value), 0) + 1
        ranked = sorted(counts, key=lambda midi: (-counts[midi], midi))
        return ranked, release

    def pitch_changes(self, t_start: float, t_end: float, stem: str, tracker: str) -> list[float]:
        """Times where a held pitch moves to another held pitch without a voicing gap."""
        times, pitch = self.contour(t_start, t_end, stem, tracker)
        if len(times) < 2:
            return []
        hop = float(np.median(np.diff(times)))
        hold = max(1, int(round(self.LEGATO_HOLD_SECONDS / hop)))
        runs: list[tuple[int, int, float]] = []
        i = 0
        while i < len(pitch):
            j = i + 1
            while j < len(pitch) and (
                pitch[j] == pitch[i] or (np.isnan(pitch[j]) and np.isnan(pitch[i]))
            ):
                j += 1
            runs.append((i, j, float(pitch[i])))
            i = j
        changes = []
        for (a0, a1, pa), (b0, b1, pb) in zip(runs, runs[1:]):
            if np.isfinite(pa) and np.isfinite(pb) and a1 - a0 >= hold and b1 - b0 >= hold \
                    and times[b0] - times[a1 - 1] <= 1.5 * hop:
                changes.append(float(times[b0]))
        return changes
