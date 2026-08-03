"""
core/audio_engine.py
====================
AudioMarkers + TimeGrid — the data layer for audio analysis.
Manages raw audio, CQT, onsets, chroma, pitch cache, and ML model.
"""

import json
import os
import pathlib
import threading
from dataclasses import dataclass, replace
import numpy as np
import librosa
from scipy import optimize

from core.analysis import (
    load_audio, detect_onsets, detect_chroma_shifts,
    compute_cqt, compute_stft, compute_mel_spectrogram,
    detect_key_from_chroma, detect_key_from_file,
    HOP_SIZE, CHROMA_HOP, CHROMA_SHIFT_THRESH,
)
from core.models import CorrectionModel
from core.pitch_models import PitchModelPolicy, create_pitch_model
from core.pitch_adaptation_models import MultiAtomNNLSModel
from core.polyphonic_nnls import (
    build_register_aware_dictionary,
    auto_calibrate_by_register,
    solve_polyphonic_activations,
    compute_flux_frame,
    solve_single_frame,
)


@dataclass(frozen=True)
class PitchedNoteSpan:
    note_id: str
    onset: float
    release: float
    midi: int
    source: str
    energy: float
    block_keys: tuple[tuple[int, int], ...] = ()


def _compute_am_band_beatiness(cqt_mag, f0_frame, f1_frame, sr):
    """Per-bin beatiness via AM-band energy ratio.

    For each bin, extract envelope (magnitude over time in block window),
    detrend via high-pass filter, compute FFT, and measure energy ratio
    in the 2–12 Hz beat band vs the full low-frequency spectrum.
    This detects periodic wobble without being fooled by single dropouts.

    Returns: beatiness score per bin [0, 1].
    """
    n_bins = cqt_mag.shape[0]
    window = cqt_mag[:, f0_frame:f1_frame]
    if window.shape[1] < 4:
        return np.zeros(n_bins, dtype=np.float32)

    # Per-bin envelope is the time series across the block.
    # Detrend each envelope via simple moving-average high-pass.
    ma_len = max(1, window.shape[1] // 4)  # ~25% of block duration
    detrended = np.zeros_like(window, dtype=np.float64)
    for i in range(window.shape[1]):
        lo = max(0, i - ma_len // 2)
        hi = min(window.shape[1], i + ma_len // 2 + 1)
        baseline = window[:, lo:hi].mean(axis=1)
        detrended[:, i] = np.maximum(0.0, window[:, i].astype(np.float64) - baseline)

    # Compute FFT of detrended envelope for each bin.
    frame_rate = sr / HOP_SIZE  # frames per second
    fft_len = int(2 ** np.ceil(np.log2(window.shape[1])))
    fft_mag = np.abs(np.fft.rfft(detrended, n=fft_len, axis=1))
    freqs = np.fft.rfftfreq(fft_len, d=1.0 / frame_rate)  # Hz

    # Energy ratio: beat band (2–12 Hz) vs full low spectrum (0.5–20 Hz).
    beat_band = (freqs >= 2.0) & (freqs <= 12.0)
    full_band = (freqs >= 0.5) & (freqs <= 20.0)
    beat_energy = fft_mag[:, beat_band].sum(axis=1)
    full_energy = fft_mag[:, full_band].sum(axis=1)

    # Beatiness = normalized ratio; high ratio => strong beating.
    safe_full = np.maximum(full_energy, 1e-10)
    return (beat_energy / safe_full).astype(np.float32)


class TimeGrid:
    """Converts between seconds and bar.beat positions."""

    def __init__(self, bpm: float, beats_per_bar: int = 4, beat_unit: int = 4):
        self.bpm = bpm
        self.beats_per_bar = beats_per_bar
        self.beat_unit = beat_unit
        self.beat_dur = (60.0 / bpm) * (4.0 / beat_unit)
        self.bar_dur = beats_per_bar * self.beat_dur

    def seconds_to_bar_beat(self, t: float) -> tuple[int, int]:
        total_beats = t / self.beat_dur
        bar = int(total_beats // self.beats_per_bar) + 1
        beat = int(total_beats % self.beats_per_bar) + 1
        return bar, beat

    def label(self, t: float) -> str:
        bar, beat = self.seconds_to_bar_beat(t)
        return f"{bar}.{beat}"

    def beat_ticks(self, t_start: float, t_end: float) -> np.ndarray:
        first = int(t_start / self.beat_dur)
        last = int(t_end / self.beat_dur) + 1
        ticks = np.arange(first, last + 1) * self.beat_dur
        return ticks[(ticks >= t_start) & (ticks <= t_end)]

    def bar_ticks(self, t_start: float, t_end: float) -> np.ndarray:
        first = int(t_start / self.bar_dur)
        last = int(t_end / self.bar_dur) + 1
        ticks = np.arange(first, last + 1) * self.bar_dur
        return ticks[(ticks >= t_start) & (ticks <= t_end)]

    def zoom_duration(self, bars: int = 2) -> float:
        return bars * self.bar_dur


class AudioMarkers:
    """Holds raw audio, all detected PNCPs with strengths, and a filtered view."""

    NOTE_DURATION_OPTIONS: dict[str, float] = {
        "1/2": 2.0, "1/8": 0.5, "1/16": 0.25,
    }
    DEFAULT_DURATION = "1/16"
    MODEL_PROFILE_DIR = "model_profiles"
    _VEL_MIN = 24
    _VEL_MAX = 120
    _VEL_HARMONICS = 5
    _CAND_KEY_SCALE = 10000.0

    def __init__(self, wav_path: str, grid: TimeGrid,
                 progress_callback=None, bootstrap_from_profiles: bool = False):
        """Load audio and run all detection.

        progress_callback: optional callable(percent: int, message: str)
        """
        self.wav_path = wav_path
        self.grid = grid
        _emit = progress_callback or (lambda p, m: print(m))

        _emit(0, f"Loading: {wav_path}")
        self.audio, self.sr = load_audio(wav_path)
        self.duration = len(self.audio) / self.sr
        self.times = np.linspace(0, self.duration, len(self.audio))

        # Transient envelope
        smooth_len = max(16, int(0.028 * self.sr))
        rectified = np.abs(self.audio).astype(float)
        if len(rectified) >= smooth_len:
            kernel = np.ones(smooth_len, dtype=float) / float(smooth_len)
            smooth_env = np.convolve(rectified, kernel, mode="same")
            sample_idx = np.arange(0, len(smooth_env), HOP_SIZE, dtype=int)
            self._transient_env_np = smooth_env[sample_idx].astype(float)
            self._transient_times_np = (sample_idx.astype(float) / float(self.sr))
        else:
            self._transient_env_np = np.array([0.0], dtype=float)
            self._transient_times_np = np.array([0.0], dtype=float)

        _emit(10, "Detecting transient onsets …")
        onset_samples, onset_strengths = detect_onsets(self.audio, self.sr)
        self._onset_times_np = (onset_samples / self.sr).astype(float)
        self._onset_strengths_np = onset_strengths.astype(float)

        _emit(25, "Computing chroma …")
        self._chroma_full = librosa.feature.chroma_cqt(
            y=self.audio, sr=self.sr, hop_length=CHROMA_HOP, bins_per_octave=36)
        _rms = librosa.feature.rms(
            y=self.audio, hop_length=CHROMA_HOP, frame_length=2048, center=True)[0]
        _rms_db = librosa.amplitude_to_db(np.maximum(_rms, 1e-8), ref=np.max)
        self._chroma_times = librosa.frames_to_time(
            np.arange(self._chroma_full.shape[1]), sr=self.sr, hop_length=CHROMA_HOP)

        _emit(40, "Detecting chroma shifts …")
        chroma_samples, _ = detect_chroma_shifts(
            self.audio, self.sr, threshold=None,
            chroma=self._chroma_full, rms_db=_rms_db)

        _emit(55, "Computing CQT …")
        # Extend low-end coverage to C0 while keeping the previous top note range.
        self.cqt_fmin_midi = 12
        self.cqt_n_bins = 83
        self.cqt_db, self.cqt_sal, self.cqt_mag, self.cqt_peel_db, self.cqt_times, _, \
            self.cqt_hcqt_db, self.cqt_flux_db = compute_cqt(
            self.audio, self.sr, self.cqt_fmin_midi, self.cqt_n_bins)
        self.stft_db, self.stft_mag, self.stft_times, self.stft_freqs = compute_stft(
            self.audio, self.sr)
        self.mel_db, self.mel_mag, self.mel_times, self.mel_freqs = compute_mel_spectrogram(
            self.audio, self.sr)

        _emit(80, "Detecting key …")
        self.detected_key = self._detect_key_from_piano_stem()

        # Filtered views
        self.onset_times: np.ndarray = np.array([])
        self.chroma_times: np.ndarray = np.array([])
        self._pitch_cache: dict[tuple[int, int], tuple] = {}
        self._manual_pitches: dict[tuple[int, int], tuple[int, ...] | None] = {}
        self._explicit_note_spans: dict[str, PitchedNoteSpan] = {}
        self._next_explicit_note_id = 1
        self._generated_note_spans_cache: tuple[PitchedNoteSpan, ...] | None = None
        self._pitch_training_lock = threading.Lock()
        self._pending_pitch_training: list[tuple[tuple[int, int], int, bool]] = []
        self._pitch_edit_generation = 0
        self._note_spans_revision = 0
        self._onset_training_lock = threading.Lock()
        self._pending_onset_training: list[tuple] = []
        self._onset_edit_generation = 0
        self._manual_drums: dict[int, set[int]] = {}
        self._manual_drum_rejects: dict[int, set[int]] = {}
        self._candidate_active_keys: set[int] = set()
        self._candidate_forced_on_keys: set[int] = set()
        self._candidate_forced_off_keys: set[int] = set()
        self._candidate_raw_times: list[float] = []
        self._candidate_features: dict[int, tuple[float, float, float]] = {}
        self._candidate_percussive_spikes: dict[int, float] = {}
        self.candidate_times: np.ndarray = np.array([])
        self._pyin_seeded_keys: set[tuple[int, int]] = set()
        self._nnls_target_midis: list[int] = []
        self._nnls_midi_to_idx: dict[int, int] = {}
        self._nnls_activation: np.ndarray | None = None
        self._nnls_delta: np.ndarray | None = None
        self._nnls_dictionary: np.ndarray | None = None
        self._nnls_gain_threshold: float = 0.0
        self._nnls_onset_threshold: float = 0.0
        self.model = CorrectionModel()
        self._pitch_model: PitchModelPolicy = create_pitch_model("standard")
        self._pitch_models: dict[str, PitchModelPolicy] = {
            self._pitch_model.name: self._pitch_model,
        }
        if bootstrap_from_profiles:
            self._bootstrap_candidate_model_from_profiles()
        self._current_min_dur_sec: float = 0.0
        self._current_min_dur_label: str = self.DEFAULT_DURATION
        self._onset_sensitivity_gain: float = 1.0
        self._chroma_sensitivity_gain: float = 1.0
        self._nnls_max_polyphony: int = 4
        self._onset_base_thresh = float(
            np.percentile(self._transient_env_np, 30)
        ) if len(self._transient_env_np) else 0.0
        self._chroma_base_thresh = float(CHROMA_SHIFT_THRESH)

        _emit(90, "Applying filters …")
        self.apply_min_duration(self.DEFAULT_DURATION)
        _emit(96, "Building polyphonic NNLS engine …")
        self._rebuild_polyphonic_engine()
        self._refresh_multi_atom_model()
        _emit(100, "Ready")

    # ── Candidate key helpers ─────────────────────────────────────────────

    def _cand_key(self, t: float) -> int:
        return int(round(float(t) * self._CAND_KEY_SCALE))

    def _cand_time(self, key: int) -> float:
        return float(key) / self._CAND_KEY_SCALE

    # ── Feature extraction ────────────────────────────────────────────────

    def _window_rms_db(self, t0, t1):
        s0 = max(0, int(t0 * self.sr))
        s1 = min(len(self.audio), int(t1 * self.sr))
        if s1 <= s0:
            return -80.0
        seg = self.audio[s0:s1]
        rms = float(np.sqrt(np.mean(seg * seg) + 1e-12))
        return float(20.0 * np.log10(max(rms, 1e-6)))

    def _window_abs_transient_intensity(self, t0, t1):
        times = self._transient_times_np
        if len(times) == 0:
            return 0.0
        mid = 0.5 * (t0 + t1)
        i0 = int(np.searchsorted(times, t0))
        im = int(np.searchsorted(times, mid))
        i1 = int(np.searchsorted(times, t1, side='right'))
        if im <= i0 or i1 <= im:
            return 0.0
        pre = self._transient_env_np[i0:im]
        post = self._transient_env_np[im:i1]
        if len(pre) == 0 or len(post) == 0:
            return 0.0
        return float(max(0.0, float(np.max(post)) - float(np.median(pre))))

    def _chroma_delta_at_grid(self, g, grid):
        ct = self._chroma_times
        half = 0.5 * grid
        b0 = int(np.searchsorted(ct, g - half))
        b1 = int(np.searchsorted(ct, g))
        a0 = b1
        a1 = int(np.searchsorted(ct, g + half))
        if b1 <= b0 or a1 <= a0:
            return 0.0
        b = self._chroma_full[:, b0:b1].mean(axis=1)
        a = self._chroma_full[:, a0:a1].mean(axis=1)
        bin_diffs = np.abs(a - b)
        top_k = min(2, len(bin_diffs))
        largest = np.partition(bin_diffs, -top_k)[-top_k:]
        return float(np.sum(largest))

    def _percussive_spike_at_grid(self, g, grid):
        """Calculate normalized percussive spike (onset sharpness).
        
        Returns a value in [0, 1] where:
        - 0: energy distributed equally across center and wings (flat envelope)
        - 1: all energy concentrated at center (sharp percussive onset)
        
        Normalized to be independent of recording loudness/gain.
        Prioritizes low "needle" energy (C0-C3) while retaining a small full-band cue.
        """
        ct = self.cqt_times
        quarter = 0.25 * grid
        half = 0.5 * grid
        o0 = int(np.searchsorted(ct, g - half))
        o1 = int(np.searchsorted(ct, g - quarter))
        i0 = o1
        i1 = int(np.searchsorted(ct, g + quarter))
        o2 = i1
        o3 = int(np.searchsorted(ct, g + half))

        # Helper: normalized peak-vs-wings score in [0, 1].
        # 0 means center energy is not stronger than both wings combined.
        # 1 means nearly all band energy is concentrated in the center.
        def _peak_score(b0: int, b1: int) -> float:
            if b1 <= b0:
                return 0.0
            inner = float(np.sum(self.cqt_mag[b0:b1, i0:i1]))
            outer_l = float(np.sum(self.cqt_mag[b0:b1, o0:o1])) if o1 > o0 else 0.0
            outer_r = float(np.sum(self.cqt_mag[b0:b1, o2:o3])) if o3 > o2 else 0.0
            outer = outer_l + outer_r
            total = inner + outer
            if total <= 1e-12:
                return 0.0
            return float(max(0.0, (inner - outer) / total))

        # Low "needle" band emphasized by user observation: C0..C3.
        lo_midi = max(int(self.cqt_fmin_midi), 12)
        hi_midi = min(int(self.cqt_fmin_midi + self.cqt_n_bins - 1), 48)
        lo_bin = max(0, lo_midi - int(self.cqt_fmin_midi))
        hi_bin = min(self.cqt_mag.shape[0], hi_midi - int(self.cqt_fmin_midi) + 1)

        low_score = _peak_score(lo_bin, hi_bin)
        full_score = _peak_score(0, self.cqt_mag.shape[0])

        # Low-band score carries most weight; full-band adds robustness.
        return float(0.80 * low_score + 0.20 * full_score)

    def transient_envelope_slice(self, t0, t1):
        times = self._transient_times_np
        if len(times) == 0:
            return np.array([]), np.array([])
        i0 = int(np.searchsorted(times, t0))
        i1 = int(np.searchsorted(times, t1, side='right'))
        if i1 <= i0:
            return np.array([]), np.array([])
        return times[i0:i1], self._transient_env_np[i0:i1]

    def visible_onset_markers(self, t_start: float, t_end: float) -> list[tuple[float, str]]:
        """Return visible active onset markers as (time, source).

        source is "manual" for user-forced onsets and "guess" for auto-detected onsets.
        """
        markers: list[tuple[float, str]] = []
        for key in sorted(self._candidate_active_keys):
            t = self._cand_time(key)
            if t < t_start or t > t_end:
                continue
            source = "manual" if key in self._candidate_forced_on_keys else "guess"
            markers.append((t, source))
        return markers

    # ── Sensitivity / duration filtering ──────────────────────────────────

    def set_sensitivity_gains(self, onset_gain, chroma_gain):
        self._onset_sensitivity_gain = max(0.05, min(float(onset_gain), 20.0))
        self._chroma_sensitivity_gain = max(0.05, min(float(chroma_gain), 20.0))
        self.apply_min_duration(self._current_min_dur_label)

    def set_polyphony(self, max_notes: int) -> None:
        """Set the maximum number of simultaneous MIDI notes returned per block."""
        value = max(1, min(int(max_notes), 10))
        if value != self._nnls_max_polyphony:
            self._nnls_max_polyphony = value
            self._invalidate_note_spans()

    def set_pitch_model(self, name: str) -> None:
        """Select a pitch policy while preserving compatible learned policy state."""
        if self._pitch_model.name == str(name):
            return
        requested = str(name)
        self._pitch_model = self._pitch_models.get(requested) or create_pitch_model(requested)
        self._pitch_models[self._pitch_model.name] = self._pitch_model
        if self._pitch_model.max_polyphony is not None:
            self.set_polyphony(self._pitch_model.max_polyphony)
        if hasattr(self, "_multi_atom_model"):
            self._refresh_multi_atom_model()
        self._pitch_cache.clear()
        self._invalidate_note_spans()

    @property
    def pitch_model_name(self) -> str:
        return self._pitch_model.name

    @property
    def pitch_model_summary(self) -> str:
        offset = getattr(self._pitch_model, "octave_offset", None)
        return self._pitch_model.label if offset is None else f"{self._pitch_model.label} ({offset:+d})"

    def apply_min_duration(self, label: str):
        fraction = self.NOTE_DURATION_OPTIONS[label]
        min_dur = self.grid.beat_dur * fraction
        self._current_min_dur_label = label
        onset_thresh = self._onset_base_thresh / max(self._onset_sensitivity_gain, 1e-8)
        chroma_thresh = self._chroma_base_thresh / max(self._chroma_sensitivity_gain, 1e-8)
        grid = min_dur
        if grid <= 0:
            self._candidate_raw_times = []
            self._candidate_features = {}
            self._candidate_active_keys = set()
            self.candidate_times = np.array([])
            self.onset_times = np.array([])
            self.chroma_times = np.array([])
            self._current_min_dur_sec = min_dur
            self._invalidate_note_spans()
            return
        if min_dur != self._current_min_dur_sec:
            self._current_min_dur_sec = min_dur
            self._extract_candidate_features(grid)
        self._activate_candidates(onset_thresh, chroma_thresh)

    def current_grid_duration(self) -> float:
        """Return the active note-duration grid in seconds."""
        return float(self._current_min_dur_sec)

    def _extract_candidate_features(self, grid):
        cand_times = np.arange(grid, self.duration, grid)
        self._candidate_raw_times = [float(t) for t in cand_times]
        self._candidate_features = {}
        self._candidate_percussive_spikes = {}
        for g in self._candidate_raw_times:
            t0 = max(0.0, g - 0.5 * grid)
            t1 = min(self.duration, g + 0.5 * grid)
            transient = self._window_abs_transient_intensity(t0, t1)
            chroma_delta = self._chroma_delta_at_grid(g, grid)
            percussive_spike = self._percussive_spike_at_grid(g, grid)
            energy_db = self._window_rms_db(t0, t1)
            key = self._cand_key(g)
            self._candidate_features[key] = (transient, chroma_delta, energy_db)
            self._candidate_percussive_spikes[key] = percussive_spike

    def _activate_candidates(self, onset_thresh, chroma_thresh):
        active_keys: set[int] = set()
        feat_list = list(self._candidate_features.items())
        if feat_list:
            transients = np.array([v[0] for _, v in feat_list], dtype=float)
            for i, (key, (transient, chroma_delta, energy_db)) in enumerate(feat_list):
                if energy_db <= -42.0:
                    continue
                lo = max(0, i - 3)
                hi = min(len(transients), i + 4)
                local = transients[lo:hi]
                local_max = float(np.max(local))
                local_med = float(np.median(local))
                peak_ratio = transient / max(local_max, 1e-6)
                prominence = transient - local_med
                onset_hit = (transient >= onset_thresh * 1.25 and
                             peak_ratio >= 0.74 and prominence >= 0.035)
                chroma_hit = (chroma_delta >= max(chroma_thresh * 2.4, 0.020) and
                              transient >= onset_thresh * 0.80 and peak_ratio >= 0.58)
                if onset_hit or chroma_hit:
                    active_keys.add(key)
        valid_keys = set(self._candidate_features.keys())
        active_keys |= (self._candidate_forced_on_keys & valid_keys)
        active_keys -= (self._candidate_forced_off_keys & valid_keys)
        self._candidate_forced_on_keys &= valid_keys
        self._candidate_forced_off_keys &= valid_keys
        self._candidate_active_keys = active_keys
        self._update_active_times()

    def _update_active_times(self):
        active_times = np.array(sorted(self._cand_time(k) for k in self._candidate_active_keys))
        self.candidate_times = active_times
        self.onset_times = active_times
        self.chroma_times = np.array([])
        self._pitch_cache.clear()
        self._pyin_seeded_keys.clear()
        self._invalidate_note_spans()

    # ── Beat Grid Quantizer ──────────────────────────────────────────────

    def get_beat_slice_duration(self, bpm: float, beats_per_bar: int,
                                duration_str: str) -> float:
        """Calculate beat slice duration in seconds.
        
        Args:
            bpm: beats per minute
            beats_per_bar: numerator of time signature (e.g., 4 for 4/4)
            duration_str: duration selection ("1/2", "1/8", "1/16")
        
        Returns:
            duration of each beat slice in seconds
        """
        fraction = self.NOTE_DURATION_OPTIONS.get(duration_str, 0.25)
        beat_dur = 60.0 / float(bpm)  # Duration of one beat (quarter note)
        return beat_dur * fraction

    def time_to_beat_index(self, time_sec: float, bpm: float, beats_per_bar: int,
                          duration_str: str) -> int:
        """Convert a time in seconds to the beat slice index."""
        slice_dur = self.get_beat_slice_duration(bpm, beats_per_bar, duration_str)
        if slice_dur <= 0:
            return 0
        return int(time_sec / slice_dur)

    def beat_index_to_time(self, beat_index: int, bpm: float, beats_per_bar: int,
                          duration_str: str) -> float:
        """Convert beat index to time in seconds."""
        slice_dur = self.get_beat_slice_duration(bpm, beats_per_bar, duration_str)
        return float(beat_index) * slice_dur

    def extract_cqt_slice_for_beat(self, beat_index: int, bpm: float,
                                   beats_per_bar: int, duration_str: str) -> np.ndarray:
        """Extract CQT column(s) for a given beat slice.
        
        Returns:
            1D CQT vector (all frequencies, single frame or averaged over slice)
        """
        slice_dur = self.get_beat_slice_duration(bpm, beats_per_bar, duration_str)
        t_start = float(beat_index) * slice_dur
        t_end = t_start + slice_dur
        
        # Find CQT frames within this beat slice
        frame_indices = np.where((self.cqt_times >= t_start) & (self.cqt_times < t_end))[0]
        
        if len(frame_indices) == 0:
            # Narrow slices can miss frame boundaries; fall back to nearest frame.
            if len(self.cqt_times) == 0:
                return np.zeros(self.cqt_n_bins, dtype=np.float32)
            center_t = 0.5 * (t_start + t_end)
            nearest = int(np.argmin(np.abs(self.cqt_times - center_t)))
            nearest = max(0, min(nearest, self.cqt_db.shape[1] - 1))
            return self.cqt_db[:, nearest].astype(np.float32)
        elif len(frame_indices) == 1:
            # Single frame: return as-is
            return self.cqt_db[:, frame_indices[0]].astype(np.float32)
        else:
            # Multiple frames: average across the slice
            return np.mean(self.cqt_db[:, frame_indices], axis=1).astype(np.float32)

    def extract_stft_slice_for_beat(self, beat_index: int, bpm: float,
                                    beats_per_bar: int, duration_str: str) -> np.ndarray:
        """Extract STFT column(s) for a given beat slice."""
        slice_dur = self.get_beat_slice_duration(bpm, beats_per_bar, duration_str)
        t_start = float(beat_index) * slice_dur
        t_end = t_start + slice_dur

        frame_indices = np.where((self.stft_times >= t_start) & (self.stft_times < t_end))[0]
        if len(frame_indices) == 0:
            if len(self.stft_times) == 0:
                return np.zeros(self.stft_mag.shape[0], dtype=np.float32)
            center_t = 0.5 * (t_start + t_end)
            nearest = int(np.argmin(np.abs(self.stft_times - center_t)))
            nearest = max(0, min(nearest, self.stft_mag.shape[1] - 1))
            return self.stft_mag[:, nearest].astype(np.float32)
        if len(frame_indices) == 1:
            return self.stft_mag[:, frame_indices[0]].astype(np.float32)
        return np.mean(self.stft_mag[:, frame_indices], axis=1).astype(np.float32)

    # ── Drum NNLS Solver ─────────────────────────────────────────────────

    def solve_drum_nnls_for_slice(self, cqt_slice_vector: np.ndarray,
                                  drum_patterns: list[tuple[np.ndarray, int]],
                                  threshold: float = 0.50) -> list[tuple[int, float]]:
        """Solve NNLS for drum patterns in a CQT slice.
        
        Args:
            cqt_slice_vector: 1D CQT vector for this beat slice
            drum_patterns: list of (fingerprint, drum_midi) tuples
            threshold: minimum coefficient to include a drum
        
        Returns:
            list of (drum_midi, confidence) tuples for drums above threshold
        """
        if not drum_patterns or len(cqt_slice_vector) == 0:
            return []
        
        # Build dictionary matrix from drum patterns
        # Shape: (n_frequencies, n_drums)
        dictionary = np.column_stack([fp for fp, _ in drum_patterns]).astype(np.float32)
        target = cqt_slice_vector.astype(np.float32)
        
        # Normalize target
        target_norm = float(np.linalg.norm(target))
        if target_norm < 1e-8:
            return []
        target = target / target_norm
        
        try:
            # Solve: minimize ||target - dictionary @ x||^2, subject to x >= 0
            coeffs, _ = optimize.nnls(dictionary, target)
            coeffs = np.maximum(coeffs, 0.0)  # Ensure non-negative
        except Exception:
            return []
        
        # Extract drums above threshold
        results = []
        for (_, drum_midi), coeff in zip(drum_patterns, coeffs):
            if coeff >= threshold:
                results.append((int(drum_midi), float(coeff)))
        
        return sorted(results, key=lambda x: x[1], reverse=True)

    # ── Candidate editing ─────────────────────────────────────────────────

    def _nearest_candidate_key(self, t, tol_sec):
        if not self._candidate_raw_times:
            return None
        arr = np.array(self._candidate_raw_times, dtype=float)
        idx = int(np.argmin(np.abs(arr - t)))
        if abs(arr[idx] - t) > tol_sec:
            return None
        return self._cand_key(float(arr[idx]))

    def remove_nearest_candidate(self, t, tol_sec=0.08):
        key = self._nearest_candidate_key(t, tol_sec)
        if key is None or key not in self._candidate_active_keys:
            return False
        feat = self._candidate_features.get(key)
        if feat is None:
            return False
        tr, ch, en = feat
        ps = self._candidate_percussive_spikes.get(key, 0.0)
        with self._onset_training_lock:
            self._pending_onset_training.append(
                (tr, ch, en, ps, False, key, "user")
            )
            self._onset_edit_generation += 1
        self._candidate_active_keys.discard(key)
        self._candidate_forced_off_keys.add(key)
        self._candidate_forced_on_keys.discard(key)
        self._update_active_times()
        return True

    def restore_nearest_candidate(self, t, tol_sec=0.08):
        key = self._nearest_candidate_key(t, tol_sec)
        if key is None:
            return False
        feat = self._candidate_features.get(key)
        if feat is None:
            return False

        # If the onset is already active as a guess, promote it to user-forced.
        if key in self._candidate_active_keys:
            if key in self._candidate_forced_on_keys:
                return False
            tr, ch, en = feat
            ps = self._candidate_percussive_spikes.get(key, 0.0)
            with self._onset_training_lock:
                self._pending_onset_training.append(
                    (tr, ch, en, ps, True, key, "user")
                )
                self._onset_edit_generation += 1
            self._candidate_forced_on_keys.add(key)
            self._candidate_forced_off_keys.discard(key)
            self._update_active_times()
            return True

        tr, ch, en = feat
        ps = self._candidate_percussive_spikes.get(key, 0.0)
        with self._onset_training_lock:
            self._pending_onset_training.append(
                (tr, ch, en, ps, True, key, "user")
            )
            self._onset_edit_generation += 1
        self._candidate_active_keys.add(key)
        self._candidate_forced_on_keys.add(key)
        self._candidate_forced_off_keys.discard(key)
        self._update_active_times()
        return True

    @property
    def onset_edit_generation(self) -> int:
        with self._onset_training_lock:
            return int(self._onset_edit_generation)

    def has_pending_onset_training(self) -> bool:
        with self._onset_training_lock:
            return bool(self._pending_onset_training)

    def process_pending_onset_training(self) -> int:
        """Drain queued onset labels and retrain only the candidate model."""
        with self._onset_training_lock:
            examples = self._pending_onset_training
            self._pending_onset_training = []
            generation = int(self._onset_edit_generation)
        if examples:
            self.model.record_candidate_batch(examples)
        return generation

    def candidate_keep_probs(self):
        out = {}
        for t in self._candidate_raw_times:
            key = self._cand_key(t)
            feat = self._candidate_features.get(key)
            if feat is None:
                out[float(t)] = None
                continue
            tr, ch, en = feat
            ps = self._candidate_percussive_spikes.get(key, 0.0)
            out[float(t)] = self.model.candidate_keep_prob(tr, ch, en, ps)
        return out

    # ── Polyphonic NNLS pitch engine ─────────────────────────────────────

    def _rebuild_polyphonic_engine(self):
        total_bins = int(self.cqt_mag.shape[0]) if self.cqt_mag.ndim == 2 else 0
        total_frames = int(self.cqt_mag.shape[1]) if self.cqt_mag.ndim == 2 else 0
        if total_bins <= 0 or total_frames <= 0:
            self._nnls_target_midis = []
            self._nnls_midi_to_idx = {}
            self._nnls_activation = None
            self._nnls_delta = None
            self._nnls_dictionary = None
            self._nnls_gain_threshold = 0.0
            self._nnls_onset_threshold = 0.0
            return

        # Register-aware calibration: one timbre fingerprint per ~octave so
        # the harmonic template reflects how the instrument actually sounds at
        # different pitches (bass vs. treble rolloff changes significantly).
        cal_midis, cal_weights = auto_calibrate_by_register(
            self.cqt_mag,
            n_points=7,
            bins_per_octave=12,
            fmin_midi=self.cqt_fmin_midi,
        )

        midi_hi = int(self.cqt_fmin_midi + self.cqt_n_bins - 1)
        self._nnls_target_midis = list(range(int(self.cqt_fmin_midi), midi_hi + 1))
        self._nnls_midi_to_idx = {midi: i for i, midi in enumerate(self._nnls_target_midis)}
        dictionary = build_register_aware_dictionary(
            cal_midis,
            cal_weights,
            target_pitches=self._nnls_target_midis,
            total_bins=total_bins,
            bins_per_octave=12,
        )
        self._nnls_dictionary = dictionary
        self._nnls_activation = solve_polyphonic_activations(self.cqt_mag, dictionary)

        if self._nnls_activation.size == 0:
            self._nnls_delta = np.zeros_like(self._nnls_activation)
            self._nnls_gain_threshold = 0.0
            self._nnls_onset_threshold = 0.0
            return

        self._nnls_delta = np.maximum(
            0.0,
            self._nnls_activation - np.column_stack([
                np.zeros((self._nnls_activation.shape[0], 1), dtype=float),
                self._nnls_activation[:, :-1],
            ]),
        )
        self._nnls_gain_threshold = float(np.percentile(self._nnls_activation, 90) * 0.45)
        pos_delta = self._nnls_delta[self._nnls_delta > 0.0]
        if pos_delta.size:
            self._nnls_onset_threshold = float(np.percentile(pos_delta, 85))
        else:
            self._nnls_onset_threshold = 0.0

    def _nnls_feature_vectors_for_block(self, t0, t1):
        if self._nnls_activation is None or not self._nnls_target_midis:
            return None
        f0 = int(np.searchsorted(self.cqt_times, t0))
        f1 = int(np.searchsorted(self.cqt_times, t1))
        f1 = min(f1, self._nnls_activation.shape[1])
        if f1 <= f0:
            return None

        # Attack window in frames: ~100 ms, to skip the transient.
        n_attack = max(1, int(round(0.10 * self.sr / HOP_SIZE)))
        # Flux window: 100 ms → 300 ms after block start captures new energy
        # after the attack has decayed, giving cleaner pitch evidence.
        n_flux_hi = max(n_attack + 1, int(round(0.30 * self.sr / HOP_SIZE)))
        f_sus_lo = min(f0 + n_attack, f1)
        f_flux_lo = f0 + n_attack
        f_flux_hi = min(f0 + n_flux_hi, f1)

        # Onset novelty window: compare per-bin post-onset mean against
        # pre-onset mean around the block boundary.
        n_onset_pre = max(1, int(round(0.12 * self.sr / HOP_SIZE)))
        n_onset_post = max(1, int(round(0.12 * self.sr / HOP_SIZE)))
        f_on_pre_lo = max(0, f0 - n_onset_pre)
        f_on_pre_hi = f0
        f_on_post_lo = f0
        f_on_post_hi = min(f0 + n_onset_post, f1)

        if self._nnls_dictionary is None:
            return None

        # Feature 1 observation: sustained CQT magnitude (skip first ~100 ms).
        src_lo = f_sus_lo if f_sus_lo < f1 else f0
        sustain_obs = self.cqt_mag[:, src_lo:f1].mean(axis=1)

        # Feature 1b observation: sustained spectral-peel energy.
        peel_lin = np.power(10.0, self.cqt_peel_db[:, src_lo:f1] / 20.0)
        peel_obs = peel_lin.mean(axis=1)

        # Feature 1c observation: sustained HCQT/pitch-salience energy.
        sal_lin = np.power(10.0, self.cqt_hcqt_db[:, src_lo:f1] / 20.0)
        salience_obs = sal_lin.mean(axis=1)

        # Feature 2 observation: post-attack spectral flux (100–300 ms).
        if f_flux_hi > f_flux_lo:
            flux_obs = compute_flux_frame(self.cqt_mag, f_flux_lo, f_flux_hi)
        else:
            flux_obs = np.zeros_like(sustain_obs)

        # Feature 2b observation: onset-centered pre/post novelty per bin.
        if f_on_pre_hi > f_on_pre_lo and f_on_post_hi > f_on_post_lo:
            pre_mean = self.cqt_mag[:, f_on_pre_lo:f_on_pre_hi].mean(axis=1)
            post_mean = self.cqt_mag[:, f_on_post_lo:f_on_post_hi].mean(axis=1)
            onset_obs = np.maximum(0.0, post_mean - pre_mean)
        else:
            onset_obs = np.zeros_like(sustain_obs)

        # Feature 3 observation: beatiness over the full block duration.
        # Per-bin AM-band beatiness: FFT-based detection of periodic wobble.
        # Immune to single dropouts, targets true beating artifacts.
        beat_obs = _compute_am_band_beatiness(self.cqt_mag, f0, f1, self.sr)

        # Normalize feature channels before stacking so one channel cannot
        # dominate purely due to scale.
        sustain_obs_n = sustain_obs / max(float(np.max(sustain_obs)), 1e-8)
        peel_obs_n = peel_obs / max(float(np.max(peel_obs)), 1e-8)
        salience_obs_n = salience_obs / max(float(np.max(salience_obs)), 1e-8)
        flux_obs_n = flux_obs / max(float(np.max(flux_obs)), 1e-8)
        onset_obs_n = onset_obs / max(float(np.max(onset_obs)), 1e-8)
        beat_obs_n = beat_obs / max(float(np.max(beat_obs)), 1e-8)

        # Solve each channel independently for debug/inspection.
        sustain_gain = solve_single_frame(self._nnls_dictionary, sustain_obs_n)
        peel_gain = solve_single_frame(self._nnls_dictionary, peel_obs_n)
        salience_gain = solve_single_frame(self._nnls_dictionary, salience_obs_n)
        flux_gain = solve_single_frame(self._nnls_dictionary, flux_obs_n)
        onset_gain = solve_single_frame(self._nnls_dictionary, onset_obs_n)
        beat_gain = solve_single_frame(self._nnls_dictionary, beat_obs_n)

        # Feed all positive feature channels into one solve by stacking.
        # min || [x1; x2; ...] - [W; W; ...] h ||^2
        stacked_W = np.vstack([
            self._nnls_dictionary,  # sustain CQT
            self._nnls_dictionary,  # spectral peel
            self._nnls_dictionary,  # pitch salience (HCQT)
            self._nnls_dictionary,  # post-attack flux
            self._nnls_dictionary,  # onset novelty pre/post
        ])
        stacked_obs = np.concatenate([
            sustain_obs_n,
            peel_obs_n,
            salience_obs_n,
            flux_obs_n,
            onset_obs_n,
        ])
        model_gain_raw = solve_single_frame(stacked_W, stacked_obs)

        # Negative flux influencer: suppress scores where beatiness is high.
        # This downweights oscillatory sidebands that often outrank true bass f0.
        beat_penalty_weight = 0.60
        model_gain = np.maximum(0.0, model_gain_raw - beat_penalty_weight * beat_gain)

        # Keep legacy names for downstream/debug formatting compatibility.
        combined = model_gain
        combined_abs = model_gain.copy()
        onset = onset_gain

        # Adaptive score gate from the model's own per-block distribution.
        model_gate = max(float(np.percentile(model_gain, 90)) * 0.70, 1e-5)
        onset_gate = max(float(np.percentile(onset_gain, 85)) * 0.65, 1e-5)
        active = (
            (model_gain >= model_gate) &
            (onset >= onset_gate)
        )

        return {
            "f0": f0,
            "f1": f1,
            "f_sus_lo": f_sus_lo,
            "f_flux_lo": f_flux_lo,
            "f_flux_hi": f_flux_hi,
            "f_on_pre_lo": f_on_pre_lo,
            "f_on_pre_hi": f_on_pre_hi,
            "f_on_post_lo": f_on_post_lo,
            "f_on_post_hi": f_on_post_hi,
            "sustain_gain": sustain_gain,
            "peel_gain": peel_gain,
            "salience_gain": salience_gain,
            "flux_gain": -beat_gain,
            "flux_novelty_gain": flux_gain,
            "onset_gain": onset_gain,
            "beat_gain": beat_gain,
            "model_gain": model_gain,
            "model_gate": model_gate,
            "onset_gate": onset_gate,
            "combined": combined,
            "combined_abs": combined_abs,
            "onset": onset,
            "active": active,
        }

    def debug_pitch_features_at(self, t: float, midi: int) -> dict | None:
        """Return detailed NNLS feature scores for the hovered block/note."""
        if midi not in self._nnls_midi_to_idx:
            return None
        boundaries = np.sort(np.unique(
            np.concatenate([[0.0], self.onset_times, self.chroma_times, [self.duration]])))
        t0 = t1 = None
        for i in range(len(boundaries) - 1):
            a, b = float(boundaries[i]), float(boundaries[i + 1])
            if a <= t < b:
                t0, t1 = a, b
                break
        if t0 is None or t1 is None:
            return None

        feats = self._nnls_feature_vectors_for_block(t0, t1)
        if feats is None:
            return None

        idx = self._nnls_midi_to_idx[int(midi)]
        combined_abs = feats["combined_abs"]
        order = np.argsort(combined_abs)[::-1]
        rank = int(np.where(order == idx)[0][0]) + 1
        top = [
            (int(self._nnls_target_midis[i]), float(combined_abs[i]))
            for i in order[:3]
        ]
        oct_up_midi = int(midi) + 12
        oct_up_score = None
        if oct_up_midi in self._nnls_midi_to_idx:
            oct_up_score = float(combined_abs[self._nnls_midi_to_idx[oct_up_midi]])

        return {
            "midi": int(midi),
            "t": float(t),
            "block_t0": float(t0),
            "block_t1": float(t1),
            "sustain": float(feats["sustain_gain"][idx]),
            "peel": float(feats["peel_gain"][idx]),
            "salience": float(feats["salience_gain"][idx]),
            "flux": float(feats["flux_gain"][idx]),
            "model": float(feats["model_gain"][idx]),
            "model_gate": float(feats["model_gate"]),
            "combined": float(feats["combined"][idx]),
            "combined_abs": float(feats["combined_abs"][idx]),
            "onset_delta": float(feats["onset"][idx]),
            "active": bool(feats["active"][idx]),
            "rank": rank,
            "top": top,
            "gain_threshold": float(self._nnls_gain_threshold),
            "onset_threshold": float(feats["onset_gate"]),
            "flux_window_ms": (
                float((feats["f_flux_lo"] - feats["f0"]) * HOP_SIZE / self.sr * 1000.0),
                float((feats["f_flux_hi"] - feats["f0"]) * HOP_SIZE / self.sr * 1000.0),
            ),
            "onset_window_ms": (
                float((feats["f_on_pre_lo"] - feats["f0"]) * HOP_SIZE / self.sr * 1000.0),
                float((feats["f_on_post_hi"] - feats["f0"]) * HOP_SIZE / self.sr * 1000.0),
            ),
            "beat_window_ms": (
                0.0,
                float((feats["f1"] - feats["f0"]) * HOP_SIZE / self.sr * 1000.0),
            ),
            "octave_up_midi": oct_up_midi,
            "octave_up_score": oct_up_score,
        }

    def _nnls_notes_for_block(self, t0, t1) -> list[tuple[int, float]]:
        feats = self._nnls_feature_vectors_for_block(t0, t1)
        if feats is None:
            return []
        combined = feats["combined"]
        combined_abs = feats["combined_abs"]
        active = feats["active"]
        idxs = np.where(active)[0]

        if idxs.size == 0:
            best = int(np.argmax(combined_abs))
            if float(combined_abs[best]) <= max(self._nnls_gain_threshold * 0.8, 1e-8):
                return []
            return [(int(self._nnls_target_midis[best]), float(combined_abs[best]))]

        # Keep notes ordered by combined score, capped by polyphony limit.
        ordered = idxs[np.argsort(combined[idxs])[::-1]]
        ordered = ordered[:self._nnls_max_polyphony]
        return [(int(self._nnls_target_midis[i]), float(combined_abs[i])) for i in ordered]

    # ── Pitch management ──────────────────────────────────────────────────

    def get_block_key_at(self, t):
        boundaries = np.sort(np.unique(
            np.concatenate([[0.0], self.onset_times, self.chroma_times, [self.duration]])))
        for i in range(len(boundaries) - 1):
            t0, t1 = boundaries[i], boundaries[i + 1]
            if t0 <= t < t1:
                return (int(t0 * self.sr), int(min(t1 * self.sr, len(self.audio))))
        return None

    def _train_pitch_model_for_manual_block(self, key: tuple[int, int]) -> None:
        with self._pitch_training_lock:
            manual = self._manual_pitches.get(key)
        if manual is None or len(manual) == 0:
            return
        t0 = float(key[0]) / float(self.sr)
        t1 = float(key[1]) / float(self.sr)
        self._multi_atom_model.observe(t0, t1, [int(m) for m in manual])
        self._multi_atom_model.calibrate(self._confirmed_pitch_examples())
        self.model.multi_atom_threshold_ratio = self._multi_atom_model.threshold_ratio

    def _confirmed_pitch_examples(self) -> list[tuple[float, float, frozenset[int]]]:
        with self._pitch_training_lock:
            items = list(self._manual_pitches.items())
        return [
            (
                float(key[0]) / float(self.sr),
                float(key[1]) / float(self.sr),
                frozenset(int(midi) for midi in manual),
            )
            for key, manual in items
            if manual
        ]

    def _refresh_multi_atom_model(self) -> None:
        self._multi_atom_model = MultiAtomNNLSModel(
            self,
            atom_bank=self.model.instrument_dictionary,
            threshold_ratio=self.model.multi_atom_threshold_ratio,
        )

    def _record_pitch_gate_example_with_nnls(self, key: tuple[int, int], midi: int,
                                              keep: bool, source: str = "user") -> None:
        """Record gate example using real NNLS features from the block."""
        example = self._pitch_gate_example_with_nnls(key, midi, keep, source)
        if example is not None:
            self.model.record_pitch_gate_batch([example])

    def _pitch_gate_example_with_nnls(self, key: tuple[int, int], midi: int,
                                      keep: bool, source: str = "user") -> tuple | None:
        """Build one pitch-gate training row from a block's NNLS features."""
        t0 = float(key[0]) / float(self.sr)
        t1 = float(key[1]) / float(self.sr)
        feats = self._nnls_feature_vectors_for_block(t0, t1)
        if feats is None:
            return None
        idx = self._nnls_midi_to_idx.get(int(midi))
        if idx is None:
            return None

        peel = float(feats["peel_gain"][idx])
        salience = float(feats["salience_gain"][idx])
        flux = float(feats["flux_gain"][idx])
        onset = float(feats["onset"][idx])
        note_energy = self._harmonic_block_energy(t0, t1, int(midi))
        energy_db = float(20.0 * np.log10(max(note_energy, 1e-8)))

        # Relative intensity in this block: weaker notes get more negative values.
        candidate_db = []
        for m in self._nnls_target_midis:
            e_m = self._harmonic_block_energy(t0, t1, int(m))
            candidate_db.append(float(20.0 * np.log10(max(e_m, 1e-8))))
        max_db = max(candidate_db) if candidate_db else energy_db
        energy_rel_db = float(energy_db - max_db)

        sample_key = f"{key[0]}:{key[1]}:{int(midi)}"
        return (
            peel, salience, flux, onset, energy_db, energy_rel_db,
            bool(keep), sample_key, source,
        )

    def add_manual_pitch(self, t, midi):
        key = self.get_block_key_at(t)
        if key is None:
            return False
        with self._pitch_training_lock:
            existing = self._manual_pitches.get(key)
            notes = list(existing) if existing else []
            if int(midi) not in notes:
                notes.append(int(midi))
            self._manual_pitches[key] = tuple(sorted(set(notes)))
            self._pending_pitch_training.append((key, int(midi), True))
            self._pitch_edit_generation += 1
            self._note_spans_revision += 1
            self._generated_note_spans_cache = None
        return True

    def remove_manual_pitch(self, t, midi, tol_semitones=1):
        current_key = self.get_block_key_at(t)
        if current_key is None:
            return False
        with self._pitch_training_lock:
            removed_midi = int(midi)
            removed = False
            rejection_keys: set[tuple[int, int]] = set()

            # Manual intervals remain authoritative even if onset edits have
            # since changed the current analysis block boundaries.
            for key, manual in list(self._manual_pitches.items()):
                if manual is None or not (key[0] <= int(t * self.sr) < key[1]):
                    continue
                notes = list(manual)
                idx = int(np.argmin(np.abs(np.array(notes) - int(midi))))
                if abs(notes[idx] - int(midi)) > tol_semitones:
                    continue
                removed_midi = int(notes.pop(idx))
                removed = True
                rejection_keys.add(key)
                self._manual_pitches[key] = (
                    tuple(sorted(set(notes))) if notes else None
                )

            # Release edits materialize a separate explicit span. Remove that
            # representation too, otherwise it overlays the rejected result.
            matching_explicit = [
                (note_id, span)
                for note_id, span in self._explicit_note_spans.items()
                if span.onset <= float(t) < span.release
                and abs(int(span.midi) - int(midi)) <= tol_semitones
            ]
            for note_id, span in matching_explicit:
                self._explicit_note_spans.pop(note_id, None)
                removed_midi = int(span.midi)
                removed = True
                if span.block_keys:
                    rejection_keys.update(span.block_keys)
                else:
                    rejection_keys.add((
                        int(span.onset * self.sr),
                        int(min(span.release * self.sr, len(self.audio))),
                    ))

            if not rejection_keys:
                rejection_keys.add(current_key)
            if removed and matching_explicit:
                for key in rejection_keys:
                    existing = self._manual_pitches.get(key)
                    if existing is None:
                        self._manual_pitches[key] = None

            self._pending_pitch_training.extend(
                (key, removed_midi, False) for key in sorted(rejection_keys)
            )
            self._pitch_edit_generation += 1
            self._note_spans_revision += 1
            self._generated_note_spans_cache = None
        return removed

    def mark_as_rest(self, t):
        key = self.get_block_key_at(t)
        if key is None:
            return False
        with self._pitch_training_lock:
            self._manual_pitches[key] = None
            self._note_spans_revision += 1
            self._generated_note_spans_cache = None
        return True

    @property
    def pitch_edit_generation(self) -> int:
        with self._pitch_training_lock:
            return int(self._pitch_edit_generation)

    def has_pending_pitch_training(self) -> bool:
        with self._pitch_training_lock:
            return bool(self._pending_pitch_training)

    def process_pending_pitch_training(self) -> int:
        """Drain queued pitch labels and retrain once; intended for PitchWorker."""
        with self._pitch_training_lock:
            events = self._pending_pitch_training
            self._pending_pitch_training = []
            generation = int(self._pitch_edit_generation)

        if not events:
            return generation

        gate_examples = []
        positive_keys: set[tuple[int, int]] = set()
        for key, midi, keep in events:
            if keep:
                self._pitch_model.observe_confirmation(key, midi)
                positive_keys.add(key)
            else:
                self._pitch_model.observe_rejection(key, midi)
            example = self._pitch_gate_example_with_nnls(
                key, midi, keep=keep, source="user"
            )
            if example is not None:
                gate_examples.append(example)

        if gate_examples:
            self.model.record_pitch_gate_batch(gate_examples)
        for key in positive_keys:
            self._train_pitch_model_for_manual_block(key)
        self._invalidate_note_spans()
        return generation

    def get_block_current_pitch(self, t):
        key = self.get_block_key_at(t)
        if key is None:
            return None
        if key in self._manual_pitches:
            manual = self._manual_pitches[key]
            if manual is None or len(manual) == 0:
                return None
            return int(manual[0])
        boundaries = np.sort(np.unique(
            np.concatenate([[0.0], self.onset_times, self.chroma_times, [self.duration]])))
        for i in range(len(boundaries) - 1):
            t0, t1 = boundaries[i], boundaries[i + 1]
            if t0 <= t < t1:
                nnls_notes = self._nnls_notes_for_block(t0, t1)
                if nnls_notes:
                    return int(nnls_notes[0][0])
                break
        return None

    def _cqt_fingerprint_for_block(self, t0, t1):
        """Extract 5-harmonic fingerprint from CQT frame with max energy in time range.
        Returns normalized 5-element weight vector [1f, 2f, 3f, 4f, 5f]."""
        from core.polyphonic_nnls import define_harmonic_weights
        
        f0 = int(np.searchsorted(self.cqt_times, t0))
        f1 = int(np.searchsorted(self.cqt_times, t1))
        f1 = min(f1, self.cqt_db.shape[1])
        if f1 <= f0:
            return np.zeros(5, dtype=np.float32)
        
        # Find brightest frame in range (for calibration)
        energy_per_frame = np.sum(self.cqt_db[:, f0:f1], axis=0)
        if len(energy_per_frame) == 0:
            return np.zeros(5, dtype=np.float32)
        brightest_idx = int(np.argmax(energy_per_frame))
        best_frame = f0 + brightest_idx
        
        # Extract 5-harmonic profile from brightest frame; use median MIDI as register
        # For now use a generic calibration point (C3 = MIDI 48)
        harmonic_weights = define_harmonic_weights(
            self.cqt_db, best_frame, fundamental_pitch=48, bins_per_octave=12
        )
        return harmonic_weights.astype(np.float32)

    # ── Pitch queries ─────────────────────────────────────────────────────

    def get_cached_block_pitches(self, t_start, t_end):
        return [row[:5] for row in self.get_pitched_note_rows(t_start, t_end)]

    def get_pitched_note_rows(self, t_start: float, t_end: float) -> list[tuple]:
        spans = self.get_pitched_note_spans(t_start, t_end)
        velocities = self._velocities_from_energies([span.energy for span in spans])
        return [
            (span.onset, span.release, span.midi, span.source, vel, span.note_id)
            for span, vel in zip(spans, velocities)
        ]

    def get_pitched_note_spans(self, t_start: float, t_end: float) -> list[PitchedNoteSpan]:
        while True:
            with self._pitch_training_lock:
                cached = getattr(self, "_generated_note_spans_cache", None)
                revision = int(getattr(self, "_note_spans_revision", 0))
            if cached is not None:
                break
            computed = tuple(self._assemble_note_spans(
                self._collect_block_note_rows(0.0, float(self.duration))
            ))
            with self._pitch_training_lock:
                if revision == int(getattr(self, "_note_spans_revision", 0)):
                    self._generated_note_spans_cache = computed
                    cached = computed
                    break
        spans = list(cached)
        spans = self._overlay_explicit_note_spans(spans)
        return [
            span for span in spans
            if span.release > float(t_start) and span.onset < float(t_end)
        ]

    def _invalidate_note_spans(self) -> None:
        lock = getattr(self, "_pitch_training_lock", None)
        if lock is None:
            self._note_spans_revision = int(
                getattr(self, "_note_spans_revision", 0)
            ) + 1
            self._generated_note_spans_cache = None
            return
        with lock:
            self._note_spans_revision = int(
                getattr(self, "_note_spans_revision", 0)
            ) + 1
            self._generated_note_spans_cache = None

    def _analysis_boundaries(self) -> np.ndarray:
        return np.sort(np.unique(np.concatenate([
            [0.0], self.onset_times, self.chroma_times, [self.duration]
        ])))

    def _assemble_note_spans(self, rows: list[tuple]) -> list[PitchedNoteSpan]:
        active: dict[int, PitchedNoteSpan] = {}
        completed: list[PitchedNoteSpan] = []

        for t0, t1, midi, source, energy in sorted(rows, key=lambda row: (row[0], row[2])):
            midi = int(midi)
            block_key = (
                int(float(t0) * self.sr),
                int(min(float(t1) * self.sr, len(self.audio))),
            )
            previous = active.get(midi)
            touches_previous = (
                previous is not None
                and abs(previous.release - float(t0)) <= (1.0 / max(float(self.sr), 1.0))
            )
            if touches_previous:
                previous_duration = max(previous.release - previous.onset, 0.0)
                row_duration = max(float(t1) - float(t0), 0.0)
                total_duration = previous_duration + row_duration
                combined_energy = (
                    (previous.energy * previous_duration + float(energy) * row_duration)
                    / total_duration
                    if total_duration > 0.0 else float(energy)
                )
                active[midi] = replace(
                    previous,
                    release=float(t1),
                    source="manual" if "manual" in {previous.source, source} else previous.source,
                    energy=combined_energy,
                    block_keys=previous.block_keys + (block_key,),
                )
                continue

            if previous is not None:
                completed.append(previous)
            note_id = f"generated:{midi}:{block_key[0]}"
            active[midi] = PitchedNoteSpan(
                note_id=note_id,
                onset=float(t0),
                release=float(t1),
                midi=midi,
                source=str(source),
                energy=float(energy),
                block_keys=(block_key,),
            )

        completed.extend(active.values())
        return sorted(completed, key=lambda span: (span.onset, span.midi, span.release))

    def _overlay_explicit_note_spans(
        self, generated: list[PitchedNoteSpan]
    ) -> list[PitchedNoteSpan]:
        with self._pitch_training_lock:
            explicit = sorted(
                self._explicit_note_spans.values(),
                key=lambda span: (span.onset, span.midi, span.release),
            )
        if not explicit:
            return generated

        output: list[PitchedNoteSpan] = list(explicit)
        for span in generated:
            fragments = [span]
            for manual in explicit:
                if manual.midi != span.midi:
                    continue
                if set(manual.block_keys) & set(span.block_keys):
                    fragments = []
                    break
                next_fragments: list[PitchedNoteSpan] = []
                for fragment in fragments:
                    if manual.release <= fragment.onset or manual.onset >= fragment.release:
                        next_fragments.append(fragment)
                        continue
                    if fragment.onset < manual.onset:
                        next_fragments.append(replace(
                            fragment,
                            note_id=f"{fragment.note_id}:before:{manual.note_id}",
                            release=manual.onset,
                        ))
                    if manual.release < fragment.release:
                        next_fragments.append(replace(
                            fragment,
                            note_id=f"{fragment.note_id}:after:{manual.note_id}",
                            onset=manual.release,
                        ))
                fragments = next_fragments
            output.extend(fragments)
        return sorted(output, key=lambda span: (span.onset, span.midi, span.release))

    def preview_note_release(self, note_id: str, requested_release: float) -> float | None:
        spans = self.get_pitched_note_spans(0.0, float(self.duration))
        span = next((item for item in spans if item.note_id == note_id), None)
        if span is None:
            return None

        boundaries = self._analysis_boundaries()
        if len(boundaries):
            release = float(boundaries[int(np.argmin(np.abs(boundaries - requested_release)))])
        else:
            release = float(requested_release)
        later_onsets = [
            item.onset for item in spans
            if item.midi == span.midi and item.note_id != span.note_id
            and item.onset > span.onset
        ]
        upper_bound = min(later_onsets) if later_onsets else float(self.duration)
        minimum_release = min(
            upper_bound,
            span.onset + max(1.0 / max(float(self.sr), 1.0), 1e-6),
        )
        return max(minimum_release, min(release, upper_bound, float(self.duration)))

    def set_note_release(self, note_id: str, requested_release: float) -> float | None:
        spans = self.get_pitched_note_spans(0.0, float(self.duration))
        span = next((item for item in spans if item.note_id == note_id), None)
        release = self.preview_note_release(note_id, requested_release)
        if span is None or release is None:
            return None

        if note_id.startswith("explicit:"):
            explicit_id = note_id
        else:
            explicit_id = f"explicit:{self._next_explicit_note_id}"
            self._next_explicit_note_id += 1
        with self._pitch_training_lock:
            if not note_id.startswith("explicit:"):
                self._pending_pitch_training.extend(
                    (key, span.midi, True) for key in span.block_keys
                )
                self._pitch_edit_generation += 1
            self._explicit_note_spans[explicit_id] = replace(
                span,
                note_id=explicit_id,
                release=release,
                source="manual",
            )
        return release

    def _add_explicit_note_span(
        self, onset: float, release: float, midi: int, energy: float | None = None
    ) -> PitchedNoteSpan:
        boundaries = self._analysis_boundaries()
        block_keys = []
        for index in range(len(boundaries) - 1):
            t0 = float(boundaries[index])
            t1 = float(boundaries[index + 1])
            if t1 <= onset or t0 >= release:
                continue
            block_keys.append((
                int(t0 * self.sr),
                int(min(t1 * self.sr, len(self.audio))),
            ))

        note_id = f"explicit:{self._next_explicit_note_id}"
        self._next_explicit_note_id += 1
        span = PitchedNoteSpan(
            note_id=note_id,
            onset=float(onset),
            release=float(release),
            midi=int(midi),
            source="manual",
            energy=(
                float(energy) if energy is not None
                else self._harmonic_block_energy(float(onset), float(release), int(midi))
            ),
            block_keys=tuple(block_keys),
        )
        with self._pitch_training_lock:
            self._explicit_note_spans[note_id] = span
        return span

    def get_cached_drum_hits(self, t_start: float, t_end: float,
                             threshold: float = 0.50) -> list[tuple[float, float, int, str, int]]:
        """Return drum hits for the visible window using the selected beat grid.

        Rows match pitch shape: (t0, t1, midi, source, velocity).
        source is "manual" for user-confirmed hits and "ml" for NNLS predictions.
        """
        bpm = float(self.grid.bpm)
        beats_per_bar = int(self.grid.beats_per_bar)
        duration_str = self._current_min_dur_label
        slice_dur = self.get_beat_slice_duration(bpm, beats_per_bar, duration_str)
        if slice_dur <= 0.0:
            return []

        first = max(0, int(np.floor(float(t_start) / slice_dur)))
        last = int(np.ceil(float(t_end) / slice_dur))
        drum_patterns = self.model.get_drum_patterns(expected_size=self.stft_mag.shape[0])

        rows: list[tuple[float, float, int, str, float]] = []
        for beat_index in range(first, last + 1):
            bt0 = self.beat_index_to_time(beat_index, bpm, beats_per_bar, duration_str)
            bt1 = bt0 + slice_dur
            if bt1 <= t_start or bt0 >= t_end:
                continue

            manual_midis = sorted(self._manual_drums.get(beat_index, set()))
            rejected_midis = self._manual_drum_rejects.get(beat_index, set())

            for midi in manual_midis:
                e = self._drum_bin_energy(bt0, bt1, int(midi))
                rows.append((float(max(bt0, t_start)), float(min(bt1, t_end)), int(midi), "manual", float(e)))

            if not drum_patterns:
                continue

            spec_slice = self.extract_stft_slice_for_beat(beat_index, bpm, beats_per_bar, duration_str)
            for midi, conf in self.solve_drum_nnls_for_slice(spec_slice, drum_patterns, threshold=threshold):
                if int(midi) in manual_midis or int(midi) in rejected_midis:
                    continue
                
                # Score the drum candidate using the trained acceptance model
                # This filters out drums that have been rejected by the user
                features = self._extract_drum_slice_features(beat_index, bpm, beats_per_bar, duration_str)
                if features is not None:
                    _, (attack_slope, impulsivity, decay_rate, energy_db, energy_rel_db, 
                        band_low, band_mid, band_high, _) = features
                    accept_prob = self.model.get_drum_acceptance_prob(
                        attack_slope, impulsivity, decay_rate, energy_db, energy_rel_db,
                        band_low, band_mid, band_high
                    )
                    # Only include if model is confident enough (default 0.5 if not trained yet)
                    if accept_prob is not None and accept_prob < 0.5:
                        continue
                
                rows.append((float(max(bt0, t_start)), float(min(bt1, t_end)), int(midi), "ml", float(conf)))

        velocities = self._velocities_from_energies([r[4] for r in rows])
        return [
            (t0, t1, midi, source, vel)
            for (t0, t1, midi, source, _), vel in zip(rows, velocities)
        ]

    def compute_visible_block_pitches(self, t_start, t_end):
        return self.get_cached_block_pitches(t_start, t_end)

    # ── MIDI export ───────────────────────────────────────────────────────

    def build_midi_notes(self, run_pyin_for_missing=False):
        spans = self.get_pitched_note_spans(0.0, float(self.duration))
        velocities = self._velocities_from_energies([span.energy for span in spans])
        return [
            (span.onset, span.release, span.midi, velocity)
            for span, velocity in zip(spans, velocities)
        ]

    def _train_note_boundaries_from_midi(
        self, note_rows: list[tuple[float, float, int, int, int]]
    ) -> tuple[int, int]:
        """Label grid candidates from known MIDI note boundaries and silence."""
        if not note_rows or not self._candidate_raw_times:
            return 0, 0

        tolerance = 0.5 * max(float(self._current_min_dur_sec), 0.0) + 1e-9
        positive_keys: set[int] = set()
        for row in note_rows:
            for boundary in (float(row[0]), float(row[1])):
                key = self._nearest_candidate_key(boundary, tolerance)
                if key is not None:
                    positive_keys.add(key)

        intervals = [(float(row[0]), float(row[1])) for row in note_rows]
        negative_keys = {
            self._cand_key(t)
            for t in self._candidate_raw_times
            if self._cand_key(t) not in positive_keys
            and not any(t0 < float(t) < t1 for t0, t1 in intervals)
        }

        examples = []
        for key in sorted(positive_keys | negative_keys):
            feat = self._candidate_features.get(key)
            if feat is None:
                continue
            transient, chroma_delta, energy_db = feat
            examples.append((
                transient, chroma_delta, energy_db,
                self._candidate_percussive_spikes.get(key, 0.0),
                key in positive_keys, key, "user",
            ))
        if examples:
            self.model.record_candidate_batch(examples)

        valid_keys = set(self._candidate_features)
        positive_keys &= valid_keys
        negative_keys &= valid_keys
        self._candidate_forced_on_keys |= positive_keys
        self._candidate_forced_on_keys -= negative_keys
        self._candidate_forced_off_keys |= negative_keys
        self._candidate_forced_off_keys -= positive_keys
        self._candidate_active_keys |= positive_keys
        self._candidate_active_keys -= negative_keys
        self._update_active_times()
        return len(positive_keys), len(negative_keys)

    def load_user_notes_from_midi(self, midi_path: str,
                                  prefer_percussion: bool = False) -> tuple[int, int, int, int]:
        """Load notes from a MIDI file as user-entered manual data.

        Args:
            midi_path: MIDI file path.
            prefer_percussion: If True, import all notes through drum lanes.

        Returns:
            (added_pitched_notes, added_drum_hits, boundary_labels, negative_labels)
        """
        from core.midi_export import load_midi_notes

        parsed = load_midi_notes(midi_path)
        if not parsed:
            return 0, 0, 0, 0

        clipped: list[tuple[float, float, int, int, int]] = []
        for t0, t1, midi, velocity, channel in parsed:
            if float(t1) <= 0.0 or float(t0) >= float(self.duration):
                continue
            tt0 = max(0.0, float(t0))
            tt1 = min(float(self.duration), float(t1))
            if tt1 <= tt0:
                tt1 = min(float(self.duration), tt0 + 1e-3)
            if tt1 > tt0:
                clipped.append((tt0, tt1, int(midi), int(velocity), int(channel)))

        boundary_labels, silent_candidate_labels = self._train_note_boundaries_from_midi(clipped)

        bpm = float(self.grid.bpm)
        beats_per_bar = int(self.grid.beats_per_bar)
        duration_str = self._current_min_dur_label

        added_pitches = 0
        added_drums = 0
        pitch_gate_examples: list[tuple] = []
        touched_pitch_keys: set[tuple[int, int]] = set()
        seen_pitches: set[tuple[int, int, int]] = set()
        seen_drums: set[tuple[int, int]] = set()

        for tt0, tt1, midi, _vel, channel in clipped:
            import_as_drum = bool(prefer_percussion) or int(channel) == 9
            if import_as_drum:
                t_mid = 0.5 * (tt0 + tt1)
                beat_index = self.time_to_beat_index(
                    t_mid, bpm, beats_per_bar, duration_str)
                drum_key = (int(beat_index), int(midi))
                if drum_key in seen_drums:
                    continue
                if self.add_manual_drum(
                    beat_index, int(midi), bpm, beats_per_bar, duration_str
                ):
                    seen_drums.add(drum_key)
                    added_drums += 1
                continue

            pitch_key = (
                int(round(tt0 * self.sr)),
                int(round(tt1 * self.sr)),
                int(midi),
            )
            if pitch_key in seen_pitches:
                continue
            span = self._add_explicit_note_span(tt0, tt1, int(midi))
            for block_key in span.block_keys:
                self._pitch_model.observe_confirmation(block_key, int(midi))
                existing = self._manual_pitches.get(block_key)
                notes = list(existing) if existing else []
                notes.append(int(midi))
                self._manual_pitches[block_key] = tuple(sorted(set(notes)))
                touched_pitch_keys.add(block_key)
            seen_pitches.add(pitch_key)
            added_pitches += 1

        for key in touched_pitch_keys:
            self._train_pitch_model_for_manual_block(key)
            for midi in self._manual_pitches.get(key, ()) or ():
                example = self._pitch_gate_example_with_nnls(key, int(midi), True, "user")
                if example is not None:
                    pitch_gate_examples.append(example)

        pitched_intervals = [
            (t0, t1) for t0, t1, _midi, _vel, channel in clipped
            if not prefer_percussion and int(channel) != 9
        ]
        negative_pitch_labels = 0
        if pitched_intervals:
            boundaries = np.sort(np.unique(np.concatenate([
                [0.0], self.onset_times, self.chroma_times, [self.duration]
            ])))
            for idx in range(len(boundaries) - 1):
                t0, t1 = float(boundaries[idx]), float(boundaries[idx + 1])
                if any(note_t0 < t1 and note_t1 > t0 for note_t0, note_t1 in pitched_intervals):
                    continue
                key = (int(t0 * self.sr), int(min(t1 * self.sr, len(self.audio))))
                candidates = self._pitch_model.select_nnls_notes(
                    self._nnls_notes_for_block(t0, t1)
                )
                for midi, _score in candidates:
                    example = self._pitch_gate_example_with_nnls(key, int(midi), False, "user")
                    if example is not None:
                        pitch_gate_examples.append(example)
                        negative_pitch_labels += 1

        if pitch_gate_examples:
            self.model.record_pitch_gate_batch(pitch_gate_examples)

        negative_labels = silent_candidate_labels + negative_pitch_labels
        self._invalidate_note_spans()
        return added_pitches, added_drums, boundary_labels, negative_labels

    def accept_visible_recommendations(self, t_start: float, t_end: float,
                                       pitch_rows: list[tuple] | None = None,
                                       current_mode: str = "Pitched") -> tuple[int, int]:
        """Promote visible pitch and onset recommendations to user-confirmed data.
        
        Args:
            t_start, t_end: time window
            pitch_rows: pre-fetched rows (drums in Percussive, pitches in Pitched)
            current_mode: "Pitched" or "Percussive" to branch behavior
        """
        accepted_pitches = 0
        accepted_onsets = 0

        rows = pitch_rows if pitch_rows is not None else self._collect_note_rows(t_start, t_end)
        
        # Percussive mode: accept ML drum hits via drum APIs (beat-indexed)
        if current_mode == "Percussive":
            bpm = float(self.grid.bpm)
            beats_per_bar = int(self.grid.beats_per_bar)
            duration_str = self._current_min_dur_label
            
            for entry in rows:
                t0, t1, midi, source, _ = entry[:5]
                if midi is None or source != "ml":
                    continue
                beat_index = self.time_to_beat_index(0.5 * (t0 + t1), bpm, beats_per_bar, duration_str)
                if self.add_manual_drum(beat_index, int(midi), bpm, beats_per_bar, duration_str):
                    accepted_pitches += 1
            return accepted_pitches, accepted_onsets
        
        # Pitched mode: accept ML/NNLS pitches via pitch APIs (time-indexed)
        for entry in rows:
            t0, t1, midi, source, _ = entry[:5]
            if midi is None or source not in {"nnls", "ml"}:
                continue
            t_mid = 0.5 * (float(t0) + float(t1))
            if self.add_manual_pitch(t_mid, int(midi)):
                accepted_pitches += 1

        for t, source in self.visible_onset_markers(t_start, t_end):
            if source != "guess":
                continue
            if self.restore_nearest_candidate(float(t)):
                accepted_onsets += 1

        return accepted_pitches, accepted_onsets

    def _collect_note_rows(self, t_start, t_end):
        spans = self.get_pitched_note_spans(t_start, t_end)
        return [
            (span.onset, span.release, span.midi, span.source, span.energy)
            for span in spans
        ]

    def _collect_block_note_rows(self, t_start, t_end):
        boundaries = self._analysis_boundaries()
        with self._pitch_training_lock:
            manual_pitches = dict(self._manual_pitches)
        note_rows = []

        manual_intervals = []
        for key, manual in manual_pitches.items():
            manual_t0 = float(key[0]) / float(self.sr)
            manual_t1 = float(key[1]) / float(self.sr)
            if manual_t1 <= t_start or manual_t0 >= t_end:
                continue
            manual_intervals.append((manual_t0, manual_t1))
            if manual is None:
                continue
            grid_duration = float(getattr(self, "_current_min_dur_sec", 0.0))
            rendered_t1 = (
                min(manual_t1, manual_t0 + grid_duration)
                if grid_duration > 0.0 else manual_t1
            )
            for midi in manual:
                energy = self._harmonic_block_energy(
                    manual_t0, manual_t1, int(midi)
                )
                note_rows.append((
                    manual_t0, rendered_t1, int(midi), "manual", energy
                ))

        for i in range(len(boundaries) - 1):
            t0, t1 = boundaries[i], boundaries[i + 1]
            if t1 <= t_start or t0 >= t_end:
                continue
            if any(mt1 > t0 and mt0 < t1 for mt0, mt1 in manual_intervals):
                continue

            multi_atom_model = getattr(self, "_multi_atom_model", None)
            if multi_atom_model is not None and multi_atom_model.has_learned_atoms:
                suggested_notes = sorted(multi_atom_model.predict(t0, t1))
            else:
                fingerprint = self._cqt_fingerprint_for_block(t0, t1)
                suggested_notes, _ = self.model.pitch_suggestions(fingerprint)

            feats = self._nnls_feature_vectors_for_block(t0, t1)
            nnls_notes = self._nnls_notes_for_block(t0, t1)
            nnls_notes = self._pitch_model.select_nnls_notes(nnls_notes)
            if suggested_notes:
                ml_midis = [int(m) for m in suggested_notes[:self._nnls_max_polyphony]]
                ml_energy_db = {}
                for midi in ml_midis:
                    e_m = self._harmonic_block_energy(t0, t1, int(midi))
                    ml_energy_db[int(midi)] = float(20.0 * np.log10(max(e_m, 1e-8)))
                ml_max_db = max(ml_energy_db.values()) if ml_energy_db else -120.0
                learned_release = min(
                    float(t1), float(t0) + float(self._current_min_dur_sec)
                )

                for midi in ml_midis:
                    e = self._harmonic_block_energy(t0, t1, int(midi))
                    e_db = ml_energy_db[int(midi)]
                    e_rel_db = float(e_db - ml_max_db)

                    idx = self._nnls_midi_to_idx.get(int(midi))
                    if feats is not None and idx is not None:
                        gate_prob = self.model.pitch_gate_prob_with_features(
                            peel=float(feats["peel_gain"][idx]),
                            salience=float(feats["salience_gain"][idx]),
                            flux=float(feats["flux_gain"][idx]),
                            onset=float(feats["onset"][idx]),
                            energy_db=e_db,
                            energy_rel_db=e_rel_db,
                        )
                        if gate_prob is not None:
                            if e_rel_db <= -18.0:
                                intensity_factor = 0.60
                            elif e_rel_db >= -6.0:
                                intensity_factor = 1.00
                            else:
                                intensity_factor = 0.60 + (e_rel_db + 18.0) * (0.40 / 12.0)
                            final_prob = gate_prob * float(intensity_factor)
                            if final_prob < self.model.PITCH_GATE_THRESHOLD:
                                continue

                    note_rows.append((float(t0), learned_release, int(midi), "ml", e))
            elif nnls_notes:
                nnls_energy_db = {}
                for midi, _ in nnls_notes:
                    sounding_midi = self._pitch_model.sounding_midi(int(midi))
                    e = self._harmonic_block_energy(t0, t1, sounding_midi)
                    nnls_energy_db[int(midi)] = float(20.0 * np.log10(max(e, 1e-8)))
                nnls_max_db = max(nnls_energy_db.values()) if nnls_energy_db else -120.0

                for midi, _ in nnls_notes:
                    sounding_midi = self._pitch_model.sounding_midi(int(midi))
                    e = self._harmonic_block_energy(t0, t1, sounding_midi)
                    e_db = nnls_energy_db[int(midi)]
                    e_rel_db = float(e_db - nnls_max_db)
                    idx = self._nnls_midi_to_idx.get(int(midi))
                    if self._pitch_model.uses_pitch_gate and feats is not None and idx is not None:
                        gate_prob = self.model.pitch_gate_prob_with_features(
                            peel=float(feats["peel_gain"][idx]),
                            salience=float(feats["salience_gain"][idx]),
                            flux=float(feats["flux_gain"][idx]),
                            onset=float(feats["onset"][idx]),
                            energy_db=e_db,
                            energy_rel_db=e_rel_db,
                        )
                        if gate_prob is not None:
                            # Soft prior: progressively downweight very weak notes in block.
                            if e_rel_db <= -18.0:
                                intensity_factor = 0.60
                            elif e_rel_db >= -6.0:
                                intensity_factor = 1.00
                            else:
                                intensity_factor = 0.60 + (e_rel_db + 18.0) * (0.40 / 12.0)
                            final_prob = gate_prob * float(intensity_factor)
                            if final_prob < self.model.PITCH_GATE_THRESHOLD:
                                continue
                    note_rows.append((float(t0), float(t1), int(midi), "nnls", e))
        return note_rows

    def _harmonic_block_energy(self, t0, t1, midi):
        f0 = int(np.searchsorted(self.cqt_times, t0))
        f1 = int(np.searchsorted(self.cqt_times, t1))
        f1 = min(f1, self.cqt_db.shape[1])
        if f1 <= f0:
            return 0.0
        block_lin = np.power(10.0, self.cqt_db[:, f0:f1] / 20.0)
        mean_lin = block_lin.mean(axis=1)
        lo, hi = float(self.cqt_fmin_midi), float(self.cqt_fmin_midi + self.cqt_n_bins - 1)
        e_sum = w_sum = 0.0
        for n in range(1, self._VEL_HARMONICS + 1):
            hmidi = float(midi) + 12.0 * np.log2(float(n))
            if hmidi < lo or hmidi > hi:
                continue
            idx = hmidi - lo
            i0 = int(np.floor(idx))
            i1 = min(i0 + 1, self.cqt_n_bins - 1)
            frac = idx - i0
            p = (1.0 - frac) * float(mean_lin[i0]) + frac * float(mean_lin[i1])
            w = 1.0 / (n ** 0.7)
            e_sum += w * p
            w_sum += w
        return e_sum / w_sum if w_sum > 0.0 else 0.0

    def _velocities_from_energies(self, energies):
        if not energies:
            return []
        arr = np.array(energies, dtype=float)
        lo, hi = float(np.min(arr)), float(np.max(arr))
        if hi <= lo + 1e-12:
            return [90] * len(energies)
        scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        return [int(round(self._VEL_MIN + s * (self._VEL_MAX - self._VEL_MIN))) for s in scaled]

    def _drum_bin_energy(self, t0: float, t1: float, midi: int) -> float:
        """Energy proxy for a single drum MIDI bin over a time slice."""
        f0 = int(np.searchsorted(self.stft_times, t0))
        f1 = int(np.searchsorted(self.stft_times, t1))
        f1 = min(f1, self.stft_mag.shape[1])
        if f1 <= f0:
            return 0.0
        return float(np.mean(self.stft_mag[:, f0:f1]))

    # ── ML suggestions ────────────────────────────────────────────────────

    def apply_ml_suggestions_window(self, t_start, t_end,
                                     delete_threshold=0.30, pin_threshold=0.80):
        if not self.model._cand_trained:
            return 0, 0
        n_del = n_pin = 0
        for t in self._candidate_raw_times:
            if t < t_start or t > t_end:
                continue
            key = self._cand_key(t)
            # Skip candidates the user explicitly forced
            if key in self._candidate_forced_on_keys or key in self._candidate_forced_off_keys:
                continue
            feat = self._candidate_features.get(key)
            if feat is None:
                continue
            tr, ch, en = feat
            ps = self._candidate_percussive_spikes.get(key, 0.0)
            p = self.model.candidate_keep_prob(tr, ch, en, ps)
            if p is None:
                continue
            is_active = key in self._candidate_active_keys
            if p < delete_threshold and is_active:
                self._candidate_active_keys.discard(key)
                n_del += 1
            elif p > pin_threshold and not is_active:
                self._candidate_active_keys.add(key)
                n_pin += 1
        if n_del or n_pin:
            self._update_active_times()
        return n_del, n_pin

    def apply_ml_suggestions_global(self, delete_threshold=0.30, pin_threshold=0.80):
        """Apply the onset ML model to ALL candidates in the song."""
        if not self._candidate_raw_times:
            return 0, 0
        return self.apply_ml_suggestions_window(
            0.0, self.duration, delete_threshold, pin_threshold)

    # ── Model profiles ────────────────────────────────────────────────────

    def list_model_profiles(self) -> list[str]:
        d = pathlib.Path(self.MODEL_PROFILE_DIR)
        if not d.exists():
            return []
        names = [p.stem for p in d.glob("*.json") if p.is_file()]
        return sorted(names)

    def _bootstrap_candidate_model_from_profiles(self,
                                                 max_profiles: int = 8,
                                                 max_rows: int = 400) -> int:
        d = pathlib.Path(self.MODEL_PROFILE_DIR)
        if not d.exists():
            return 0

        files = [p for p in d.glob("*.json") if p.is_file()]
        if not files:
            return 0
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        merged_rows: list[dict] = []
        for path in files[:max_profiles]:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue

            for item in self.model.candidate_rows_from_payload(payload, user_only=True):
                merged_rows.append({
                    "row": list(item["row"]),
                    "label": int(item["label"]),
                    "key": None,
                    "source": "user",
                })

            if len(merged_rows) >= max_rows:
                break

        if not merged_rows:
            return 0

        merged_rows = merged_rows[:max_rows]
        self.model.load_dict({"schema": 7, "cand": merged_rows, "pitch": []}, merge=True)
        return len(merged_rows)

    def _profile_path(self, name):
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip()) or "default"
        d = pathlib.Path(self.MODEL_PROFILE_DIR)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{safe}.json"

    def save_model_profile(self, name):
        path = self._profile_path(name)
        payload = self.model.to_dict(user_only=True)
        payload["pitch_model"] = self._pitch_model.to_dict()
        payload["name"] = name
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def load_model_profile(self, name, merge=False):
        path = self._profile_path(name)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.model.load_dict(payload, merge=merge)
        policy_data = payload.get("pitch_model")
        if isinstance(policy_data, dict):
            self._pitch_model = create_pitch_model(
                str(policy_data.get("name", "standard")), policy_data
            )
            self._pitch_models[self._pitch_model.name] = self._pitch_model
            if self._pitch_model.max_polyphony is not None:
                self.set_polyphony(self._pitch_model.max_polyphony)
        elif not merge:
            self.set_pitch_model("standard")
        self._refresh_multi_atom_model()
        self._invalidate_note_spans()
        return path

    # ── Key detection ─────────────────────────────────────────────────────

    def _detect_key_from_piano_stem(self):
        try:
            input_dir = os.path.dirname(self.wav_path)
            for pattern in ["*Piano*.wav", "*piano*.wav", "*PIANO*.wav"]:
                for piano_file in pathlib.Path(input_dir).glob(pattern):
                    print(f"  Found Piano stem: {piano_file.name}")
                    return detect_key_from_file(str(piano_file), sr=self.sr)
            return detect_key_from_chroma(np.mean(self._chroma_full, axis=1))
        except Exception:
            return "? (?)"

    def _extract_drum_slice_features(self, beat_index: int, bpm: float,
                                    beats_per_bar: int, duration_str: str) -> tuple[np.ndarray, list[float]] | None:
        """Extract STFT fingerprint + 9 percussive-only features from a drum beat slice.

        Returns (fingerprint, [attack_slope, impulsivity, decay_rate, energy_db, energy_rel_db, 
                               band_low, band_mid, band_high, 0.0])
        or None if the slice is empty/invalid.
        """
        stft_slice = self.extract_stft_slice_for_beat(beat_index, bpm, beats_per_bar, duration_str)
        if stft_slice is None or np.linalg.norm(stft_slice) < 1e-8:
            print(f"[drum_grid] empty_stft_slice slice={beat_index} dur={duration_str}")
            return None

        fingerprint = stft_slice.astype(np.float32)
        fp_norm = float(np.linalg.norm(fingerprint))
        if fp_norm > 1e-8:
            fingerprint = fingerprint / fp_norm
        else:
            print(f"[drum_grid] zero_norm_stft slice={beat_index} dur={duration_str}")
            return None

        slice_dur = self.get_beat_slice_duration(bpm, beats_per_bar, duration_str)
        t_start = float(beat_index) * slice_dur
        t_end = t_start + slice_dur

        f0 = int(np.searchsorted(self.stft_times, t_start))
        f1 = int(np.searchsorted(self.stft_times, t_end))
        f1 = min(f1, self.stft_db.shape[1])
        if f1 <= f0:
            if len(self.stft_times) == 0:
                return None
            center_t = 0.5 * (t_start + t_end)
            nearest = int(np.argmin(np.abs(self.stft_times - center_t)))
            nearest = max(0, min(nearest, self.stft_db.shape[1] - 1))
            f0 = nearest
            f1 = min(nearest + 1, self.stft_db.shape[1])
            if f1 <= f0:
                return None

        # Extract magnitude window and compute percussive features
        window = self.stft_mag[:, f0:f1].astype(np.float64)
        window_db = self.stft_db[:, f0:f1].astype(np.float64)
        mean_spec = window.mean(axis=1)
        mean_spec_db = window_db.mean(axis=1)
        spec_sum = float(np.sum(mean_spec))
        if spec_sum <= 1e-12:
            print(f"[drum_grid] silent_stft_window slice={beat_index} frames={f0}:{f1}")
            return None

        # Percussive Feature 1: Attack Slope (dB rise per ms at beat boundary)
        # Compare pre-attack vs post-attack
        pre_frames = max(1, int(0.02 * self.sr / 512))  # 20ms pre
        post_frames = max(1, int(0.05 * self.sr / 512))  # 50ms post
        if f1 - f0 >= pre_frames + post_frames:
            pre_spec_db = window_db[:, :pre_frames].mean()
            post_spec_db = window_db[:, pre_frames:min(pre_frames + post_frames, window.shape[1])].mean()
            attack_slope_db_per_50ms = float(post_spec_db - pre_spec_db)
            attack_slope = float(np.clip((attack_slope_db_per_50ms + 30.0) / 60.0, 0.0, 1.0))
        else:
            attack_slope = 0.5

        # Percussive Feature 2: Impulsivity (peak / RMS ratio)
        peak_db = float(np.max(window_db))
        rms_db = float(np.sqrt(np.mean(window_db ** 2)))
        impulsivity_ratio = float(np.clip((peak_db - rms_db) / 20.0, 0.0, 1.0))

        # Percussive Feature 3: Decay Rate (dB drop over 100ms from peak)
        if f1 - f0 > int(0.1 * self.sr / 512):  # At least 100ms
            peak_frame = int(np.argmax(window_db.mean(axis=0)))
            decay_window_frames = min(int(0.1 * self.sr / 512), window.shape[1] - peak_frame - 1)
            if decay_window_frames > 1:
                peak_val_db = window_db[:, peak_frame].mean()
                decay_val_db = window_db[:, peak_frame + decay_window_frames].mean()
                decay_db_per_100ms = float(peak_val_db - decay_val_db)
                decay_rate = float(np.clip(decay_db_per_100ms / 30.0, 0.0, 1.0))
            else:
                decay_rate = 0.5
        else:
            decay_rate = 0.5

        # Absolute & relative energy
        energy_db = float(20.0 * np.log10(max(spec_sum, 1e-8)))
        energy_rel_db = float(energy_db - (-30.0))  # Normalize to typical background

        # Band energy distribution (low/mid/high frequencies)
        freq_hz = self.stft_freqs.astype(np.float64)
        band_low_mask = freq_hz < 200  # Kick range
        band_mid_mask = (freq_hz >= 200) & (freq_hz < 2000)  # Snare/Tom range
        band_high_mask = freq_hz >= 2000  # Cymbal range
        
        band_low_energy = float(np.sum(mean_spec[band_low_mask]))
        band_mid_energy = float(np.sum(mean_spec[band_mid_mask]))
        band_high_energy = float(np.sum(mean_spec[band_high_mask]))
        total_energy = band_low_energy + band_mid_energy + band_high_energy
        
        if total_energy > 1e-8:
            band_low = band_low_energy / total_energy
            band_mid = band_mid_energy / total_energy
            band_high = band_high_energy / total_energy
        else:
            band_low = band_mid = band_high = 1.0 / 3.0

        return fingerprint, [attack_slope, impulsivity_ratio, decay_rate, energy_db, energy_rel_db,
                             band_low, band_mid, band_high, 0.0]

    def add_manual_drum(self, beat_index: int, drum_midi: int, bpm: float,
                       beats_per_bar: int, duration_str: str) -> bool:
        """Train a drum pattern from a beat slice (left click confirm)."""
        if self.model is None:
            return False
        
        result = self._extract_drum_slice_features(beat_index, bpm, beats_per_bar, duration_str)
        if result is None:
            print(f"[drum_grid] add_failed slice={beat_index} drum={drum_midi} dur={duration_str}")
            return False
        
        fingerprint, features = result
        attack_slope, impulsivity, decay_rate, energy_db, energy_rel_db, band_low, band_mid, band_high, _ = features
        
        sample_key = f"beat:{beat_index}:{int(drum_midi)}"
        self.model.record_drum_pattern(
            drum_midi=int(drum_midi),
            fingerprint=fingerprint,
            attack_slope=attack_slope,
            impulsivity=impulsivity,
            decay_rate=decay_rate,
            energy_db=energy_db,
            energy_rel_db=energy_rel_db,
            band_low=band_low,
            band_mid=band_mid,
            band_high=band_high,
            keep=True,
            sample_key=sample_key,
            source="user"
        )
        manual = self._manual_drums.setdefault(int(beat_index), set())
        manual.add(int(drum_midi))
        rejects = self._manual_drum_rejects.get(int(beat_index))
        if rejects is not None:
            rejects.discard(int(drum_midi))
        print(
            f"[drum_grid] add_ok slice={beat_index} drum={drum_midi} "
            f"attack={attack_slope:.2f} impulsivity={impulsivity:.2f} decay={decay_rate:.2f} "
            f"dB={energy_db:.1f} patterns={len(self.model.get_drum_patterns(expected_size=len(fingerprint)))}"
        )
        return True
    
    def remove_manual_drum(self, beat_index: int, drum_midi: int, bpm: float,
                          beats_per_bar: int, duration_str: str) -> bool:
        """Record negative drum training sample (right click reject)."""
        if self.model is None:
            return False
        
        result = self._extract_drum_slice_features(beat_index, bpm, beats_per_bar, duration_str)
        if result is None:
            print(f"[drum_grid] reject_failed slice={beat_index} drum={drum_midi} dur={duration_str}")
            return False
        
        fingerprint, features = result
        attack_slope, impulsivity, decay_rate, energy_db, energy_rel_db, band_low, band_mid, band_high, _ = features
        
        sample_key = f"beat:{beat_index}:{int(drum_midi)}"
        self.model.record_drum_pattern(
            drum_midi=int(drum_midi),
            fingerprint=fingerprint,
            attack_slope=attack_slope,
            impulsivity=impulsivity,
            decay_rate=decay_rate,
            energy_db=energy_db,
            energy_rel_db=energy_rel_db,
            band_low=band_low,
            band_mid=band_mid,
            band_high=band_high,
            keep=False,
            sample_key=sample_key,
            source="user"
        )
        manual = self._manual_drums.get(int(beat_index))
        if manual is not None:
            manual.discard(int(drum_midi))
            if not manual:
                self._manual_drums.pop(int(beat_index), None)
        rejects = self._manual_drum_rejects.setdefault(int(beat_index), set())
        rejects.add(int(drum_midi))
        print(
            f"[drum_grid] reject_ok slice={beat_index} drum={drum_midi} "
            f"attack={attack_slope:.2f} impulsivity={impulsivity:.2f} decay={decay_rate:.2f} "
            f"dB={energy_db:.1f}"
        )
        return True
