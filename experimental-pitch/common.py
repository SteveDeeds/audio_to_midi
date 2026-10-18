"""Shared helpers for the pitch-detection bake-off: audio I/O, note
segmentation, MIDI export and simple additive resynthesis."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import scipy.ndimage
import soundfile as sf

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DEFAULT_SONG_DIR = REPO_ROOT / "Firm Foundation (He Won't)"
OUT_DIR = ROOT / "outputs"
STEM_NAMES = ("bass", "vocals", "other", "piano", "guitar", "drums")
PITCHED_STEMS = ("bass", "vocals", "other", "piano", "guitar")
GM_PROGRAM = {"bass": 33, "vocals": 53, "other": 0, "piano": 0, "guitar": 25}

# Populated by configure(); defaults to the Firm Foundation stems.
SONG = {"dir": DEFAULT_SONG_DIR, "stems": {}, "bpm": 75.0}


def discover_stems(song_dir: Path) -> dict[str, Path]:
    """Find stems in UVR style ('..._(Bass).wav') or Demucs/EchoGrid style ('bass.wav')."""
    found: dict[str, Path] = {}
    for p in sorted(Path(song_dir).glob("*.wav")):
        low = p.stem.lower()
        for name in STEM_NAMES:
            if low == name or re.search(rf"\({name}\)$", low):
                found[name] = p
    return found


def detect_bpm(song_dir: Path, stems: dict[str, Path]) -> float | None:
    """EchoGrid convention: 'NNNbpm' in a file or folder name; Ableton exports here use 'NN exact'."""
    names = [p.name for p in stems.values()] + [Path(song_dir).name, Path(song_dir).parent.name]
    for pattern in (r"(\d+(?:\.\d+)?)\s*bpm", r"(\d+(?:\.\d+)?)\s*exact"):
        for n in names:
            m = re.search(pattern, n, re.IGNORECASE)
            if m:
                return float(m.group(1))
    return None


def configure(song_dir: Path | str | None = None, bpm: float | None = None) -> dict:
    d = Path(song_dir) if song_dir else DEFAULT_SONG_DIR
    stems = discover_stems(d)
    if not stems:
        raise SystemExit(f"No stems found in {d}")
    SONG.update(dir=d, stems=stems, bpm=float(bpm or detect_bpm(d, stems) or 120.0))
    return SONG


@dataclass
class Note:
    start: float
    end: float
    pitch: int
    velocity: int = 90

    @property
    def dur(self) -> float:
        return self.end - self.start


def load_stem(stem: str, sr: int, start: float = 0.0, duration: float | None = None) -> np.ndarray:
    if not SONG["stems"]:
        configure()
    y, _ = librosa.load(str(SONG["stems"][stem]), sr=sr, mono=True, offset=start, duration=duration)
    return y.astype(np.float32)


def song_out_dir() -> Path:
    return OUT_DIR / SONG["dir"].name


def onset_times(y: np.ndarray, sr: int, low_freq: bool = False) -> np.ndarray:
    """Spectral-flux onsets; low_freq restricts flux to < ~500 Hz for bass."""
    hop = 256
    if low_freq:
        S = np.abs(librosa.cqt(y, sr=sr, hop_length=hop, fmin=librosa.note_to_hz("C1"), n_bins=60))
        env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S, ref=np.max), sr=sr, hop_length=hop)
    else:
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop, backtrack=False,
                                        delta=0.08, wait=4)
    return librosa.frames_to_time(frames, sr=sr, hop_length=hop)


def attack_onsets(y: np.ndarray, sr: int) -> np.ndarray:
    """SuperFlux-style onsets (mel flux with lag 2 and a 3-bin max filter that
    suppresses vibrato/slide energy). On the bass stem these lock to the beat
    grid ~3x better than note boundaries derived from pitch changes."""
    hop = 128
    M = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop, n_mels=128, fmax=8000)
    env = librosa.onset.onset_strength(S=librosa.power_to_db(M, ref=np.max), sr=sr, hop_length=hop,
                                       lag=2, max_size=3)
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop, delta=0.03,
                                        wait=int(0.1 * sr / hop), backtrack=False)
    return librosa.frames_to_time(frames, sr=sr, hop_length=hop)


# --------------------------------------------------------------------------- key awareness
PC_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
# Krumhansl-Kessler key profiles.
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
SCALE_STEPS = {"major": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10, 11)}  # minor incl. leading tone


def key_scores(chroma_mean: np.ndarray) -> list[tuple[float, int, str]]:
    out = []
    for mode, prof in (("major", KK_MAJOR), ("minor", KK_MINOR)):
        for tonic in range(12):
            out.append((float(np.corrcoef(chroma_mean, np.roll(prof, tonic))[0, 1]), tonic, mode))
    return sorted(out, reverse=True)


def make_key(tonic: int, mode: str, confidence: float | None = None) -> dict:
    return {"tonic": int(tonic), "mode": mode, "name": f"{PC_NAMES[tonic]} {mode}",
            "scale": sorted((tonic + s) % 12 for s in SCALE_STEPS[mode]), "confidence": confidence}


def parse_key(text: str) -> dict:
    """'Bb major', 'g minor', 'F#m', 'Eb'."""
    t = text.strip().replace("♭", "b").replace("♯", "#")
    mode = "minor" if t.lower().endswith(("minor", "min", "m")) and not t.lower().endswith("major") else "major"
    root = t.split()[0].rstrip("m") if " " in t else (t[:-1] if mode == "minor" and t.endswith("m") else t)
    root = root[0].upper() + root[1:]
    alias = {"Db": "C#", "D#": "Eb", "Gb": "F#", "G#": "Ab", "A#": "Bb", "Cb": "B", "Fb": "E", "E#": "F", "B#": "C"}
    return make_key(PC_NAMES.index(alias.get(root, root)), mode)


def estimate_key(chroma: np.ndarray) -> dict:
    """Global key from a [12, n] chroma (Krumhansl-Schmuckler)."""
    scores = key_scores(chroma.mean(axis=1))
    best = scores[0]
    key = make_key(best[1], best[2], round(best[0], 3))
    key["runner_up"] = f"{PC_NAMES[scores[1][1]]} {scores[1][2]} ({scores[1][0]:.3f})"
    return key


def harmony_chroma(y: np.ndarray, sr: int, start: float = 0.0, hop: int = 2048,
                   smooth_s: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Beat-smoothed chroma of an accompaniment stem, normalised per frame to max 1."""
    C = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop, norm=None)
    C = scipy.ndimage.uniform_filter1d(C, size=max(1, int(smooth_s * sr / hop)), axis=1)
    level = C.sum(axis=0)
    C = C / (C.max(axis=0, keepdims=True) + 1e-9)
    C[:, level < 0.05 * np.percentile(level, 95)] = 0.0  # silence -> no harmonic evidence
    return librosa.times_like(C, sr=sr, hop_length=hop) + start, C


def allowed_pitch_classes(times: np.ndarray, key: dict, harmony: tuple[np.ndarray, np.ndarray] | None = None,
                          harmony_thresh: float = 0.8) -> np.ndarray:
    """[12, n_frames] bool: in the key's scale, or strongly present in the local harmony
    (protects genuine chromatic notes such as secondary-dominant roots)."""
    allowed = np.zeros((12, len(times)), dtype=bool)
    allowed[key["scale"]] = True
    if harmony is not None:
        h_t, h_c = harmony
        idx = np.clip(np.searchsorted(h_t, times), 0, len(h_t) - 1)
        allowed |= h_c[:, idx] >= harmony_thresh
    return allowed


def key_clean_contour(q: np.ndarray, raw: np.ndarray, allowed: np.ndarray, hop: float,
                      hold_s: float = 0.25) -> tuple[np.ndarray, int]:
    """Fold short out-of-key pitch runs into the neighbouring in-key note.

    A run of a non-diatonic (and not locally harmonised) semitone shorter than
    hold_s is assumed to be part of a bend/slide/scoop: it takes the pitch of
    the nearest-in-pitch adjacent in-key run within the same voiced phrase
    (ties go to the following note, i.e. the slide target). Isolated runs snap
    to the closest allowed semitone. Returns (cleaned q, number of runs changed).
    """
    q = q.copy()
    n = len(q)
    runs: list[list] = []  # [start, end_excl, pitch, ok]
    i = 0
    while i < n:
        if np.isnan(q[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and q[j + 1] == q[i]:
            j += 1
        p = int(q[i])
        in_key = allowed[p % 12, i:j + 1].mean() >= 0.5
        runs.append([i, j + 1, p, in_key or (j + 1 - i) * hop >= hold_s])
        i = j + 1
    changed = 0
    for k, (s, e, p, ok) in enumerate(runs):
        if ok:
            continue
        target_pitch = float(np.nanmedian(raw[s:e]))
        cands = []
        for step in (1, -1):  # following run first so ties prefer the slide target
            m = k + step
            while 0 <= m < len(runs):
                contiguous = (runs[m][0] == runs[m - 1][1]) if step == 1 else (runs[m + 1][0] == runs[m][1])
                if not contiguous:
                    break
                if runs[m][3]:
                    cands.append(runs[m][2])
                    break
                m += step
        if cands:
            new = min(cands, key=lambda c: abs(c - target_pitch))
        else:
            frame_ok = allowed[:, s:e].mean(axis=1) >= 0.5
            options = [p + d for d in (-1, 1, -2, 2) if frame_ok[(p + d) % 12]]
            new = min(options, key=lambda c: abs(c - target_pitch)) if options else p
        if new != p:
            q[s:e] = new
            changed += 1
    return q, changed


def onset_segment(times: np.ndarray, midi: np.ndarray, onsets: np.ndarray, min_dur: float = 0.09,
                  stable_s: float = 0.12, gap_s: float = 0.08, smooth_frames: int = 5,
                  rms: np.ndarray | None = None, allowed: np.ndarray | None = None) -> list[Note]:
    """Onset-driven note segmentation of a mono f0 contour (NaN = unvoiced).

    Notes start at attacks (or at voicing that resumes after a gap); each takes
    the pitch of its stable region, so slides/scoops fold into the target note.
    A legato change to a new pitch held >= stable_s still starts a new note.
    """
    hop = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
    voiced = ~np.isnan(midi)
    filled = np.where(voiced, midi, 0.0)
    q = np.where(voiced, np.round(scipy.ndimage.median_filter(filled, size=smooth_frames, mode="nearest")), np.nan)
    # Key cleaning only relabels pitches; boundaries/stability come from the uncleaned contour,
    # otherwise folding a slide into its target can create spurious legato splits.
    qk = key_clean_contour(q, midi, allowed, hop)[0] if allowed is not None else q

    starts = list(np.searchsorted(times, onsets))
    gap = int(round(gap_s / hop))
    unv_run = 0
    for i, v in enumerate(voiced):  # voicing that resumes after a real gap is an implicit onset
        if v and unv_run >= gap:
            starts.append(i)
        unv_run = 0 if v else unv_run + 1
    bounds = sorted(set(int(s) for s in starts if 0 <= s < len(times))) + [len(times)]

    stable_n = max(1, int(round(stable_s / hop)))
    notes: list[Note] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = np.where(voiced[a:b])[0] + a
        if len(idx) == 0:
            continue
        # End the note where voicing stops for good inside this segment.
        end_i = idx[-1]
        seg = q[a:end_i + 1]
        segk = qk[a:end_i + 1]
        runs: list[tuple[int, int, int]] = []  # (start, length, pitch)
        i = 0
        while i < len(seg):
            if np.isnan(seg[i]):
                i += 1
                continue
            j = i
            while j + 1 < len(seg) and seg[j + 1] == seg[i]:
                j += 1
            runs.append((i, j - i + 1, int(segk[i])))
            i = j + 1
        stable = [r for r in runs if r[1] >= stable_n]
        if not stable:
            vals = segk[~np.isnan(segk)][max(1, int(0.04 / hop)):]
            if len(vals) == 0:
                vals = segk[~np.isnan(segk)]
            pitch = int(np.bincount(vals.astype(int)).argmax())
            stable = [(0, len(seg), pitch)]
        merged = [stable[0]]
        for r in stable[1:]:
            if r[2] != merged[-1][2]:
                merged.append(r)
        for k, (rs, _, pitch) in enumerate(merged):
            t0 = times[a] if k == 0 else times[a + rs]
            t1 = times[a + merged[k + 1][0]] if k + 1 < len(merged) else times[end_i] + hop
            if t1 - t0 >= min_dur:
                notes.append(Note(float(t0), float(t1), pitch))
    if rms is not None:
        assign_velocity(notes, times, rms)
    return notes


def contour_to_notes(times: np.ndarray, midi: np.ndarray, voiced: np.ndarray,
                     onsets: np.ndarray | None = None, min_dur: float = 0.07,
                     smooth_frames: int = 5, rms: np.ndarray | None = None,
                     allowed: np.ndarray | None = None) -> list[Note]:
    """Convert a monophonic f0 contour (MIDI, NaN where unvoiced) into notes.

    Boundaries: voicing changes, stable semitone changes, and detected onsets
    (to separate repeated same-pitch notes, which contours alone cannot do).
    With `allowed` (see allowed_pitch_classes), short out-of-key runs are
    folded into neighbouring in-key notes first.
    """
    midi = np.where(voiced, midi, np.nan).astype(float)
    raw = midi.copy()
    if smooth_frames > 1:
        filled = np.where(np.isnan(midi), 0.0, midi)
        med = scipy.ndimage.median_filter(filled, size=smooth_frames, mode="nearest")
        midi = np.where(np.isnan(midi), np.nan, med)
    q = np.round(midi)
    hop = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
    if allowed is not None:
        q, _ = key_clean_contour(q, raw, allowed, hop)
    notes: list[Note] = []
    seg_start = None
    for i in range(len(q) + 1):
        cur = q[i] if i < len(q) else np.nan
        prev = q[i - 1] if i > 0 else np.nan
        boundary = (np.isnan(cur) != np.isnan(prev)) or (not np.isnan(cur) and cur != prev)
        if boundary:
            if seg_start is not None and not np.isnan(prev):
                notes.append(Note(float(times[seg_start]), float(times[i - 1] + hop), int(prev)))
            seg_start = i if not np.isnan(cur) else None
    # Absorb short blips caused by vibrato/scoops into the neighbour with the same pitch.
    merged: list[Note] = []
    for n in notes:
        if merged and n.dur < min_dur and n.start - merged[-1].end < 0.03:
            merged[-1].end = n.end
            continue
        if merged and merged[-1].pitch == n.pitch and n.start - merged[-1].end < 0.03:
            merged[-1].end = n.end
            continue
        merged.append(n)
    notes = [n for n in merged if n.dur >= min_dur]
    if onsets is not None and len(onsets):
        notes = split_at_onsets(notes, onsets, min_dur)
    if rms is not None:
        assign_velocity(notes, times, rms)
    return notes


def split_at_onsets(notes: list[Note], onsets: np.ndarray, min_dur: float) -> list[Note]:
    out: list[Note] = []
    for n in notes:
        cuts = [t for t in onsets if n.start + min_dur < t < n.end - min_dur]
        s = n.start
        for c in cuts:
            out.append(Note(s, float(c), n.pitch, n.velocity))
            s = float(c)
        out.append(Note(s, n.end, n.pitch, n.velocity))
    return out


def assign_velocity(notes: list[Note], times: np.ndarray, rms: np.ndarray) -> None:
    if not notes:
        return
    db = librosa.amplitude_to_db(rms + 1e-9, ref=np.max(rms) + 1e-9)
    for n in notes:
        i0, i1 = np.searchsorted(times, [n.start, n.start + 0.1])
        seg = db[i0:max(i1, i0 + 1)]
        level = float(seg.max()) if len(seg) else -40.0
        n.velocity = int(np.clip(110 + level * 1.5, 30, 120))


def roll_to_notes(roll: np.ndarray, times: np.ndarray, midi_lo: int, threshold: float,
                  min_dur: float = 0.07, onsets: np.ndarray | None = None) -> list[Note]:
    """Threshold an activation roll [n_pitches, n_frames] into notes."""
    hop = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
    active = roll >= threshold
    active = scipy.ndimage.binary_closing(active, structure=np.ones((1, 3)))
    notes: list[Note] = []
    for k in range(roll.shape[0]):
        row = active[k]
        if not row.any():
            continue
        edges = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        for s, e in zip(starts, ends):
            t0, t1 = float(times[s]), float(times[e - 1] + hop)
            if t1 - t0 >= min_dur:
                peak = float(roll[k, s:e].max())
                notes.append(Note(t0, t1, midi_lo + k, int(np.clip(40 + 80 * peak / (roll.max() + 1e-9), 30, 120))))
    if onsets is not None and len(onsets):
        notes = split_at_onsets(notes, onsets, min_dur)
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


def enforce_monophonic(notes: list[Note], prefer_low: bool = True, min_dur: float = 0.04) -> list[Note]:
    """Reduce a polyphonic note list to one voice (bass/lead vocal assumption)."""
    out: list[Note] = []
    for n in sorted(notes, key=lambda n: (n.start, -n.velocity)):
        n = Note(n.start, n.end, n.pitch, n.velocity)
        if out and n.start < out[-1].end:
            prev = out[-1]
            if n.start - prev.start < 0.05:  # simultaneous onsets, e.g. octave duplicates
                key = lambda x: (x.velocity, -x.pitch if prefer_low else x.pitch)
                out[-1] = max(prev, n, key=key)
                out[-1].end = max(prev.end, n.end)
                continue
            if n.end <= prev.end and n.velocity < prev.velocity:
                continue  # weaker note buried inside a sustained one
            prev.end = n.start
        out.append(n)
    return [n for n in out if n.dur >= min_dur]


def song_key(out_dir: Path, chroma_source: np.ndarray, sr: int, override: str | None = None) -> dict:
    """Key for the song: --key override, else cached key.json, else estimated and cached."""
    path = out_dir / "key.json"
    if override:
        key = parse_key(override) | {"source": "user"}
    elif path.exists():
        return json.loads(path.read_text())
    else:
        C = librosa.feature.chroma_cqt(y=chroma_source, sr=sr, hop_length=4096)
        key = estimate_key(C) | {"source": "estimated"}
    path.write_text(json.dumps(key, indent=1))
    return key


def write_midi(notes: list[Note], path: Path, program: int = 0, bpm: float | None = None) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm or SONG["bpm"])
    inst = pretty_midi.Instrument(program=program)
    for n in notes:
        inst.notes.append(pretty_midi.Note(velocity=int(n.velocity), pitch=int(n.pitch),
                                           start=float(n.start), end=float(n.end)))
    pm.instruments.append(inst)
    pm.write(str(path))


def save_notes_json(notes: list[Note], path: Path, meta: dict) -> None:
    path.write_text(json.dumps({"meta": meta, "notes": [asdict(n) for n in notes]}, indent=1))


def load_notes_json(path: Path) -> tuple[list[Note], dict]:
    d = json.loads(path.read_text())
    return [Note(**n) for n in d["notes"]], d["meta"]


def synthesize(notes: list[Note], sr: int, start: float, duration: float) -> np.ndarray:
    """Cheap additive synth (4 harmonics, plucked envelope) for A/B listening."""
    out = np.zeros(int(sr * duration), dtype=np.float32)
    for n in notes:
        if n.end <= start or n.start >= start + duration:
            continue
        s = max(n.start, start)
        e = min(n.end, start + duration) + 0.05
        t = np.arange(int((e - s) * sr)) / sr
        f0 = 440.0 * 2 ** ((n.pitch - 69) / 12)
        phase_t = t + (s - n.start)
        wave = sum((1.0 / h) * np.sin(2 * np.pi * f0 * h * phase_t) for h in range(1, 5) if f0 * h < sr / 2)
        env = np.exp(-phase_t / 1.2) * np.minimum(1.0, phase_t / 0.005)
        rel = np.clip((min(n.end, start + duration) + 0.05 - (s + t)) / 0.05, 0, 1)
        sig = (wave * env * rel * (n.velocity / 127.0)).astype(np.float32)
        i0 = int((s - start) * sr)
        i1 = min(len(out), i0 + len(sig))
        out[i0:i1] += sig[: i1 - i0]
    peak = np.abs(out).max()
    return out / peak * 0.8 if peak > 0 else out


def write_wav(path: Path, y: np.ndarray, sr: int) -> None:
    sf.write(str(path), y, sr, subtype="PCM_16")
