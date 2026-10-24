"""Project use cases shared by the CLI and the Qt shell."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math
import uuid

import soundfile as sf

from audio_to_midi.domain import Note, Project, Track
from audio_to_midi.adapters import MODELS, Prediction, TrackBackend, normalize_key
from audio_to_midi.notation import preferred_key


def add_stem(project: Project, path: Path, role: str) -> Track:
    if role not in MODELS:
        raise ValueError(f"Unknown stem role: {role}")
    path = path.resolve(strict=True)
    info = _stem_info(project, path)
    if any(t.role == role for t in project.tracks):
        raise ValueError(f"A {role} track already exists in this project.")
    track = Track(
        id=uuid.uuid4().hex, role=role, audio_path=str(path),
        duration=info.duration, sample_rate=info.samplerate,
        model=MODELS[role][0] if MODELS[role] else "",
    )
    track.settings["feature_schema"] = "legacy-cqt-v1"
    project.tracks.append(track)
    project.revision += 1
    return track


def _stem_info(project: Project, path: Path) -> sf._SoundFileInfo:
    from audio_to_midi.pipeline import decoded_frames

    info = sf.info(path)
    if info.frames <= 0:
        raise ValueError(f"Empty audio file: {path}")
    if project.grid.offset_seconds >= info.duration:
        raise ValueError("Declared downbeat lies beyond this audio.")
    reference_path = project.mix_path or (project.tracks[0].audio_path if project.tracks else None)
    reference = sf.info(reference_path) if reference_path else None
    if reference and (
        reference.samplerate != info.samplerate
        or abs(decoded_frames(reference_path) - decoded_frames(path)) > 1
    ):
        raise ValueError(
            "Stem sample rate/duration differs from the project reference. "
            "Explicitly resample/pad externally, then reimport and audition alignment."
        )
    return info


def replace_stem(project: Project, track: Track, path: Path) -> None:
    """Swap a track's audio (e.g. re-separated stem); authored notes keep their bar/beat positions."""
    path = path.resolve(strict=True)
    info = _stem_info(project, path)
    track.audio_path = str(path)
    track.duration = info.duration
    track.sample_rate = info.samplerate
    track.notes.generated = []
    track.notes.revision += 1
    track.revision += 1
    project.revision += 1


def track_end(project: Project, track: Track) -> Fraction:
    return max(Fraction(0), project.grid.seconds_to_beat(track.duration))


def media_identity(path: str) -> tuple[str, int, int]:
    stat = Path(path).stat()
    return path, stat.st_size, stat.st_mtime_ns


def harmony_identity(project: Project) -> list[tuple[str, int, int]]:
    paths = [project.mix_path] if project.mix_path else [
        t.audio_path for t in project.tracks if t.role != "drums"
    ]
    # Other remains local harmony evidence even when a mix is supplied.
    paths += [t.audio_path for t in project.tracks if t.role == "other" and t.audio_path not in paths]
    return [media_identity(path) for path in paths]


def identity(project: Project, track: Track) -> str:
    return hashlib.sha256(json.dumps({
        "project": project.id, "track": track.id,
        "file": media_identity(track.audio_path),
        "grid": repr(project.grid), "key": project.key, "model": track.model,
        "harmony": harmony_identity(project),
        "settings": {k: v for k, v in track.settings.items() if k not in ("coverage", "coverage_identity")},
        "training": track.training, "revision": track.revision,
    }, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class ResultToken:
    project_id: str
    track_id: str
    identity: str
    edits: int


def token(project: Project, track: Track) -> ResultToken:
    return ResultToken(project.id, track.id, identity(project, track), track.notes.revision)


def publish(project: Project, result_token: ResultToken, prediction: Prediction) -> bool:
    if project.id != result_token.project_id:
        return False
    try:
        track = project.track(result_token.track_id)
    except KeyError:
        return False
    if result_token != token(project, track):
        return False
    track.notes.replace_generated(prediction.start, prediction.end, list(prediction.notes))
    track.notes.onsets = sorted(
        {b for b in track.notes.onsets if not prediction.start <= b < prediction.end}
        | set(prediction.onsets)
    )
    coverage = track.settings.get("coverage", []) if (
        track.settings.get("coverage_identity") == result_token.identity
    ) else []
    intervals = sorted([
        (Fraction(a), Fraction(b)) for a, b in coverage
    ] + [(prediction.start, prediction.end)])
    merged: list[list[Fraction]] = []
    for a, b in intervals:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    track.settings["coverage"] = [[str(a), str(b)] for a, b in merged]
    track.settings["coverage_identity"] = result_token.identity
    return True


def complete(project: Project, track: Track) -> bool:
    if track.role == "other":
        return True
    if track.settings.get("imported_midi") and not track.notes.generated:
        return True
    return covered(project, track, Fraction(0), track_end(project, track))


def covered(project: Project, track: Track, start: Fraction, end: Fraction) -> bool:
    if track.settings.get("coverage_identity") != identity(project, track):
        return False
    return any(
        Fraction(a) <= start and Fraction(b) >= end
        for a, b in track.settings.get("coverage", [])
    )


class TranscriptionService:
    """Call from one worker lane; provider caches never cross worker threads."""

    def __init__(self) -> None:
        self._backends: dict[str, tuple[str, TrackBackend]] = {}

    def predict(
        self, project: Project, track: Track, start: Fraction, end: Fraction,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> Prediction:
        if start < 0 or end <= start or end > track_end(project, track):
            raise ValueError("Invalid transcription interval.")
        if project.grid.snap(start) != start:
            raise ValueError("Transcription starts must lie on the selected onset grid.")
        backend_key = repr((
            project.id, track.id, media_identity(track.audio_path),
            project.grid, project.key, track.model,
            harmony_identity(project),
            track.settings.get("license_acknowledged", False),
        ))
        cached = self._backends.get(track.id)
        if cached is None or cached[0] != backend_key:
            backend = TrackBackend.create(track, project)
            self._backends[track.id] = backend_key, backend
        else:
            backend = cached[1]
        return backend.predict(track, start, end, cancelled)


def record_onset(track: Track, beat: Fraction, keep: bool) -> None:
    track.training[:] = [
        row for row in track.training
        if not (row.get("domain") == "onset" and row.get("beat") == str(beat))
    ]
    track.training.append({"domain": "onset", "beat": str(beat), "keep": keep})
    track.revision += 1


def record_drum(track: Track, project: Project, start: Fraction, end: Fraction, pitch: int, keep: bool) -> None:
    t0, t1 = project.grid.beat_to_seconds(start), project.grid.beat_to_seconds(end)
    track.training[:] = [
        row for row in track.training
        if not (row.get("domain") == "drum" and row.get("start") == t0 and row.get("pitch") == pitch)
    ]
    track.training.append({"domain": "drum", "start": t0, "end": t1, "pitch": pitch, "keep": keep})
    track.revision += 1


def mark_project_edited(project: Project) -> None:
    """Protect attached preparation revisions before authoring downstream state."""
    preparation = project.stages.get("preparation")
    if preparation:
        from audio_to_midi.pipeline import ArtifactStore
        store = ArtifactStore(preparation["directory"])
        for name in ("separation_revision",):
            if preparation.get(name):
                store.mark_edited(preparation[name])


def import_profile(track: Track, path: Path) -> list[str]:
    from core.models import CorrectionModel

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Model profile must contain a JSON object.")
    if payload.get("feature_schema", "legacy-cqt-v1") != "legacy-cqt-v1":
        raise ValueError(
            "Profile uses incompatible onset features; expected legacy-cqt-v1."
        )
    rows = CorrectionModel.candidate_rows_from_payload(payload)
    if not rows:
        raise ValueError("The profile has no compatible onset examples.")
    if any(
        len(row["row"]) != 4 or not all(math.isfinite(value) for value in row["row"])
        or row["label"] not in (0, 1)
        for row in rows
    ):
        raise ValueError("Profile onset examples require four finite features and binary labels.")
    track.settings["onset_profile"] = {"cand": rows}
    track.revision += 1
    return [key for key in payload if key not in ("cand", "feature_schema")]


def mix_cqt(path: str, cancelled: Callable[[], bool] = lambda: False):
    """(semitone CQT image, frame seconds, duration) of a whole file for chords and key."""
    import librosa
    from audio_to_midi.adapters import Cancelled
    from audio_to_midi.spectrogram import HOP_LENGTH, SAMPLE_RATE, cqt_image

    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if cancelled():
        raise Cancelled()
    return cqt_image(audio, SAMPLE_RATE), HOP_LENGTH / SAMPLE_RATE, len(audio) / SAMPLE_RATE


def detect_key(cqt) -> str:
    from audio_to_midi.chords import fold_chroma, harmonic_power
    from audio_to_midi.spectrogram import LOW_PITCH

    chroma = fold_chroma(harmonic_power(cqt), LOW_PITCH)
    if not chroma.any():
        raise ValueError("No pitched harmony is audible in the mix.")
    return key_from_chroma(chroma.mean(axis=1))


def key_from_chroma(vector) -> str:
    """Best-matching key for a 12-bin pitch-class profile, spelled with the fewest accidentals."""
    from core.analysis import detect_key_from_chroma

    return preferred_key(normalize_key(detect_key_from_chroma(vector)))


HARMONY_ROLES = ("piano", "guitar", "other")
VOCAL_KEY_WEIGHT = 0.5


@dataclass(frozen=True)
class HarmonySources:
    """Audio used for chords: summed chord stems, a bass stem for slash basses and vocals for the key.

    Without current separated stems ``chords`` is just the full mix.
    """

    chords: tuple[str, ...]
    bass: str | None = None
    vocals: str | None = None
    from_stems: bool = False

    def identity(self) -> tuple:
        paths = (*self.chords, self.bass, self.vocals)
        return tuple(media_identity(p) if p and Path(p).exists() else p for p in paths)


def harmony_sources(project: Project) -> HarmonySources:
    """Chord, bass and vocal stems from the current separation; the mix when there are none.

    Piano, guitar and other carry the chord shape, and the bass stem is added so
    the root is heard. Drums never count. Vocals are left out of chord shapes
    (scoops, vibrato and melody passing tones) and only help the key, unless the
    other stems are silent.
    """
    usable = {
        t.role: t.audio_path for t in project.tracks
        if t.audio_path and Path(t.audio_path).exists()
    }
    harmonic = tuple(usable[role] for role in HARMONY_ROLES if role in usable)
    if not harmonic:
        if not project.mix_path:
            raise ValueError("Import a mix before detecting chords.")
        return HarmonySources((project.mix_path,))
    bass = usable.get("bass")
    return HarmonySources(harmonic + ((bass,) if bass else ()), bass, usable.get("vocals"), True)


def stems_for_chords(project: Project) -> bool:
    """True when current piano/guitar/other stems exist, so chords need not come from the mix."""
    return any(
        t.role in HARMONY_ROLES and t.audio_path and Path(t.audio_path).exists()
        for t in project.tracks
    )


def _load_sum(paths, cancelled: Callable[[], bool]):
    import librosa
    import numpy as np
    from audio_to_midi.adapters import Cancelled
    from audio_to_midi.spectrogram import SAMPLE_RATE

    total = np.zeros(0, dtype=np.float32)
    for path in paths:
        if cancelled():
            raise Cancelled()
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        total = _add(total, audio)
    return total


def _add(a, b, weight: float = 1.0):
    import numpy as np

    out = np.zeros(max(len(a), len(b)), dtype=np.float32)
    out[:len(a)] += a
    out[:len(b)] += weight * b
    return out


def _rms(audio) -> float:
    import numpy as np

    return float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0


def harmony_audio(project: Project, cancelled: Callable[[], bool] = lambda: False):
    """(chord audio, bass audio or None, key audio) as 22050 Hz mono, from the stems or the mix."""
    from audio_to_midi.adapters import Cancelled

    sources = harmony_sources(project)
    if not sources.from_stems:
        mix = _load_sum(sources.chords, cancelled)
        return mix, None, mix
    # The bass is left out of the key: a loud bass dwelling on IV pulls the key there.
    harmonic = _load_sum([p for p in sources.chords if p != sources.bass], cancelled)
    bass_audio = _load_sum([sources.bass], cancelled) if sources.bass else None
    chord_audio = harmonic if bass_audio is None else _add(harmonic, bass_audio)
    key_audio = harmonic
    if sources.vocals:
        vocals = _load_sum([sources.vocals], cancelled)
        if _rms(chord_audio) < 0.05 * _rms(vocals):  # a cappella: the voice is the harmony
            chord_audio = _add(chord_audio, vocals)
            key_audio = _add(harmonic, vocals)
        else:
            key_audio = _add(harmonic, vocals, VOCAL_KEY_WEIGHT)
    if cancelled():
        raise Cancelled()
    return chord_audio, bass_audio, key_audio


def harmony_cqt(project: Project, cancelled: Callable[[], bool] = lambda: False, audio=None):
    """(chord CQT, bass CQT or None, key CQT, frame seconds, duration) for chord detection and display."""
    from audio_to_midi.spectrogram import HOP_LENGTH, SAMPLE_RATE, cqt_image

    if audio is None and not harmony_sources(project).from_stems:
        cqt, frame_seconds, duration = mix_cqt(harmony_sources(project).chords[0], cancelled)
        return cqt, None, cqt, frame_seconds, duration
    chord_audio, bass_audio, key_audio = audio or harmony_audio(project, cancelled)
    chord_cqt = cqt_image(chord_audio, SAMPLE_RATE)
    bass_cqt = cqt_image(bass_audio, SAMPLE_RATE) if bass_audio is not None else None
    key_cqt = chord_cqt if key_audio is chord_audio else cqt_image(key_audio, SAMPLE_RATE)
    return chord_cqt, bass_cqt, key_cqt, HOP_LENGTH / SAMPLE_RATE, len(chord_audio) / SAMPLE_RATE


CHORD_MODELS = {"template": "Built-in templates", "btc": "BTC transformer (deep learning)"}


def analyze_harmony(project: Project, cancelled: Callable[[], bool] = lambda: False, model: str = "template"):
    """(key, chord segments) from the chord stems (or the mix).

    ``template`` ranks chord templates with the project key as a soft prior;
    ``btc`` runs the BTC transformer on the same audio (see audio_to_midi/btc.py).
    """
    from audio_to_midi.chords import recognize
    from audio_to_midi.spectrogram import LOW_PITCH, SAMPLE_RATE

    if model not in CHORD_MODELS:
        raise ValueError(f"Unknown chord model {model!r}")
    audio = harmony_audio(project, cancelled) if model == "btc" else None
    cqt, bass_cqt, key_cqt, frame_seconds, duration = harmony_cqt(project, cancelled, audio)
    key = project.key or detect_key(key_cqt)
    if model == "btc":
        from audio_to_midi import btc

        if SAMPLE_RATE != btc.SAMPLE_RATE:
            raise RuntimeError("BTC expects 22050 Hz audio")
        return key, btc.recognize(
            audio[0], project.grid, duration, bass_cqt, LOW_PITCH, frame_seconds, cancelled,
        )
    return key, recognize(
        cqt, LOW_PITCH, frame_seconds, project.grid, duration, cancel=cancelled, key=key, bass_cqt=bass_cqt,
    )
