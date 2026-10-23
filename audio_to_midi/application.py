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
    track.settings["warp"] = current_warp(project)
    project.tracks.append(track)
    project.revision += 1
    return track


def _stem_info(project: Project, path: Path) -> sf._SoundFileInfo:
    info = sf.info(path)
    if info.frames <= 0:
        raise ValueError(f"Empty audio file: {path}")
    if project.grid.offset_seconds >= info.duration:
        raise ValueError("Declared downbeat lies beyond this audio.")
    reference = sf.info(project.mix_path) if project.mix_path else (
        sf.info(project.tracks[0].audio_path) if project.tracks else None
    )
    if reference and (
        reference.samplerate != info.samplerate
        or abs(reference.duration - info.duration) > 1 / reference.samplerate
    ):
        raise ValueError(
            "Stem sample rate/duration differs from the project reference. "
            "Explicitly resample/pad externally, then reimport and audition alignment."
        )
    return info


def current_warp(project: Project) -> str | None:
    preparation = project.stages.get("preparation")
    return preparation.get("warp_revision") if preparation else None


def stale(project: Project, track: Track) -> bool:
    """True when the stem was separated/imported against an older warp of the mix."""
    return track.settings.get("warp", current_warp(project)) != current_warp(project)


def replace_stem(project: Project, track: Track, path: Path) -> None:
    """Swap a track's audio (e.g. re-separated stem); authored notes keep their bar/beat positions."""
    path = path.resolve(strict=True)
    info = _stem_info(project, path)
    track.audio_path = str(path)
    track.duration = info.duration
    track.sample_rate = info.samplerate
    track.notes.generated = []
    track.notes.revision += 1
    track.settings["warp"] = current_warp(project)
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
        for name in ("warp_revision", "separation_revision"):
            if preparation.get(name):
                store.mark_edited(preparation[name])


def attach_prepared(project: Project, artifacts: Path) -> None:
    from audio_to_midi.pipeline import ArtifactStore

    if project.tracks or project.chords:
        raise ValueError("Attach preparation to an empty project; existing edits must remain in their original revision.")
    store = ArtifactStore(artifacts)
    warp_id = store.manifest["active"].get("warp")
    if not warp_id:
        raise ValueError("No completed warp revision.")
    adopt_warp(project, store, warp_id)


def adopt_warp(project: Project, store, warp_id: str) -> None:
    """Make a warp revision the project mix and grid (bar 1 from the warp metadata)."""
    from audio_to_midi.timebase import Grid

    warp = store.revision(warp_id)
    if not store.valid(warp) or warp.get("stale"):
        raise ValueError("Warp revision is invalid or stale.")
    analysis = store.revision(warp["dependencies"][0])
    data = json.loads(store.output(analysis, "analysis").read_text(encoding="utf-8"))
    numerator, denominator = data["confirmed_meter"]
    project.grid = Grid(
        data["confirmed_bpm"], numerator, denominator, project.grid.subdivision,
        warp["metadata"]["bar_one_seconds"],
    )
    project.mix_path = str(store.output(warp, "full_mix"))
    project.stages["preparation"] = {
        "directory": str(store.root), "warp_revision": warp_id,
        "analysis_revision": analysis["id"],
    }
    project.revision += 1


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
    from core.analysis import detect_key_from_chroma
    from audio_to_midi.chords import fold_chroma
    from audio_to_midi.spectrogram import LOW_PITCH

    chroma = fold_chroma(cqt ** 4, LOW_PITCH)
    if not chroma.any():
        raise ValueError("No pitched harmony is audible in the mix.")
    return normalize_key(detect_key_from_chroma(chroma.mean(axis=1)))


def analyze_harmony(project: Project, cancelled: Callable[[], bool] = lambda: False):
    """(key, chord segments) from the mix with the placeholder recognizer."""
    from audio_to_midi.chords import recognize
    from audio_to_midi.spectrogram import LOW_PITCH

    if not project.mix_path:
        raise ValueError("Import a mix before detecting chords.")
    cqt, frame_seconds, duration = mix_cqt(project.mix_path, cancelled)
    key = project.key or detect_key(cqt)
    return key, recognize(cqt, LOW_PITCH, frame_seconds, project.grid, duration, cancel=cancelled)
