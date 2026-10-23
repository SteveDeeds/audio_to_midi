"""Versioned, atomic project manifests, without backend or GUI dependencies.

Schema 3 stores every Fraction as {numerator, denominator} and chords as
lead-sheet segments. Relative media paths
are relative to the manifest folder; loading returns absolute media paths.
History is deliberately ephemeral, while edit feedback and revisions are durable.
"""

from dataclasses import asdict
from fractions import Fraction
import json
import os
from pathlib import Path
from uuid import uuid4

from .domain import ChordSegment, Note, NoteStore, Project, Track
from .timebase import Grid


SCHEMA_VERSION = 3


class ProjectStorageError(Exception):
    """Base class for explicit manifest loading failures."""


class InvalidProjectError(ProjectStorageError, ValueError):
    """The manifest is corrupt or contains invalid domain state."""


class SchemaVersionError(ProjectStorageError, ValueError):
    """The manifest uses an unsupported schema."""


class MissingMediaError(ProjectStorageError, FileNotFoundError):
    """A nonempty media reference does not name an existing file."""

    def __init__(self, path: Path):
        self.path = path
        super().__init__(f"Project media is missing: {path}")


class MissingProjectError(ProjectStorageError, FileNotFoundError):
    """There is no project.json in the requested folder."""


def _fraction(value: Fraction) -> dict:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _read_fraction(value: object) -> Fraction:
    value = _object(value, {"numerator", "denominator"}, "fraction")
    numerator, denominator = value["numerator"], value["denominator"]
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise InvalidProjectError("fraction requires integer numerator and positive denominator")
    return Fraction(numerator, denominator)


def _note(note: Note) -> dict:
    return {
        "id": note.id, "start": _fraction(note.start), "end": _fraction(note.end),
        "pitch": note.pitch, "velocity": note.velocity, "source": note.source,
    }


def _read_note(value: object) -> Note:
    value = _object(value, {"id", "start", "end", "pitch", "velocity", "source"}, "note")
    return Note(**{**value, "start": _read_fraction(value["start"]),
                   "end": _read_fraction(value["end"])})


def _chord(chord: ChordSegment) -> dict:
    return {
        "start": _fraction(chord.start), "end": _fraction(chord.end), "symbol": chord.symbol,
        "source": chord.source, "confidence": chord.confidence,
        "alternatives": list(chord.alternatives),
    }


def _read_chord(value: object) -> ChordSegment:
    value = _object(value, {"start", "end", "symbol", "source", "confidence", "alternatives"}, "chord")
    if type(value["symbol"]) is not str or any(type(a) is not str for a in _list(value["alternatives"], "alternatives")):
        raise InvalidProjectError("chord symbols must be strings")
    if type(value["confidence"]) not in (int, float):
        raise InvalidProjectError("chord confidence must be a number")
    return ChordSegment(
        _read_fraction(value["start"]), _read_fraction(value["end"]), value["symbol"],
        value["source"], float(value["confidence"]), tuple(value["alternatives"]),
    )


def _object(value: object, keys: set[str], name: str) -> dict:
    if type(value) is not dict or set(value) != keys:
        raise InvalidProjectError(f"invalid {name} fields; expected {sorted(keys)}")
    return value


def _list(value: object, name: str) -> list:
    if type(value) is not list:
        raise InvalidProjectError(f"{name} must be a list")
    return value


def _saved_path(value: str | None, folder: Path) -> str | None:
    if not value:
        return value
    path = Path(value)
    path = (folder / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        return str(path.relative_to(folder))
    except ValueError:
        return str(path)


def _loaded_path(value: object, folder: Path) -> str | None:
    if value is None or value == "":
        return value
    if not isinstance(value, str):
        raise InvalidProjectError("media paths must be strings or null")
    path = Path(value)
    path = (folder / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file():
        raise MissingMediaError(path)
    return str(path)


def _store(store: NoteStore) -> dict:
    return {
        "generated": [_note(n) for n in store.generated],
        "authored": [_note(n) for n in store.authored],
        "rejections": [
            {"start": _fraction(s), "end": _fraction(e), "pitch": p}
            for s, e, p in store.rejections
        ],
        "onsets": [_fraction(b) for b in store.onsets],
        "revision": store.revision,
    }


def _read_store(value: object) -> NoteStore:
    value = _object(value, {"generated", "authored", "rejections", "onsets", "revision"}, "note store")
    rejections = []
    for item in _list(value["rejections"], "rejections"):
        item = _object(item, {"start", "end", "pitch"}, "rejection")
        rejections.append((_read_fraction(item["start"]), _read_fraction(item["end"]), item["pitch"]))
    return NoteStore(
        generated=[_read_note(n) for n in _list(value["generated"], "generated")],
        authored=[_read_note(n) for n in _list(value["authored"], "authored")],
        rejections=rejections,
        onsets=[_read_fraction(b) for b in _list(value["onsets"], "onsets")],
        revision=value["revision"],
    )


def _manifest(project: Project, folder: Path) -> dict:
    grid = asdict(project.grid)
    grid["subdivision"] = _fraction(project.grid.subdivision)
    tracks = []
    for track in project.tracks:
        tracks.append({
            "id": track.id, "role": track.role,
            "audio_path": _saved_path(track.audio_path, folder),
            "duration": track.duration, "sample_rate": track.sample_rate,
            "model": track.model, "notes": _store(track.notes),
            "settings": track.settings, "training": track.training,
            "revision": track.revision,
        })
    return {
        "schema_version": SCHEMA_VERSION, "id": project.id, "name": project.name,
        "grid": grid, "tracks": tracks, "mix_path": _saved_path(project.mix_path, folder),
        "key": project.key, "chords": [_chord(c) for c in project.chords],
        "stages": project.stages, "revision": project.revision,
    }


def save_project(project: Project, folder: Path) -> None:
    """Fsync a same-folder staging file, then atomically replace project.json.

    Serialization, validation, write and replacement failures never modify the
    previous manifest. Save errors propagate to the caller; partial files are
    removed. Media is referenced, not copied, and is checked when reopening.
    """
    project.validate()
    folder = Path(folder).resolve()
    payload = json.dumps(_manifest(project, folder), ensure_ascii=False, allow_nan=False, indent=2)
    folder.mkdir(parents=True, exist_ok=True)
    staging = folder / f".project-{uuid4().hex}.json.pending"
    try:
        with staging.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, folder / "project.json")
    finally:
        if staging.exists():
            staging.unlink()


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidProjectError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise InvalidProjectError(f"non-finite JSON constant: {value}")


def load_project(folder: Path) -> Project:
    folder = Path(folder).resolve()
    try:
        with (folder / "project.json").open("r", encoding="utf-8") as stream:
            data = json.load(stream, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except FileNotFoundError as exc:
        raise MissingProjectError(f"Project manifest is missing: {folder / 'project.json'}") from exc
    except (ValueError, UnicodeError) as exc:
        raise InvalidProjectError(f"Cannot decode project manifest: {exc}") from exc
    if type(data) is not dict:
        raise InvalidProjectError("manifest must be an object")
    if type(data.get("schema_version")) is not int or data["schema_version"] != SCHEMA_VERSION:
        raise SchemaVersionError(f"Unsupported project schema: {data.get('schema_version')!r}")
    try:
        data = _object(data, {
            "schema_version", "id", "name", "grid", "tracks", "mix_path",
            "key", "chords", "stages", "revision",
        }, "project")
        grid = _object(data["grid"], {
            "bpm", "numerator", "denominator", "subdivision", "offset_seconds",
        }, "grid")
        grid = Grid(**{**grid, "subdivision": _read_fraction(grid["subdivision"])})
        tracks = []
        for value in _list(data["tracks"], "tracks"):
            value = _object(value, {
                "id", "role", "audio_path", "duration", "sample_rate", "model",
                "notes", "settings", "training", "revision",
            }, "track")
            tracks.append(Track(**{
                **value, "notes": _read_store(value["notes"]),
                "audio_path": _loaded_path(value["audio_path"], folder),
            }))
        return Project(
            id=data["id"], name=data["name"], grid=grid, tracks=tracks,
            mix_path=_loaded_path(data["mix_path"], folder), key=data["key"],
            chords=[_read_chord(c) for c in _list(data["chords"], "chords")],
            stages=data["stages"], revision=data["revision"],
        )
    except (TypeError, ValueError, KeyError, OverflowError) as exc:
        raise InvalidProjectError(f"Invalid project state: {exc}") from exc
