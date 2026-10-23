"""Optional, headless preparation (no UI or project-domain dependencies).

Public entry points: analyze_mix(full_mix, *, bpm, meter, downbeat_seconds,
beat_anchors), render_full_mix(full_mix, destination, warp_map, *, lead_in),
prepare_mix(full_mix, artifact_directory, ...), separate_mix(store, ...).
Beat anchors are (source seconds, quarter-note position); BPM is quarter notes
per minute, including compound meters. Librosa proposes beats, NOT downbeats.

Warping is an EXPERIMENTAL continuous, multichannel phase vocoder. It preserves
stationary pitch and interchannel phase relationships, but does not guarantee
transient quality or sample-accurate audible onset placement. Listening approval
is required before replacing externally warped material. Never pass external
stems to this API: only the original FULL MIX is rendered; separation consumes
the resulting normalized mix. There is deliberately no stem-warp API.

Artifacts use an independent preparation manifest, with immutable revisions.
One writer per artifact directory is supported. Downstream authoring integrations
must call mark_edited() before changing authoritative state outside this store.
Separation runs Demucs htdemucs_6s in a subprocess of the running interpreter
(audio_to_midi.separation_worker); the package and weights are fetched on first use.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid

import librosa
import numpy as np
import soundfile as sf


_MANIFEST_LOCK = threading.RLock()
_RUNNING_REVISIONS: set[str] = set()


class PreparationError(RuntimeError):
    """Preparation failed without publishing partial outputs."""


class Cancelled(PreparationError):
    pass


class RevisionConflict(PreparationError):
    pass


def _check_cancel(cancel):
    if cancel is not None and cancel():
        raise Cancelled("Preparation cancelled")


def decoded_frames(path) -> int:
    """Frames a decoder actually returns. MP3 headers only estimate the length
    (encoder delay/padding), so those are counted by decoding."""
    path = Path(path)
    stat = path.stat()
    return _decoded_frames(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


@functools.lru_cache(maxsize=16)
def _decoded_frames(path: str, _size: int, _mtime: int) -> int:
    with sf.SoundFile(path) as audio:
        if audio.format not in ("MP3", "MPEG"):
            return audio.frames
        total = 0
        while True:  # blocks() trusts the header length, so read until the decoder runs dry
            read = len(audio.read(1 << 16, dtype="float32"))
            if not read:
                return total
            total += read


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_json(path, value):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(_canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@dataclass(frozen=True)
class WarpMap:
    """Strict piecewise-linear seconds mapping; extrapolation is rejected."""

    source_seconds: tuple[float, ...]
    target_seconds: tuple[float, ...]

    def __post_init__(self):
        source = tuple(float(x) for x in self.source_seconds)
        target = tuple(float(x) for x in self.target_seconds)
        if len(source) < 2 or len(source) != len(target):
            raise ValueError("A warp requires at least two paired anchors")
        for values in (source, target):
            if not all(math.isfinite(x) for x in values) or any(
                b <= a for a, b in zip(values, values[1:])
            ):
                raise ValueError("Warp coordinates must be finite and strictly increasing")
        if source[0] < 0 or target[0] < 0:
            raise ValueError("Warp coordinates cannot be negative")
        object.__setattr__(self, "source_seconds", source)
        object.__setattr__(self, "target_seconds", target)

    def _convert(self, values, source, target):
        values = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < source[0]) or np.any(values > source[-1]):
            raise ValueError("Position outside warp map")
        result = np.interp(values, source, target)
        return float(result) if result.ndim == 0 else result

    def forward(self, source_seconds):
        return self._convert(source_seconds, self.source_seconds, self.target_seconds)

    def inverse(self, target_seconds):
        return self._convert(target_seconds, self.target_seconds, self.source_seconds)

    def to_dict(self):
        return {"source_seconds": self.source_seconds, "target_seconds": self.target_seconds}


def analyze_mix(
    full_mix, *, bpm=None, meter=None, downbeat_seconds=None, beat_anchors=(),
    cancel=None,
):
    """Inspect original mix; manual timing is optional and never inferred as fact.

    beat_anchors pairs source seconds with quarter notes relative to bar 1.
    Explicit downbeat is quarter note zero; conflicting anchors are rejected.
    Returns JSON-serializable analysis, candidates and confirmed timing.
    """
    _check_cancel(cancel)
    info = sf.info(full_mix)
    if info.frames < 2:
        raise ValueError("Mix must contain audio")
    if bpm is not None and (not math.isfinite(bpm) or bpm <= 0):
        raise ValueError("BPM must be positive and finite")
    if meter is not None:
        meter = tuple(meter)
        if len(meter) != 2 or any(type(x) is not int or x <= 0 for x in meter):
            raise ValueError("Meter must be positive integer numerator/denominator")
        if meter[1] & (meter[1] - 1):
            raise ValueError("Meter denominator must be a power of two")
    duration = info.frames / info.samplerate
    if downbeat_seconds is not None and (
        not math.isfinite(downbeat_seconds) or not 0 <= downbeat_seconds < duration
    ):
        raise ValueError("Downbeat must lie inside mix")
    anchors = [(float(s), float(q)) for s, q in beat_anchors]
    if downbeat_seconds is not None:
        existing = [q for s, q in anchors if s == downbeat_seconds]
        if existing and existing != [0.0]:
            raise ValueError("Downbeat conflicts with manual anchors")
        if not existing:
            anchors.append((float(downbeat_seconds), 0.0))
    anchors.sort()
    if any(not math.isfinite(s) or not math.isfinite(q) or not 0 <= s <= duration for s, q in anchors):
        raise ValueError("Manual anchors must be finite and inside mix")
    if any(s2 <= s1 or q2 <= q1 for (s1, q1), (s2, q2) in zip(anchors, anchors[1:])):
        raise ValueError("Manual beat anchors must be strictly increasing")
    audio, rate = librosa.load(full_mix, sr=22050, mono=True)
    _check_cancel(cancel)
    tempo, beats = librosa.beat.beat_track(y=audio, sr=rate, units="time")
    _check_cancel(cancel)
    return {
        "source": str(Path(full_mix).resolve()), "source_sha256": _hash(full_mix),
        "sample_rate": info.samplerate, "channels": info.channels,
        "duration_seconds": duration, "candidate_bpm": float(np.asarray(tempo).reshape(-1)[0]),
        "candidate_beats_seconds": np.asarray(beats).tolist(),
        "candidate_downbeats": None,
        "confirmed_bpm": bpm, "confirmed_meter": meter,
        "downbeat_seconds": downbeat_seconds, "beat_anchors": anchors,
        "backend": f"librosa-{librosa.__version__}:beat_track",
    }


def build_warp_map(analysis, *, lead_in):
    """Build a full-tail map. trim removes pre-downbeat; pad retains the lead-in.

    For pad, the rendered file starts at zero and bar one is at
    ``warp.forward(downbeat)`` (persisted as bar_one_seconds). Subtract that
    offset to obtain project seconds: retained lead-in is before project zero.
    No audio is silently discarded. Tail/lead extrapolation uses the nearest
    anchor slope. A downbeat alone cannot establish source tempo.
    """
    if lead_in not in ("trim", "pad"):
        raise ValueError("lead_in must explicitly be 'trim' or 'pad'")
    bpm, meter, downbeat = (
        analysis["confirmed_bpm"], analysis["confirmed_meter"], analysis["downbeat_seconds"]
    )
    if bpm is None or meter is None or downbeat is None:
        raise ValueError("Confirm BPM, meter and downbeat before rendering")
    duration = analysis["duration_seconds"]
    pairs = [(float(s), float(q) * 60 / bpm) for s, q in analysis["beat_anchors"]]
    if not any(s == downbeat and q == 0 for s, q in pairs):
        raise ValueError("Explicit downbeat anchor required")
    if lead_in == "trim":
        pairs = [(s, t) for s, t in pairs if s >= downbeat]
    if not pairs:
        raise ValueError("No usable anchors")
    if len(pairs) < 2:
        raise ValueError("Rendering requires at least two confirmed source beat anchors, not just a downbeat.")
    slope_first = (pairs[1][1] - pairs[0][1]) / (pairs[1][0] - pairs[0][0])
    slope_last = (pairs[-1][1] - pairs[-2][1]) / (pairs[-1][0] - pairs[-2][0])
    origin = downbeat if lead_in == "trim" else 0.0
    if pairs[0][0] > origin:
        pairs.insert(0, (origin, pairs[0][1] - (pairs[0][0] - origin) * slope_first))
    if pairs[-1][0] < duration:
        pairs.append((duration, pairs[-1][1] + (duration - pairs[-1][0]) * slope_last))
    offset = -pairs[0][1]
    return WarpMap(tuple(s for s, _ in pairs), tuple(t + offset for _, t in pairs))


def render_full_mix(
    full_mix, destination, warp_map, *, lead_in, cancel=None, progress=None,
    n_fft=2048, hop_length=512, max_working_bytes=512 * 1024 * 1024,
):
    """Render original full mix only using one continuous phase trajectory.

    Two-pass, bounded-memory offline renderer: inspect STFT energy for a stable
    shared per-bin reference, then seek source windows and stream overlap-add
    output. Phase is continuous across the entire song, not reset at anchors.
    max_working_bytes bounds conservatively reserved numerical workspace (not
    Python/library baseline RSS or caller-owned maps). It is independent of song
    duration. Transients still require listening approval. Destination is
    atomically replaced only on success.
    """
    from bisect import bisect_right

    if lead_in not in ("trim", "pad"):
        raise ValueError("Explicit trim/pad policy required")
    if Path(full_mix).resolve() == Path(destination).resolve():
        raise ValueError("Render destination must not overwrite the original mix.")
    if (type(n_fft) is not int or n_fft < 16 or n_fft % 2
            or type(hop_length) is not int or not 0 < hop_length <= n_fft // 2):
        raise ValueError("Invalid STFT configuration")
    _check_cancel(cancel)
    info = sf.info(full_mix)
    rate = info.samplerate
    duration = info.frames / rate
    workspace_bytes = 1024 * 1024 + 256 * n_fft * (info.channels + 1)
    if not math.isfinite(max_working_bytes) or workspace_bytes > max_working_bytes:
        raise PreparationError(f"Renderer needs at least {workspace_bytes} working bytes")
    if abs(warp_map.source_seconds[-1] - duration) > 1 / rate:
        raise ValueError("Warp must include entire full-mix tail")
    if warp_map.target_seconds[0] != 0 or (lead_in == "pad" and warp_map.source_seconds[0] != 0):
        raise ValueError("Warp origin conflicts with lead-in policy")
    bins = np.arange(n_fft // 2 + 1)
    window = np.hanning(n_fft + 1)[:-1]
    window_squared = window ** 2
    advance = 2 * np.pi * hop_length * bins / n_fft
    output_length = round(warp_map.target_seconds[-1] * rate)
    frames = math.ceil((output_length + n_fft // 2) / hop_length)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(suffix=".wav", prefix=".render-", dir=destination.parent)
    os.close(fd)
    try:
        with sf.SoundFile(full_mix) as source, sf.SoundFile(
            name, mode="w", samplerate=rate, channels=info.channels, subtype="FLOAT",
        ) as rendered:
            def spectrum_at(index):
                # Match centered constant-padded STFT boundary semantics.
                samples = np.zeros((info.channels, n_fft), dtype=np.float64)
                if index * hop_length - n_fft // 2 < info.frames:
                    start = index * hop_length - n_fft // 2
                    begin, end = max(0, start), min(info.frames, start + n_fft)
                    if end > begin:
                        source.seek(begin)
                        block = source.read(end - begin, dtype="float64", always_2d=True)
                        if not np.all(np.isfinite(block)):
                            raise ValueError("Mix contains non-finite samples")
                        samples[:, begin - start:end - start] = block.T
                samples *= window
                return np.fft.rfft(samples, axis=1)

            energies = np.zeros((info.channels, len(bins)))
            source_frames = info.frames // hop_length + 1
            for index in range(source_frames):
                if index % 32 == 0:
                    _check_cancel(cancel)
                    if progress:
                        progress(0.25 * index / source_frames)
                spectrum = spectrum_at(index)
                energies += spectrum.real ** 2 + spectrum.imag ** 2
            reference_channels = np.argmax(energies, axis=0)
            del energies, spectrum
            phase = None
            overlap = np.zeros((info.channels, n_fft))
            normalization = np.zeros(n_fft)
            cached = {}
            for index in range(frames):
                if index % 32 == 0:
                    _check_cancel(cancel)
                    if progress:
                        progress(0.25 + 0.75 * index / frames)
                target = index * hop_length / rate
                if target <= warp_map.target_seconds[-1]:
                    # Scalar binary search avoids converting whole anchor
                    # tuples to arrays for every output frame.
                    segment = max(0, min(
                        bisect_right(warp_map.target_seconds, target) - 1,
                        len(warp_map.target_seconds) - 2,
                    ))
                    source_a, source_b = warp_map.source_seconds[segment:segment + 2]
                    target_a, target_b = warp_map.target_seconds[segment:segment + 2]
                    source_seconds = source_a + (target - target_a) * (
                        source_b - source_a
                    ) / (target_b - target_a)
                    position = source_seconds * rate / hop_length
                else:
                    # Complete centered windows past the final output sample,
                    # extrapolating only their zero-padded boundary context.
                    slope = ((warp_map.source_seconds[-1] - warp_map.source_seconds[-2])
                             / (warp_map.target_seconds[-1] - warp_map.target_seconds[-2]))
                    position = (warp_map.source_seconds[-1]
                                + (target - warp_map.target_seconds[-1]) * slope) * rate / hop_length
                left = int(position)
                fraction = position - left
                # At most two source spectra survive each iteration, regardless
                # of local stretch ratio or total duration.
                cached = {k: v for k, v in cached.items() if k in (left, left + 1)}
                for frame in (left, left + 1):
                    if frame not in cached:
                        cached[frame] = spectrum_at(frame)
                a, b = cached[left], cached[left + 1]
                ra, rb = a[reference_channels, bins], b[reference_channels, bins]
                if phase is None:
                    phase = np.angle(ra) + fraction * np.angle(rb * ra.conj())
                magnitude = (1 - fraction) * abs(a) + fraction * abs(b)
                relative_a = np.angle(a * ra.conj())
                relative_b = np.angle(b * rb.conj())
                relative = relative_a + fraction * np.angle(np.exp(1j * (relative_b - relative_a)))
                synthesized = magnitude * np.exp(1j * (phase + relative))
                residual = np.angle(rb) - np.angle(ra) - advance
                phase += advance + residual - 2 * np.pi * np.round(residual / (2 * np.pi))
                phase %= 2 * np.pi
                overlap += np.fft.irfft(synthesized, n=n_fft, axis=1) * window
                normalization += window_squared
                # Future windows start one hop later: these samples are final.
                start = index * hop_length - n_fft // 2
                lo, hi = max(0, -start), min(hop_length, output_length - start)
                if hi > lo:
                    denominator = normalization[lo:hi]
                    block = np.divide(
                        overlap[:, lo:hi], denominator,
                        out=np.zeros((info.channels, hi - lo)), where=denominator > 1e-15,
                    )
                    rendered.write(block.T)
                overlap[:, :-hop_length] = overlap[:, hop_length:]
                overlap[:, -hop_length:] = 0
                normalization[:-hop_length] = normalization[hop_length:]
                normalization[-hop_length:] = 0
            # Remaining OLA tail has no future contributing windows.
            start = frames * hop_length - n_fft // 2
            lo, hi = max(0, -start), min(n_fft, output_length - start)
            if hi > lo:
                rendered.write(np.divide(
                    overlap[:, lo:hi], normalization[lo:hi],
                    out=np.zeros((info.channels, hi - lo)),
                    where=normalization[lo:hi] > 1e-15,
                ).T)
            if rendered.frames != output_length:
                raise PreparationError("Streaming renderer produced an incomplete output")
        _check_cancel(cancel)
        os.replace(name, destination)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    if progress:
        progress(1.0)
    return {"sample_rate": rate, "frames": output_length, "channels": info.channels,
            "quality": "experimental", "lead_in": lead_in,
            "renderer": "streaming-shared-phase-v2",
            "working_bytes_reserved": workspace_bytes,
            "removed_lead_in_seconds": warp_map.source_seconds[0],
            "warp_map": warp_map.to_dict()}


class ArtifactStore:
    """Durable revision DAG; builders write only inside their staging directory."""

    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "preparation.json"
        with _MANIFEST_LOCK:
            self.manifest = self._read()
            changed = False
            for revision in self.manifest["revisions"]:
                if revision["status"] == "running" and revision["id"] not in _RUNNING_REVISIONS:
                    revision["status"] = "interrupted"
                    changed = True
            if changed:
                self._save()

    def _read(self):
        manifest = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {
            "schema_version": 1, "revisions": [], "active": {},
        }
        if manifest["schema_version"] != 1:
            raise PreparationError("Unsupported preparation manifest schema")
        return manifest

    def _save(self):
        with _MANIFEST_LOCK:
            # Authoring may protect a dependency while this store is building an output.
            edited = {r["id"] for r in self._read()["revisions"] if r.get("edited")}
            for revision in self.manifest["revisions"]:
                if revision["id"] in edited:
                    revision["edited"] = True
            _atomic_json(self.path, self.manifest)

    def revision(self, revision_id):
        return next(r for r in self.manifest["revisions"] if r["id"] == revision_id)

    def valid(self, revision):
        return revision["status"] == "done" and bool(revision["outputs"]) and all(
            (self.root / item["path"]).is_file() and _hash(self.root / item["path"]) == item["sha256"]
            for item in revision["outputs"]
        )

    def output(self, revision, name):
        return self.root / next(x["path"] for x in revision["outputs"] if x["name"] == name)

    def mark_edited(self, revision_id):
        with _MANIFEST_LOCK:
            manifest = self._read()
            next(r for r in manifest["revisions"] if r["id"] == revision_id)["edited"] = True
            _atomic_json(self.path, manifest)
            for revision in self.manifest["revisions"]:
                if revision["id"] == revision_id:
                    revision["edited"] = True

    def _descendants(self, revision_id):
        found = {revision_id}
        while True:
            expanded = found | {r["id"] for r in self.manifest["revisions"] if found.intersection(r["dependencies"])}
            if expanded == found:
                return found
            found = expanded

    def run(self, stage, *, inputs, parameters, backend, builder, dependencies=(),
            force_new_revision=False, cancel=None):
        """builder(staging_path) -> (name-to-relative-path mapping, metadata).

        force_new_revision preserves all old media and marks descendants stale.
        Hash changes to downstream media count as edits even without mark_edited.
        """
        _check_cancel(cancel)
        dependencies = list(dependencies)
        for dependency in dependencies:
            revision = self.revision(dependency)
            if not self.valid(revision) or revision.get("stale"):
                raise RevisionConflict("Dependency is missing, edited, or stale")
        identities = {str(Path(p).resolve()): _hash(p) for p in inputs}
        key = hashlib.sha256(_canonical({
            "inputs": identities, "parameters": parameters, "backend": backend,
            "dependencies": dependencies,
        }).encode()).hexdigest()
        old_id = self.manifest["active"].get(stage)
        old = self.revision(old_id) if old_id else None
        if old and old["key"] == key and self.valid(old) and not old.get("stale") and not force_new_revision:
            return old
        affected = self._descendants(old_id) if old else set()
        for r in self.manifest["revisions"]:
            if r["id"] in affected and (r.get("edited") or (r["status"] == "done" and not self.valid(r))):
                if not force_new_revision:
                    raise RevisionConflict("Rerun affects edited artifacts; explicitly force a new revision")
        revision = {
            "id": uuid.uuid4().hex, "stage": stage, "key": key, "inputs": identities,
            "parameters": parameters, "backend": backend, "dependencies": dependencies,
            "status": "running", "outputs": [], "edited": False, "stale": False,
        }
        self.manifest["revisions"].append(revision)
        published = self.root / revision["id"]
        try:
            with _MANIFEST_LOCK:
                _RUNNING_REVISIONS.add(revision["id"])
                self._save()
            with tempfile.TemporaryDirectory(prefix=".stage-", dir=self.root) as directory:
                staging = Path(directory)
                outputs, metadata = builder(staging)
                _check_cancel(cancel)
                if not outputs:
                    raise PreparationError("Stage produced no outputs")
                items = []
                for name, relative in outputs.items():
                    path = (staging / relative).resolve()
                    if not path.is_relative_to(staging.resolve()) or not path.is_file():
                        raise PreparationError("Invalid stage output path")
                    items.append({"name": name, "path": str(Path(revision["id"]) / path.relative_to(staging)),
                                  "sha256": _hash(path)})
                # Prevent a changed source/dependency from being published as a valid cache.
                if any(_hash(p) != digest for p, digest in identities.items()) or any(
                    not self.valid(self.revision(dep)) for dep in dependencies
                ):
                    raise RevisionConflict("Inputs changed during preparation")
                _canonical(metadata)
                os.replace(staging, published)
            revision.update(status="done", outputs=items, metadata=metadata)
            for r in self.manifest["revisions"]:
                if r["id"] in affected:
                    r["stale"] = True
            self.manifest["active"][stage] = revision["id"]
            self._save()
            return revision
        except BaseException as error:
            revision["status"] = "cancelled" if isinstance(error, Cancelled) else "failed"
            revision["error"] = str(error)
            self._save()
            raise
        finally:
            with _MANIFEST_LOCK:
                _RUNNING_REVISIONS.discard(revision["id"])


def prepare_mix(
    full_mix, artifact_directory, *, bpm=None, meter=None, downbeat_seconds=None,
    beat_anchors=(), lead_in=None, force_new_revision=False, cancel=None, progress=None,
):
    """Persist analysis; optionally render if lead_in is explicitly trim/pad.

    Returns {store, analysis_revision, warp_revision}; revisions are manifest
    records. Calling without confirmed timing is a pure inspect/candidate stage.
    """
    store = ArtifactStore(artifact_directory)
    anchors = list(beat_anchors)

    def inspect(staging):
        analysis = analyze_mix(full_mix, bpm=bpm, meter=meter,
                               downbeat_seconds=downbeat_seconds, beat_anchors=anchors, cancel=cancel)
        _atomic_json(staging / "analysis.json", analysis)
        return {"analysis": "analysis.json"}, analysis

    analysis = store.run(
        "analysis", inputs=[full_mix],
        parameters={"bpm": bpm, "meter": meter, "downbeat_seconds": downbeat_seconds, "anchors": anchors},
        backend=f"librosa-{librosa.__version__}:beat_track-v1", builder=inspect,
        force_new_revision=force_new_revision, cancel=cancel,
    )
    warp = None
    if lead_in is not None:
        data = json.loads(store.output(analysis, "analysis").read_text(encoding="utf-8"))
        mapping = build_warp_map(data, lead_in=lead_in)

        def render(staging):
            metadata = render_full_mix(full_mix, staging / "mix.wav", mapping, lead_in=lead_in,
                                       cancel=cancel, progress=progress)
            metadata["bar_one_seconds"] = mapping.forward(data["downbeat_seconds"])
            _atomic_json(staging / "warp.json", metadata)
            return {"full_mix": "mix.wav", "mapping": "warp.json"}, metadata

        warp = store.run(
            "warp", inputs=[full_mix], parameters={"map": mapping.to_dict(), "lead_in": lead_in},
            backend=f"experimental-streaming-shared-phase-v2:librosa-{librosa.__version__}",
            builder=render, dependencies=[analysis["id"]], force_new_revision=force_new_revision,
            cancel=cancel,
        )
    return {"store": store, "analysis_revision": analysis, "warp_revision": warp}


def register_warped_mix(
    full_mix, artifact_directory, *, bpm, meter, offset_seconds=0,
    force_new_revision=False, cancel=None,
):
    """Register external normalized audio verbatim; no analysis or re-warping."""
    from audio_to_midi.timebase import Grid

    if len(meter) != 2:
        raise ValueError("Meter requires a numerator and denominator.")
    grid = Grid(bpm=bpm, numerator=meter[0], denominator=meter[1], offset_seconds=offset_seconds)
    info = sf.info(full_mix)
    if info.frames <= 0 or offset_seconds >= info.duration:
        raise ValueError("External mix must contain audio after its declared downbeat.")
    store = ArtifactStore(artifact_directory)
    data = {
        "source": str(Path(full_mix).resolve()), "confirmed_bpm": grid.bpm,
        "confirmed_meter": list(meter), "downbeat_seconds": offset_seconds,
        "duration_seconds": info.duration, "provenance": "external-warp",
        "warp_map": None,
    }

    def metadata(staging):
        _atomic_json(staging / "analysis.json", data)
        return {"analysis": "analysis.json"}, data

    analysis = store.run(
        "analysis", inputs=[full_mix], parameters=data, backend="external-timing-v1",
        builder=metadata, force_new_revision=force_new_revision, cancel=cancel,
    )

    def copy(staging):
        name = "mix" + Path(full_mix).suffix
        shutil.copyfile(full_mix, staging / name)
        _check_cancel(cancel)
        return {"full_mix": name}, {
            "bar_one_seconds": offset_seconds, "warp_map": None,
            "quality": "externally-warped", "provenance": "external-warp",
        }

    warp = store.run(
        "warp", inputs=[full_mix], parameters=data, backend="external-verbatim-v1",
        builder=copy, dependencies=[analysis["id"]],
        force_new_revision=force_new_revision, cancel=cancel,
    )
    return {"store": store, "analysis_revision": analysis, "warp_revision": warp}


def _run_process(argv, *, cancel=None, progress=None, timeout_seconds=None, name="Demucs", env=None):
    """Drain subprocess output without blocking cancellation; never invoke a shell."""
    _check_cancel(cancel)
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                               text=True, encoding="utf-8", errors="replace", bufsize=1,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    lines = queue.Queue(maxsize=256)

    def drain():
        for line in process.stdout:
            try:
                lines.put_nowait(line.rstrip())
            except queue.Full:
                pass

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    started = time.monotonic()
    tail = []
    try:
        while True:
            _check_cancel(cancel)
            if timeout_seconds is not None and time.monotonic() - started > timeout_seconds:
                raise PreparationError(f"{name} timed out")
            try:
                line = lines.get(timeout=0.05)
                tail = (tail + [line])[-20:]
                if progress:
                    progress(line)
            except queue.Empty:
                pass
            if process.poll() is not None and not reader.is_alive() and lines.empty():
                break
        if process.returncode:
            raise PreparationError(f"{name} failed (exit {process.returncode}):\n" + "\n".join(tail))
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        reader.join(timeout=5)
        process.stdout.close()


def separate_mix(
    store, warp_revision_id, *, force_new_revision=False, cancel=None,
    progress=None, timeout_seconds=None,
):
    """htdemucs_6s on the warp revision's full mix, consuming ONLY a warp revision.

    Demucs (MIT code and weights) is pip-installed and its weights downloaded on
    first use; the worker runs in a subprocess so cancellation can stop it.
    """
    from audio_to_midi import provision

    provision.ensure("demucs", progress, cancel)
    revision = store.revision(warp_revision_id)
    if revision["stage"] != "warp" or not store.valid(revision) or revision.get("stale"):
        raise RevisionConflict("Separation requires a valid normalized full-mix warp revision")
    from importlib import metadata

    version = metadata.version("demucs")
    full_mix = store.output(revision, "full_mix")
    roles = ("drums", "bass", "other", "vocals", "guitar", "piano")

    def separate(staging):
        package_root = str(Path(__file__).resolve().parents[1])
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [package_root, os.environ.get("PYTHONPATH")]))}
        _run_process(
            [sys.executable, "-X", "utf8", "-m", "audio_to_midi.separation_worker",
             str(full_mix), str(staging / "demucs")],
            cancel=cancel, progress=progress, timeout_seconds=timeout_seconds, env=env,
        )
        expected = sf.info(full_mix)
        expected_frames = decoded_frames(full_mix)
        result = {}
        for role in roles:
            _check_cancel(cancel)
            matches = list((staging / "demucs").rglob(role + ".wav"))
            if len(matches) != 1:
                raise PreparationError(f"Demucs missing/ambiguous {role} stem")
            info = sf.info(matches[0])
            if info.samplerate != expected.samplerate or info.frames != expected_frames or info.channels != expected.channels:
                raise PreparationError(f"Demucs {role} rate/duration/channel mismatch; no silent resampling")
            result[role] = str(matches[0].relative_to(staging))
        return result, {"model": "htdemucs_6s", "roles": roles, "alignment": "shared-warp",
                        "audition_required": True}

    return store.run(
        "separation", inputs=[full_mix],
        parameters={"model": "htdemucs_6s", "float32": True},
        backend=f"demucs-{version}", builder=separate, dependencies=[warp_revision_id],
        force_new_revision=force_new_revision, cancel=cancel,
    )
