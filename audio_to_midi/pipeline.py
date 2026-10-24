"""Optional, headless preparation (no UI or project-domain dependencies).

The mix is never stretched: separation runs on the original full mix, so stems
share its timeline and the project's tempo map applies to all of them.

Artifacts use an independent preparation manifest, with immutable revisions.
One writer per artifact directory is supported. Downstream authoring integrations
must call mark_edited() before changing authoritative state outside this store.
Separation runs Demucs htdemucs_6s in a subprocess of the running interpreter
(audio_to_midi.separation_worker); the package and weights are fetched on first use.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
import uuid

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
    store, full_mix, *, force_new_revision=False, cancel=None,
    progress=None, timeout_seconds=None,
):
    """htdemucs_6s on the original full mix.

    Demucs (MIT code and weights) is pip-installed and its weights downloaded on
    first use; the worker runs in a subprocess so cancellation can stop it.
    """
    from audio_to_midi import provision

    provision.ensure("demucs", progress, cancel)
    from importlib import metadata

    version = metadata.version("demucs")
    full_mix = Path(full_mix)
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
        return result, {"model": "htdemucs_6s", "roles": roles, "alignment": "original-mix",
                        "audition_required": True}

    return store.run(
        "separation", inputs=[full_mix],
        parameters={"model": "htdemucs_6s", "float32": True},
        backend=f"demucs-{version}", builder=separate, dependencies=[],
        force_new_revision=force_new_revision, cancel=cancel,
    )
