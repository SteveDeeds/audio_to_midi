"""Disposable inference evidence, separate from projects and benchmark artifacts."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np

SCHEMA = 1


class CacheError(RuntimeError):
    pass


def checksum(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def array_identity(audio: np.ndarray, sr: int) -> dict:
    value = np.ascontiguousarray(audio)
    return dict(sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                shape=list(value.shape), dtype=str(value.dtype), sample_rate=sr)


@contextmanager
def file_lock(name: str, cancelled: Callable[[], bool] = lambda: False, *, wait: bool = True):
    """OS-owned locks survive cache deletion and are released if a worker dies."""
    root = Path(tempfile.gettempdir()) / "audio-to-midi-locks"
    root.mkdir(parents=True, exist_ok=True)
    path = root / (hashlib.sha256(name.encode()).hexdigest() + ".lock")
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        acquired = False
        try:
            while not acquired:
                if cancelled():
                    from audio_to_midi.adapters import Cancelled
                    raise Cancelled()
                stream.seek(0)
                try:
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except OSError as exc:
                    if exc.errno not in (11, 13, 35, 36):
                        raise
                    if not wait:
                        break
                    time.sleep(.05)
            yield acquired
        finally:
            if acquired:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


class EvidenceCache:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or os.environ.get(
            "ATM_CACHE_DIR", str(Path(__file__).resolve().parents[1] / "cache")))

    def descriptor(self, audio: dict, source: str, model: dict, parameters: dict) -> dict:
        if source != "mix" and not source.startswith("stem:"):
            raise ValueError("Evidence source must be mix or stem:<role>.")
        return dict(schema=SCHEMA, audio=audio, source=source, model=model, parameters=parameters)

    def path(self, descriptor: dict) -> Path:
        key = hashlib.sha256(canonical(descriptor).encode()).hexdigest()
        return self.root / "inference" / f"v{SCHEMA}" / key[:2] / f"{key}.npz"

    def load(self, descriptor: dict, validate: Callable[[dict], None]) -> dict | None:
        path = self.path(descriptor)
        try:
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"].item()))
                arrays = {k: archive[k] for k in archive.files if k != "metadata"}
                if metadata["descriptor"] != descriptor:
                    raise ValueError("Descriptor mismatch")
                expected = {k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=_array_hash(v))
                            for k, v in arrays.items()}
                if expected != metadata["arrays"]:
                    raise ValueError("Array checksum/schema mismatch")
                validate(arrays)
                return arrays
        except FileNotFoundError:
            return None
        except (ValueError, KeyError, TypeError, EOFError, OSError, zipfile.BadZipFile) as exc:
            raise CacheError(f"Invalid inference cache {path}: {exc}. Delete this cache entry to regenerate it.") from exc

    def save(self, descriptor: dict, arrays: dict, validate: Callable[[dict], None]) -> None:
        validate(arrays)
        metadata = dict(descriptor=descriptor, arrays={
            k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=_array_hash(v))
            for k, v in arrays.items()})
        path = self.path(descriptor)
        # Deleting the disposable folder during a write is a normal cache miss.
        for attempt in range(2):
            temporary = None
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary = tempfile.mkstemp(prefix=".evidence-", suffix=".npz", dir=path.parent)
                with os.fdopen(fd, "wb") as stream:
                    np.savez_compressed(stream, metadata=np.asarray(canonical(metadata)), **arrays)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                return
            except FileNotFoundError:
                if attempt:
                    return
            finally:
                if temporary:
                    Path(temporary).unlink(missing_ok=True)

    def get(self, descriptor: dict, compute: Callable[[], dict], validate: Callable[[dict], None],
            cancelled: Callable[[], bool] = lambda: False) -> dict:
        with file_lock(str(self.path(descriptor).resolve()), cancelled):
            arrays = self.load(descriptor, validate)
            if arrays is not None:
                return arrays
            arrays = compute()
            if cancelled():
                from audio_to_midi.adapters import Cancelled
                raise Cancelled()
            self.save(descriptor, arrays, validate)
            return arrays
