"""YourMT3+ evidence through a cancellable, isolated inference subprocess."""

import os
from pathlib import Path
import sys
import tempfile
import json

import numpy as np

from audio_to_midi.evidence_cache import EvidenceCache, checksum
from audio_to_midi.pipeline import _run_process

CHECKPOINT_SPEC = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops"
CHECKPOINT_SHA256 = "ae38e415c79efd5592dcb9b658cdb99ddb11d4c4e1eaa364cab04a052473fc25"
CHECKPOINT = Path.home() / ".cache" / "mt3-infer" / "yourmt3" / CHECKPOINT_SPEC / "last.ckpt"
CHECKPOINT_URL = (
    "https://huggingface.co/spaces/mimbres/YourMT3/resolve/main/amt/logs/2024/"
    + CHECKPOINT_SPEC + "/checkpoints/last.ckpt"
)
ENVIRONMENT = Path.home() / ".audio_to_midi" / "yourmt3"


def interpreter() -> str:
    override = os.environ.get("ATM_YOURMT3_PYTHON")
    if override:
        return str(Path(override).resolve(strict=True))
    dedicated = ENVIRONMENT / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return str(dedicated) if dedicated.is_file() else sys.executable


def checkpoint() -> Path:
    return Path(os.environ.get("ATM_YOURMT3_CHECKPOINT", str(CHECKPOINT)))


def worker(command: str, out: Path, cancelled, report, audio: str | None = None):
    env = os.environ.copy()
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    args = [interpreter(), "-m", "audio_to_midi.yourmt3_worker", command, str(out), str(checkpoint())]
    if audio:
        args.append(audio)
    _run_process(args, cancel=cancelled, progress=report, name="YourMT3+", env=env)


def validate_notes(arrays: dict):
    notes = arrays["notes"]
    if notes.ndim != 2 or notes.shape[1] != 6 or not np.isfinite(notes).all():
        raise ValueError("Invalid YourMT3+ note matrix.")
    if len(notes) and (
        np.any(notes[:, 0] < 0) or np.any(notes[:, 1] <= notes[:, 0])
        or np.any((notes[:, 2:5] < 0) | (notes[:, 2:5] > 127))
        or np.any(notes[:, 3] < 1) or np.any(notes[:, 2:6] != np.round(notes[:, 2:6]))
        or np.any((notes[:, 5] != 0) & (notes[:, 5] != 1))
    ):
        raise ValueError("Invalid YourMT3+ note fields.")


def evidence(path: str, source: str, cancelled=lambda: False, report=lambda text: None,
             cache: EvidenceCache | None = None) -> dict:
    store = cache or EvidenceCache()
    before = checksum(Path(path))
    with tempfile.TemporaryDirectory(prefix="atm-yourmt3-") as directory:
        root = Path(directory)
        fingerprint = root / "model.json"
        worker("identify", fingerprint, cancelled, lambda text: None)
        identity = json.loads(fingerprint.read_text(encoding="utf-8"))
        descriptor = store.descriptor(
            dict(sha256=before), source, identity,
            dict(sr=16000, mono=True, adaptive=False, batch_size=8, decoding_revision=1,
                 device=os.environ.get("ATM_DEVICE", "auto")))

        def compute():
            out = root / "notes.npz"
            report(f"YourMT3+ queued ({source}); waiting for exclusive GPU access if needed.")
            worker("infer", out, cancelled, report, path)
            if checksum(Path(path)) != before:
                raise RuntimeError("Audio changed during YourMT3+ inference; output was not cached.")
            with np.load(out, allow_pickle=False) as archive:
                return dict(notes=archive["notes"])

        return store.get(descriptor, compute, validate_notes, cancelled)


def select_notes(arrays: dict, role: str, source: str, parameters: dict) -> np.ndarray:
    if parameters:
        raise ValueError("YourMT3+ has no retained note confidence/logits; threshold replay is unavailable.")
    notes = arrays["notes"]
    if role == "drums":
        return notes[notes[:, 5] == 1]
    pitched = notes[notes[:, 5] == 0]
    if source == "stem":
        return pitched
    programs = {"piano": (0, 7), "guitar": (24, 31), "bass": (32, 39)}
    lo, hi = programs[role]
    return pitched[(pitched[:, 4] >= lo) & (pitched[:, 4] <= hi)]
