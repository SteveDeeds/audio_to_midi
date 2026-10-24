"""First-use installation of optional model packages and checkpoints, into the running Python.

Nothing here asks the user where software lives: packages are pip-installed into
``sys.executable``'s environment and weights go to each library's usual cache.
"""

from __future__ import annotations

from importlib import invalidate_caches, util
import os
from pathlib import Path
import sys
import urllib.request

from audio_to_midi.pipeline import Cancelled, PreparationError, _run_process
from audio_to_midi.environment import PIANO_HR_CHECKPOINT

# (import name, pip arguments) per feature; installed in order, skipped when importable.
_CREPE = [("torch", ["torch"]), ("torchcrepe", ["torchcrepe>=0.0.23"])]
FEATURES: dict[str, list[tuple[str, list[str]]]] = {
    "pyin": [],
    "crepe_tiny": _CREPE,
    "crepe_full": _CREPE,
    # basic-pitch 0.4.0 declares TensorFlow, which has no Python 3.12 wheel; it runs on ONNX here.
    "basic_pitch": [
        ("onnxruntime", ["onnxruntime>=1.16"]), ("resampy", ["resampy>=0.4,<0.4.3"]),
        ("mir_eval", ["mir_eval>=0.7"]), ("sklearn", ["scikit-learn>=1.3"]),
        ("pretty_midi", ["pretty_midi>=0.2.9"]), ("basic_pitch", ["--no-deps", "basic-pitch==0.4.0"]),
    ],
    "piano_hr": [("torch", ["torch"]), ("piano_transcription_inference", ["piano-transcription-inference>=0.0.6"])],
    "adtof": [("torch", ["torch"]), ("adtof_pytorch", [
        "adtof-pytorch @ git+https://github.com/xavriley/ADTOF-pytorch@85c192e78f716ea0b111cc8a5ee4a8f6a3a4f8a9",
    ])],
    "demucs": [("torch", ["torch"]), ("demucs", ["demucs>=4.0.1"])],
    "btc": [("torch", ["torch"])],
}
LABELS = {
    "crepe_tiny": "CREPE", "crepe_full": "CREPE", "basic_pitch": "Basic Pitch", "piano_hr": "Piano-HR",
    "adtof": "ADTOF", "demucs": "Demucs", "btc": "BTC chord model",
}
PIANO_HR_URL = "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
PIANO_HR_MIN_BYTES = 1.6e8  # the library re-downloads anything smaller


def _installed(module: str) -> bool:
    return util.find_spec(module) is not None


def _requirements(feature: str) -> list[tuple[str, list[str]]]:
    if feature not in FEATURES:
        raise ValueError(f"Unknown model feature: {feature!r}")
    return FEATURES[feature]


def missing(feature: str) -> list[str]:
    """What first use of ``feature`` would install or download (empty when ready)."""
    needs = [args[-1] for module, args in _requirements(feature) if not _installed(module)]
    if feature == "piano_hr" and not _checkpoint_ready():
        needs.append("Piano-HR checkpoint (~165 MB)")
    if feature == "btc" and not _btc_ready():
        needs.append("BTC chord weights (~12 MB)")
    return needs


def _btc_ready() -> bool:
    from audio_to_midi.btc_paths import CHECKPOINT, CHECKPOINT_BYTES

    return CHECKPOINT.is_file() and CHECKPOINT.stat().st_size == CHECKPOINT_BYTES


def _checkpoint_ready() -> bool:
    return PIANO_HR_CHECKPOINT.is_file() and PIANO_HR_CHECKPOINT.stat().st_size >= PIANO_HR_MIN_BYTES


def ensure(feature: str, progress=None, cancel=None) -> None:
    """Install whatever ``feature`` lacks; no-op when everything is present."""
    report = progress or (lambda text: None)
    label = LABELS.get(feature, feature)
    for module, args in _requirements(feature):
        if _installed(module):
            continue
        report(f"Installing {args[-1]} for {label} (first use)...")
        command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--progress-bar", "off", *args]
        try:
            _run_process(command, cancel=cancel, progress=report, name=f"Installing {args[-1]}")
        except Cancelled:
            raise
        except PreparationError as error:
            # Mirrored feeds (Azure Artifacts upstreams) can 404 a file until they have cached it.
            report(f"Retrying {args[-1]} after: {str(error).splitlines()[0]}")
            _run_process(command, cancel=cancel, progress=report, name=f"Installing {args[-1]}")
        invalidate_caches()
        if not _installed(module):
            raise RuntimeError(f"Installed {args[-1]}, but '{module}' still cannot be imported; restart the app.")
    if feature == "piano_hr" and not _checkpoint_ready():
        download(PIANO_HR_URL, PIANO_HR_CHECKPOINT, "Piano-HR checkpoint", report, cancel)
    if feature == "btc" and not _btc_ready():
        from audio_to_midi.btc_paths import CHECKPOINT, CHECKPOINT_URL

        download(CHECKPOINT_URL, CHECKPOINT, "BTC chord weights", report, cancel)
        if not _btc_ready():
            CHECKPOINT.unlink(missing_ok=True)
            raise RuntimeError("The BTC chord weights downloaded incompletely; try again.")


def download(url: str, target: Path, label: str, progress=None, cancel=None) -> None:
    """Download to ``target`` via a temporary file, reporting whole percentages."""
    report = progress or (lambda text: None)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    shown = -1
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while chunk := response.read(1 << 20):
                if cancel and cancel():
                    raise Cancelled("Download cancelled")
                out.write(chunk)
                done += len(chunk)
                percent = int(100 * done / total) if total else done >> 20
                if percent != shown:
                    shown = percent
                    report(f"Downloading {label}: " + (f"{percent}%" if total else f"{percent} MB"))
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
