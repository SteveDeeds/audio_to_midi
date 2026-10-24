"""Read-only runtime checks; inspection never installs or downloads models."""

from __future__ import annotations

from importlib import import_module, metadata, util
from contextlib import redirect_stdout
from pathlib import Path
import sys


BASE_PACKAGES = {
    "numpy": "numpy", "scipy": "scipy", "soundfile": "soundfile", "mido": "mido",
    "librosa": "librosa", "PySide6": "PySide6", "pygame": "pygame",
}
OPTIONAL_PACKAGES = {
    "torchcrepe": "torchcrepe", "basic-pitch": "basic_pitch",
    "onnxruntime": "onnxruntime", "mir_eval": "mir_eval", "resampy": "resampy",
    "scikit-learn": "sklearn", "piano-transcription-inference": "piano_transcription_inference",
    "adtof-pytorch": "adtof_pytorch", "demucs": "demucs",
}
PIANO_HR_CHECKPOINT = Path.home() / "piano_transcription_inference_data" / "note_F1=0.9677_pedal_F1=0.9186.pth"


def inspect_environment() -> dict:
    packages = {}
    missing = []
    for distribution, module in {**BASE_PACKAGES, **OPTIONAL_PACKAGES}.items():
        installed = util.find_spec(module) is not None
        version = None
        if installed:
            try:
                version = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                version = "unversioned"
        packages[distribution] = {"installed": installed, "version": version}
        if distribution in BASE_PACKAGES:
            if not installed:
                missing.append(distribution)
            else:
                # Base imports catch binary ABI/Qt issues without constructing devices.
                try:
                    with redirect_stdout(sys.stderr):
                        import_module(module)
                except ImportError as exc:
                    packages[distribution]["import_error"] = str(exc)
                    missing.append(distribution)
    return {
        "python": sys.version.split()[0], "executable": sys.executable,
        "base_ready": not missing, "missing_base": missing, "packages": packages,
        "piano_hr_checkpoint_local": PIANO_HR_CHECKPOINT.is_file(),
        "notes": [
            "Optional neural models are not required to edit projects, import/export MIDI, or use pYIN.",
            "Basic Pitch 0.4.0 uses ONNX here; its mandatory TensorFlow package metadata can make pip check report TensorFlow missing.",
            "Missing models (CREPE, Basic Pitch, Piano-HR, ADTOF, Demucs) are pip-installed into this Python and their weights downloaded the first time they are used.",
            "ADTOF weights are non-commercial; review all checkpoint licenses before distribution.",
        ],
    }
