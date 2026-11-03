"""Backend-specific inference and replay. No calibrated cross-model confidence."""

from __future__ import annotations

from importlib import metadata
import os
from pathlib import Path

import librosa
import numpy as np

from audio_to_midi.evidence_cache import EvidenceCache, array_identity, checksum
from core.inference_runtime import basic_pitch_model, crepe_batch_size, neural_device

RANGES = {
    "bass": (30.0, 420.0, 28, 67),
    "guitar": (75.0, 1400.0, 38, 88),
    "piano": (27.5, 4200.0, 21, 108),
    "vocals": (70.0, 1100.0, 36, 96),
}
_BP = None


def validate_parameters(model: str, parameters: dict) -> None:
    supported = (
        {"onset_threshold", "frame_threshold"} if model in ("basic_pitch", "piano_hr") else
        {"onset_threshold", "voicing_threshold"} if model.startswith("madmom") else
        {"onset_threshold"} if model == "adtof" else
        {"voicing_threshold"} if model in ("pyin", "crepe_full", "crepe_tiny") else set()
    )
    if not isinstance(parameters, dict) or set(parameters) - supported:
        raise ValueError(f"{model} does not support these decoder parameters; YourMT3+ has no note confidence/logits.")
    for name in parameters:
        threshold(parameters, name, 0)


def check_cancel(cancelled):
    if cancelled():
        from audio_to_midi.adapters import Cancelled
        raise Cancelled()


def model_identity(model: str) -> dict:
    distributions = {
        "basic_pitch": "basic-pitch", "piano_hr": "piano-transcription-inference",
        "crepe_full": "torchcrepe", "crepe_tiny": "torchcrepe", "pyin": "librosa",
        "madmom_cnn_full": "madmom", "madmom_brnn_full": "madmom", "adtof": "adtof-pytorch",
    }
    distribution = metadata.distribution(distributions[model])
    files = list(distribution.files or [])
    if model == "piano_hr":
        from audio_to_midi.environment import PIANO_HR_CHECKPOINT
        assets = [PIANO_HR_CHECKPOINT]
    elif model.startswith("crepe"):
        assets = [distribution.locate_file(f) for f in files
                  if str(f).endswith(f"assets/{model.removeprefix('crepe_')}.pth")]
    elif model.startswith("madmom"):
        assets = [distribution.locate_file(f) for f in files
                  if "models/onsets/" in str(f) and str(f).endswith(".pkl")]
    elif model == "basic_pitch":
        assets = [distribution.locate_file(f) for f in files if str(f).endswith("nmp.onnx")]
    elif model == "adtof":
        assets = [distribution.locate_file(f) for f in files if str(f).endswith("_weights.pth")]
    else:
        assets = []
    if model != "pyin" and not assets:
        raise RuntimeError(f"No installed checkpoint assets found for {model}.")
    runtime = ["numpy", "librosa", "scipy"]
    if model in ("piano_hr", "crepe_full", "crepe_tiny", "adtof"):
        runtime.append("torch")
    if model == "basic_pitch":
        runtime.append("onnxruntime")
    return dict(name=model, package=distribution.metadata["Name"], version=distribution.version,
                runtime={package: metadata.version(package) for package in runtime},
                checkpoints={Path(p).name: checksum(Path(p)) for p in assets})


def validate(model: str, arrays: dict) -> None:
    fields = {
        "basic_pitch": {"onset": 88, "note": 88, "contour": 264},
        "piano_hr": {"reg_onset_output": 88, "reg_offset_output": 88,
                     "frame_output": 88, "velocity_output": 88},
        "crepe_full": {"pitch_bin_probabilities": 360, "f0_hz": None, "periodicity": None,
                       "smoothed_periodicity": None, "db": None},
        "crepe_tiny": {"pitch_bin_probabilities": 360, "f0_hz": None, "periodicity": None,
                       "smoothed_periodicity": None, "db": None},
        "pyin": {"f0_hz": None, "voiced": None, "voiced_probability": None, "db": None},
        "madmom_cnn_full": {"onset": None}, "madmom_brnn_full": {"onset": None},
        "adtof": {"activation": 5, "novelty": 5},
    }[model]
    t = arrays["t"]
    if t.ndim != 1 or not len(t) or not np.isfinite(t).all() or not (np.diff(t) > 0).all():
        raise ValueError("Invalid evidence timestamps.")
    for field, width in fields.items():
        value = arrays[field]
        if value.shape != ((len(t),) if width is None else (len(t), width)):
            raise ValueError(f"Invalid {model} evidence shape: {field}")
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite {model} evidence: {field}")
        if field not in ("f0_hz", "db") and np.any((value < 0) | (value > 1)):
            raise ValueError(f"Out-of-range activation: {field}")
    for value in arrays.values():
        if not np.isfinite(value).all():
            raise ValueError("Non-finite evidence.")
    if "f0_hz" in arrays and np.any(arrays["f0_hz"] <= 0):
        raise ValueError("Unmasked F0 must remain positive.")
    if model == "piano_hr":
        for name in ("reg_pedal_onset_output", "reg_pedal_offset_output", "pedal_frame_output"):
            if name in arrays and (
                arrays[name].shape != (len(t), 1) or np.any((arrays[name] < 0) | (arrays[name] > 1))
            ):
                raise ValueError(f"Invalid Piano HR pedal evidence: {name}")


def pitch_evidence(audio: np.ndarray, sr: int, role: str, model: str, cancelled=lambda: False,
                   cache: EvidenceCache | None = None) -> dict:
    low, high, _, _ = RANGES[role]
    if model.startswith("crepe"):
        low, high = max(low, 32.71), min(high, 2006.0)
    parameters = dict(fmin=low, fmax=high, preprocessing="mono-unscaled", sr=sr,
                      hop=256 if model == "pyin" else 160, frame=2048,
                      decoder="pyin-fill-na-none" if model == "pyin" else "viterbi",
                      device_policy="cpu" if model == "pyin" else os.environ.get("ATM_DEVICE", "auto"),
                      periodicity_median=3,
                      batch_size=crepe_batch_size() if model != "pyin" else None)
    store = cache or EvidenceCache()
    descriptor = store.descriptor(array_identity(audio, sr), f"stem:{role}", model_identity(model), parameters)

    def compute():
        check_cancel(cancelled)
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=256)[0]
        db = librosa.amplitude_to_db(rms + 1e-9, ref=np.max(rms) + 1e-9)
        rms_t = librosa.times_like(rms, sr=sr, hop_length=256)
        if model == "pyin":
            f0, voiced, probability = librosa.pyin(
                audio, fmin=low, fmax=high, sr=sr, frame_length=2048, hop_length=256, fill_na=None)
            arrays = dict(t=librosa.times_like(f0, sr=sr, hop_length=256),
                          f0_hz=f0, voiced=voiced, voiced_probability=probability)
        else:
            import torch
            import torchcrepe
            y = torch.from_numpy(librosa.resample(audio, orig_sr=sr, target_sr=16000))[None]
            scores, pitches, periodicities = [], [], []
            with neural_device(cancelled=cancelled) as device, torch.inference_mode():
                for frames in torchcrepe.preprocess(
                    y, 16000, hop_length=160, batch_size=crepe_batch_size(), device=device, pad=True,
                ):
                    check_cancel(cancelled)
                    activation = torchcrepe.infer(frames, model.removeprefix("crepe_"), device)
                    # postprocess masks pitch bins in place; retain the untouched planes first.
                    scores.append(activation.cpu().numpy().copy())
                    f0, probability = torchcrepe.postprocess(
                        activation.reshape(1, -1, 360).transpose(1, 2), low, high,
                        torchcrepe.decode.viterbi, return_periodicity=True)
                    pitches.append(f0.cpu())
                    periodicities.append(probability.cpu())
            f0 = torch.cat(pitches, 1)[0].numpy()
            periodicity = torch.cat(periodicities, 1)
            arrays = dict(t=np.arange(len(f0)) * .01, f0_hz=f0,
                          pitch_bin_probabilities=np.concatenate(scores),
                          periodicity=periodicity[0].numpy(),
                          smoothed_periodicity=torchcrepe.filter.median(periodicity, 3)[0].numpy())
        arrays["db"] = np.interp(arrays["t"], rms_t, db)
        return arrays

    return store.get(descriptor, compute, lambda a: validate(model, a), cancelled)


def onset_evidence(audio: np.ndarray, sr: int, role: str, model: str, cancelled=lambda: False,
                   cache: EvidenceCache | None = None) -> dict:
    store = cache or EvidenceCache()
    descriptor = store.descriptor(array_identity(audio, sr), f"stem:{role}", model_identity(model),
                                  dict(sr=44100, fps=100, online=False, preprocessing="mono-unscaled"))

    def compute():
        check_cancel(cancelled)
        from madmom.audio.signal import Signal
        from madmom.features.onsets import CNNOnsetProcessor, RNNOnsetProcessor
        processor = CNNOnsetProcessor() if model == "madmom_cnn_full" else RNNOnsetProcessor(online=False)
        activation = np.asarray(processor(Signal(
            librosa.resample(audio, orig_sr=sr, target_sr=44100), sample_rate=44100))).reshape(-1)
        check_cancel(cancelled)
        return dict(t=np.arange(len(activation)) / 100, onset=activation)

    return store.get(descriptor, compute, lambda a: validate(model, a), cancelled)


def file_evidence(path: str, source: str, model: str, cancelled=lambda: False,
                  report=lambda text: None, cache: EvidenceCache | None = None) -> dict:
    if model == "yourmt3":
        from audio_to_midi.yourmt3 import evidence
        return evidence(path, source, cancelled, report, cache)
    store = cache or EvidenceCache()
    before = checksum(Path(path))
    descriptor = store.descriptor(dict(sha256=before), source, model_identity(model),
                                  dict(preprocessing="librosa-mono-unscaled", inference_revision=1,
                                       sample_rate=22050 if model == "basic_pitch" else
                                       16000 if model == "piano_hr" else 44100,
                                       precision="float32", device_policy="cpu" if model == "basic_pitch"
                                       else os.environ.get("ATM_DEVICE", "auto")))

    def compute():
        global _BP
        check_cancel(cancelled)
        report(f"Running {model} inference ({source})...")
        if model == "basic_pitch":
            from basic_pitch.inference import run_inference
            from basic_pitch.note_creation import model_frames_to_time
            if _BP is None:
                _BP = basic_pitch_model()
            arrays = run_inference(path, _BP)
            arrays["t"] = model_frames_to_time(len(arrays["onset"]))
        elif model == "piano_hr":
            from piano_transcription_inference import PianoTranscription, sample_rate
            audio, _ = librosa.load(path, sr=sample_rate, mono=True)
            with neural_device(cancelled=cancelled) as device:
                network = PianoTranscription(device=device, checkpoint_path=None)
                output = network.transcribe(audio, None)["output_dict"]
                arrays = {k: v for k, v in output.items() if k.startswith("reg_") or
                          k in ("frame_output", "velocity_output", "pedal_frame_output")}
                arrays["t"] = np.arange(len(arrays["frame_output"])) / network.frames_per_second
                del network
        elif model == "adtof":
            from core.adtof_drums import compute_activations, peak_novelty, FPS
            audio, _ = librosa.load(path, sr=44100, mono=True)
            activation = compute_activations(audio)
            arrays = dict(t=np.arange(len(activation)) / FPS, activation=activation,
                          novelty=peak_novelty(activation))
        else:
            raise ValueError(f"Unsupported file evidence model: {model}")
        check_cancel(cancelled)
        if checksum(Path(path)) != before:
            raise RuntimeError("Audio changed during inference; result was not cached.")
        return arrays

    return store.get(descriptor, compute, lambda a: validate(model, a), cancelled)


def threshold(parameters: dict, name: str, default: float) -> float:
    value = parameters.get(name, default)
    if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and within 0..1.")
    return float(value)


def replay_poly(model: str, arrays: dict, role: str, parameters: dict) -> list[tuple]:
    validate_parameters(model, parameters)
    low, high, lo, hi = RANGES[role]
    if model == "basic_pitch":
        from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP
        from basic_pitch.note_creation import model_output_to_notes
        _, events = model_output_to_notes(
            {k: arrays[k].copy() for k in ("onset", "note", "contour")},
            onset_thresh=threshold(parameters, "onset_threshold", .5),
            frame_thresh=threshold(parameters, "frame_threshold", .3),
            min_note_len=int(np.round(.06 * AUDIO_SAMPLE_RATE / FFT_HOP)),
            min_freq=low, max_freq=high, melodia_trick=True)
        notes = [(float(a), float(b), int(p), float(v), float(v)) for a, b, p, v, _ in events]
    elif model == "piano_hr":
        from piano_transcription_inference.utilities import RegressionPostProcessor
        class ReplayProcessor(RegressionPostProcessor):
            def is_monotonic_neighbour(self, x, n, neighbour):
                # Flat plateaus have no regression peak; upstream interpolation divides by zero.
                return (x[n] > min(x[n - 1], x[n + 1])
                        and super().is_monotonic_neighbour(x, n, neighbour))

        processor = ReplayProcessor(
            100, classes_num=88, onset_threshold=threshold(parameters, "onset_threshold", .3),
            offset_threshold=.3, frame_threshold=threshold(parameters, "frame_threshold", .1),
            pedal_offset_threshold=.2)
        events, _ = processor.output_dict_to_midi_events({k: v.copy() for k, v in arrays.items() if k != "t"})
        from core.poly_pitch import piano_frame_confidence
        notes = [
            (float(e["onset_time"]), float(e["offset_time"]), int(e["midi_note"]),
             float(e["velocity"]) / 127,
             piano_frame_confidence(
                 arrays["frame_output"], float(e["onset_time"]), float(e["offset_time"]),
                 int(e["midi_note"]), 100, 21))
            for e in events
        ]
    else:
        raise ValueError(f"Unsupported polyphonic replay: {model}")
    return [n for n in notes if lo <= n[2] <= hi and n[1] > n[0]]


def replay_pitch(model: str, arrays: dict, parameters: dict) -> tuple[np.ndarray, np.ndarray]:
    validate_parameters(model, {k: v for k, v in parameters.items() if k != "onset_threshold"})
    voiced = (arrays["voiced"] if model == "pyin" and "voicing_threshold" not in parameters else
              arrays["voiced_probability"] >= threshold(parameters, "voicing_threshold", 0)
              if model == "pyin" else
              arrays["smoothed_periodicity"] > threshold(parameters, "voicing_threshold", .35))
    voiced = voiced & (arrays["db"] > -45)
    return arrays["t"], np.where(voiced, librosa.hz_to_midi(np.maximum(arrays["f0_hz"], 1)), np.nan)
