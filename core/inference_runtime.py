"""Shared device conventions and exclusive GPU ownership for app workers."""

from contextlib import contextmanager
import os

from audio_to_midi.evidence_cache import file_lock


def torch_device(default: str = "cpu") -> str:
    device = os.environ.get("ATM_DEVICE", default)
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("ATM_DEVICE must be auto, cpu, or cuda.")
    if device == "cpu":
        return device
    import torch
    available = torch.cuda.is_available()
    if device == "cuda" and not available:
        raise RuntimeError("CUDA was requested but is unavailable in this Python environment.")
    return "cuda" if available else "cpu"


@contextmanager
def neural_device(default: str = "auto", cancelled=lambda: False, *, wait: bool = False):
    device = torch_device(default)
    if device == "cpu":
        yield device
        return
    with file_lock("audio-to-midi-gpu", cancelled, wait=wait) as acquired:
        # Faster models can run on CPU while the dedicated YourMT3 worker owns CUDA.
        yield "cuda" if acquired else "cpu"


def crepe_batch_size() -> int:
    batch = int(os.environ.get("ATM_CREPE_BATCH_SIZE", "1024"))
    if batch < 1:
        raise ValueError("ATM_CREPE_BATCH_SIZE must be positive.")
    return batch


def basic_pitch_model():
    from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
    from basic_pitch.inference import Model

    path = build_icassp_2022_model_path(FilenameSuffix.onnx)
    setting = os.environ.get("ATM_INFERENCE_THREADS")
    if setting is None:
        return Model(path)
    threads = int(setting)
    if threads < 1:
        raise ValueError("ATM_INFERENCE_THREADS must be positive.")
    import onnxruntime as ort

    class ThreadLimitedOnnxModel(Model):
        def __init__(self):
            self.model_type = Model.MODEL_TYPES.ONNX
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            self.model = ort.InferenceSession(str(path), sess_options=options,
                                            providers=["CPUExecutionProvider"])

    return ThreadLimitedOnnxModel()
