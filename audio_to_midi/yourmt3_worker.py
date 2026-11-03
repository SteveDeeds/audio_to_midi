"""Inference worker entry point; imports the optional model only in the child."""

from importlib import metadata
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

from audio_to_midi.evidence_cache import checksum


def identify(checkpoint: Path) -> dict:
    distribution = metadata.distribution("mt3-infer")
    if distribution.version != "0.2.0":
        raise RuntimeError("YourMT3+ requires the validated mt3-infer==0.2.0 adapter.")
    adapter = distribution.locate_file("mt3_infer/adapters/yourmt3.py")
    return dict(name="yourmt3", variant="yptf_moe_nops", package="mt3-infer",
                version=distribution.version, adapter_sha256=checksum(Path(adapter)),
                runtime={p: metadata.version(p) for p in ("torch", "transformers", "librosa", "numpy")},
                checkpoint_sha256=checksum(checkpoint), note_confidence_available=False)


def infer(audio_path: str, checkpoint: Path) -> dict:
    import librosa
    import torch
    import pretty_midi
    from core.inference_runtime import neural_device

    # mt3-infer's vendored imports reference helpers removed in new transformers.
    # Model parallelism is not used in single-device inference.
    try:
        import transformers.utils.model_parallel_utils
    except ModuleNotFoundError:
        import types
        stub = types.ModuleType("transformers.utils.model_parallel_utils")
        def unavailable(*args, **kwargs):
            raise RuntimeError("Model parallelism is unsupported in the YourMT3+ worker.")
        stub.assert_device_map = stub.get_device_map = unavailable
        sys.modules[stub.__name__] = stub
    from mt3_infer.adapters.yourmt3 import YourMT3Adapter

    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("YourMT3+ input must contain finite audio.")
    with neural_device(wait=True) as device, torch.inference_mode():
        print(f"Loading YourMT3+ on {device.upper()}...", flush=True)
        model = YourMT3Adapter(model_key="yptf_moe_nops")
        model.load_model(checkpoint_path=str(checkpoint), device=device)
        features = model.preprocess(audio, sr=16000)
        predictions = []
        for start in range(0, len(features), 8):
            predictions.extend(model.forward(features[start:start + 8]))
            print(f"YourMT3+ inference: {min(start + 8, len(features))}/{len(features)} segments", flush=True)
        print("Decoding YourMT3+ notes (no note-confidence evidence)...", flush=True)
        midi = model.decode(predictions)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "output.mid"
            midi.save(str(path))
            parsed = pretty_midi.PrettyMIDI(str(path))
        notes = [[float(n.start), float(n.end), n.pitch, n.velocity, inst.program, int(inst.is_drum)]
                 for inst in parsed.instruments for n in inst.notes]
        return dict(notes=np.asarray(notes, dtype=np.float64).reshape(-1, 6))


def main():
    command, output, checkpoint_path = sys.argv[1:4]
    checkpoint = Path(checkpoint_path)
    if command == "identify":
        Path(output).write_text(json.dumps(identify(checkpoint)), encoding="utf-8")
    elif command == "infer":
        identify(checkpoint)
        np.savez_compressed(output, **infer(sys.argv[4], checkpoint))
    else:
        raise ValueError(f"Unknown YourMT3+ worker command: {command}")


if __name__ == "__main__":
    main()
