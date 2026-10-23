"""Subprocess entry point: ``python -m audio_to_midi.separation_worker MIX OUTPUT_DIR``.

Runs Demucs htdemucs_6s (weights download on first use) and writes one WAV per
source with the mix's own sample rate, length and channel count. Audio I/O uses
soundfile, so neither ffmpeg nor torchaudio's codecs are needed.
"""

from __future__ import annotations

from pathlib import Path
import os
import sys

import numpy as np
import soundfile as sf

MODEL = "htdemucs_6s"
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def main(argv: list[str]) -> int:
    mix, output = Path(argv[0]), Path(argv[1])
    import torch
    from demucs.apply import apply_model
    from demucs.audio import convert_audio
    from demucs.pretrained import get_model

    print(f"Loading the {MODEL} model (downloaded the first time, about 80 MB)...", flush=True)
    model = get_model(MODEL)
    model.eval()
    data, rate = sf.read(mix, dtype="float32", always_2d=True)
    frames, channels = data.shape
    wav = convert_audio(torch.from_numpy(np.ascontiguousarray(data.T)), rate, model.samplerate, model.audio_channels)
    reference = wav.mean(0)
    mean, std = reference.mean(), reference.std() + 1e-8
    length = wav.shape[-1]
    shown = [-1]

    def callback(info: dict) -> None:
        if info.get("state") != "end":
            return
        done = (info.get("model_idx_in_bag", 0) * length + info.get("segment_offset", 0)) / max(1, info.get("models", 1) * length)
        percent = min(99, int(100 * done))
        if percent >= shown[0] + 5:
            shown[0] = percent
            print(f"Separating stems: {percent}%", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Separating stems on {device.upper()}; a full song takes a few minutes...", flush=True)
    with torch.no_grad():
        sources = apply_model(model, ((wav - mean) / std)[None], device=device, split=True, overlap=0.25,
                              progress=False, callback=callback)[0]
    sources = sources * std + mean
    output.mkdir(parents=True, exist_ok=True)
    for name, source in zip(model.sources, sources):
        audio = convert_audio(source.cpu(), model.samplerate, rate, channels).numpy().T
        audio = audio[:frames] if len(audio) >= frames else np.pad(audio, ((0, frames - len(audio)), (0, 0)))
        sf.write(output / f"{name}.wav", audio, rate, subtype="FLOAT")
    print("Separating stems: 100%", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
