"""Separate a mix into the 6 htdemucs_6s stems (drums, bass, other, vocals, guitar, piano).

Usage (from experimental-pitch/):
    .venv\\Scripts\\python.exe separate_6s.py "..\\Firm Foundation (He Won't)\\Firm Foundation (He Won't) 75 exact.wav"
Writes <mix folder>\\htdemucs_6s\\<stem>.wav, the Demucs naming that run_methods.py --song-dir understands.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mix")
    ap.add_argument("--model", default="htdemucs_6s")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mix_path = Path(args.mix)
    out = Path(args.out) if args.out else mix_path.parent / args.model
    out.mkdir(parents=True, exist_ok=True)
    model = get_model(args.model)
    model.eval()
    audio, sr = sf.read(str(mix_path), dtype="float32", always_2d=True)
    if sr != model.samplerate:
        import librosa
        audio = librosa.resample(audio.T, orig_sr=sr, target_sr=model.samplerate).T
        sr = model.samplerate
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    wav = torch.from_numpy(audio.T[: model.audio_channels].copy())
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)
    print(f"separating {mix_path.name} ({wav.shape[1] / sr:.1f}s) with {args.model} -> {out}", flush=True)
    t0 = time.perf_counter()
    with torch.inference_mode():
        sources = apply_model(model, wav[None], device="cpu", shifts=1, split=True, overlap=0.25,
                              progress=True)[0]
    sources = sources * (ref.std() + 1e-8) + ref.mean()
    for name, source in zip(model.sources, sources):
        sf.write(str(out / f"{name}.wav"), source.numpy().T, sr, subtype="PCM_16")
        print(f"  wrote {name}.wav", flush=True)
    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
