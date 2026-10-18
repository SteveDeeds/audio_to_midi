"""Run the pitch bake-off on the loudest window of each pitched stem.

Each stem gets its own output folder (outputs/<label>/<stem>/<song dir>) because
run_methods/resegment/evaluate share one analysis window per folder.

Usage (from experimental-pitch/):
    .venv\\Scripts\\python.exe run_loudest.py --song-dir "..\\Firm Foundation (He Won't)\\htdemucs_6s" --bpm 75
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np

import common

SR = 22050


def loudest_window(path: Path, seconds: float) -> tuple[float, float, float]:
    """(start, window RMS dB, whole-stem RMS dB) of the loudest `seconds` window."""
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    hop = SR // 2
    rms = librosa.feature.rms(y=y, frame_length=hop, hop_length=hop, center=False)[0]
    n = max(1, int(round(seconds * SR / hop)))
    power = np.convolve(rms ** 2, np.ones(n) / n, mode="valid")
    i = int(np.argmax(power))
    db = lambda p: float(10 * np.log10(max(p, 1e-12)))
    return i * hop / SR, db(power[i]), db(float(np.mean(rms ** 2)))


def run(module: str, argv: list[str]) -> None:
    sys.argv = [module, *argv]
    __import__(module).main()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song-dir", required=True)
    ap.add_argument("--bpm", type=float, default=None)
    ap.add_argument("--stems", nargs="+", default=None)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--label", default=None, help="output folder name (default: <song dir>_loudest)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    song = common.configure(args.song_dir, args.bpm)
    stems = args.stems or [s for s in common.PITCHED_STEMS if s in song["stems"]]
    root = common.ROOT / "outputs" / (args.label or f"{Path(args.song_dir).name}_loudest")
    bar = 4 * 60.0 / song["bpm"]
    view_bars = max(1, int(args.duration // bar))
    summary = {}
    for stem in stems:
        start, window_db, stem_db = loudest_window(song["stems"][stem], args.duration)
        print(f"\n=== {stem}: loudest {args.duration:.0f}s window starts {start:.1f}s "
              f"({window_db:.1f} dB vs {stem_db:.1f} dB song average)", flush=True)
        common.OUT_DIR = root / stem
        common_args = ["--song-dir", args.song_dir]
        bpm_args = ["--bpm", str(args.bpm)] if args.bpm else []
        run("run_methods", [*common_args, *bpm_args, "--stems", stem, "--start", f"{start:.2f}",
                            "--duration", str(args.duration), *(["--force"] if args.force else [])])
        if stem in ("bass", "vocals"):
            run("resegment", [*common_args, "--stems", stem])
        view = start + max(0.0, (args.duration - view_bars * bar) / 2)
        run("evaluate", [*common_args, "--view-start", f"{view:.2f}", "--view-bars", str(view_bars)])
        summary[stem] = {"start": round(start, 2), "duration": args.duration, "window_db": round(window_db, 1),
                         "stem_db": round(stem_db, 1), "out": str(common.song_out_dir())}
    root.mkdir(parents=True, exist_ok=True)
    (root / "loudest_windows.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
