"""Run each pitch method on each stem and cache notes/contours/timings.

Usage (from experimental-pitch/):
    .venv\\Scripts\\python.exe run_methods.py                     # Firm Foundation, full song, all stems/methods
    .venv\\Scripts\\python.exe run_methods.py --stems bass --methods pyin crepe_tiny --start 60 --duration 30
    .venv\\Scripts\\python.exe run_methods.py --song-dir ..\\..\\EchoGrid\\output\\stems\\htdemucs_6s\\song_111bpm
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np

from common import GM_PROGRAM, PITCHED_STEMS, SONG, configure, load_stem, save_notes_json, song_out_dir, write_midi
from methods import METHODS, param_count

SR = 22050


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song-dir", default=None, help="folder of stems (UVR '(Bass)' or Demucs 'bass.wav' naming)")
    ap.add_argument("--bpm", type=float, default=None, help="override tempo parsed from file/folder names")
    ap.add_argument("--stems", nargs="+", default=None, help="default: all pitched stems found")
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--force", action="store_true", help="re-run even if cached")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    configure(args.song_dir, args.bpm)
    stems = args.stems or [s for s in PITCHED_STEMS if s in SONG["stems"]]
    out = song_out_dir()
    run_dir = out / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps({"song_dir": str(SONG["dir"]), "bpm": SONG["bpm"],
                                                     "start": args.start, "duration": args.duration}))
    print(f"song: {SONG['dir']}  bpm={SONG['bpm']}  stems={sorted(SONG['stems'])}")

    for stem in stems:
        y = load_stem(stem, SR, args.start, args.duration)
        audio_dur = len(y) / SR
        for name in args.methods:
            fn, stems = METHODS[name]
            if stem not in stems:
                continue
            base = run_dir / f"{stem}__{name}"
            if base.with_suffix(".json").exists() and not args.force:
                print(f"[skip] {stem}/{name} (cached)")
                continue
            print(f"[run ] {stem}/{name} ...", flush=True)
            t0 = time.perf_counter()
            try:
                notes, contour = fn(stem, y, SR)
            except Exception as exc:  # keep the bake-off going if one model breaks
                print(f"[FAIL] {stem}/{name}: {exc!r}")
                continue
            elapsed = time.perf_counter() - t0
            for n in notes:  # shift back to absolute song time
                n.start += args.start
                n.end += args.start
            meta = {"stem": stem, "method": name, "runtime_s": round(elapsed, 2),
                    "audio_s": round(audio_dur, 2), "x_realtime": round(audio_dur / max(elapsed, 1e-6), 1),
                    "params": param_count(name), "start": args.start}
            save_notes_json(notes, base.with_suffix(".json"), meta)
            write_midi(notes, base.with_suffix(".mid"), program=GM_PROGRAM.get(stem, 0))
            if contour is not None:
                np.savez_compressed(base.with_suffix(".npz"), times=contour["times"] + args.start,
                                    midi=contour["midi"], conf=contour["conf"])
            print(f"       {len(notes)} notes in {elapsed:.1f}s ({meta['x_realtime']}x realtime)")


if __name__ == "__main__":
    main()
