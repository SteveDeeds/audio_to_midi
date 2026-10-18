"""YourMT3+ drum worker. Runs inside .venv-mt3 (separate env: needs transformers==4.45.1).

    .venv-mt3\\Scripts\\python.exe mt3_drums_worker.py <audio> <out.json> [--start s] [--duration s]

Writes {"hits": [{t, pitch, vel}], "other_notes": n, "runtime_s": x, "audio_s": y}.
Only drum-channel notes are kept; pitched notes are counted (they are bleed on a drum stem).
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MT3_CHECKPOINT_DIR", str(HERE / "third_party" / "mt3_checkpoints"))

import librosa  # noqa: E402
import mt3_infer  # noqa: E402
import pretty_midi  # noqa: E402


def main() -> None:
    audio, out = sys.argv[1], Path(sys.argv[2])
    args = sys.argv[3:]
    start = float(args[args.index("--start") + 1]) if "--start" in args else 0.0
    duration = float(args[args.index("--duration") + 1]) if "--duration" in args else None
    y, _ = librosa.load(audio, sr=16000, mono=True, offset=start, duration=duration)
    t0 = time.perf_counter()
    midi = mt3_infer.transcribe(y, sr=16000, model="yourmt3")
    runtime = time.perf_counter() - t0
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "out.mid"
        midi.save(str(p))
        pm = pretty_midi.PrettyMIDI(str(p))
    hits, other = [], 0
    for inst in pm.instruments:
        if inst.is_drum:
            hits += [{"t": round(n.start + start, 4), "pitch": n.pitch, "vel": n.velocity} for n in inst.notes]
        else:
            other += len(inst.notes)
    out.write_text(json.dumps({"hits": hits, "other_notes": other, "runtime_s": round(runtime, 1),
                               "audio_s": round(len(y) / 16000, 1)}))
    print(f"yourmt3: {len(hits)} drum hits, {other} pitched notes ignored, {runtime:.0f}s")


if __name__ == "__main__":
    main()
