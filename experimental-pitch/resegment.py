"""Re-segment cached mono f0 contours without re-running the pitch models.

Variants written next to the originals (so evaluate.py compares them):
  bass   <method>+onset      attack-driven segmentation (common.onset_segment)
  bass   <method>+onset+key  ... plus key-aware folding of short out-of-key runs
  vocals <method>+key        pitch-change segmentation plus key-aware folding

Key awareness: a non-diatonic semitone held < 0.25 s that is not present in
the local harmony ('other' stem chroma) is treated as part of a bend/slide and
takes the pitch of the neighbouring in-key note.

Usage (from experimental-pitch/):
    .venv\\Scripts\\python.exe resegment.py [--song-dir ...] [--key "Bb major"] [--no-harmony]
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import librosa
import numpy as np

from common import (GM_PROGRAM, SONG, PC_NAMES, allowed_pitch_classes, attack_onsets, configure,
                    contour_to_notes, harmony_chroma, load_notes_json, load_stem, onset_segment, onset_times,
                    save_notes_json, song_key, song_out_dir, write_midi)
from methods import RANGES

SR = 22050


def out_of_key_pct(notes, scale) -> float:
    total = sum(n.dur for n in notes) or 1.0
    return round(100 * sum(n.dur for n in notes if n.pitch % 12 not in scale) / total, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song-dir", default=None)
    ap.add_argument("--stems", nargs="+", default=["bass", "vocals"])
    ap.add_argument("--key", default=None, help='override estimated key, e.g. "Bb major" or "G minor"')
    ap.add_argument("--no-harmony", action="store_true", help="scale only; ignore local chords from 'other'")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    configure(args.song_dir)
    out = song_out_dir()
    cfg = json.loads((out / "run_config.json").read_text())
    configure(cfg["song_dir"], cfg["bpm"])
    start, duration = cfg["start"], cfg["duration"]
    run_dir = out / "runs"

    pitched = [s for s in ("bass", "other", "vocals", "piano", "guitar") if s in SONG["stems"]]
    mix = sum(load_stem(s, SR, start, duration) for s in pitched)
    key = song_key(out, mix, SR, args.key)
    print(f"key: {key['name']} ({key['source']}, confidence {key.get('confidence')}) "
          f"scale = {[PC_NAMES[p] for p in key['scale']]}")
    harmony = None
    if not args.no_harmony and "other" in SONG["stems"]:
        harmony = harmony_chroma(load_stem("other", SR, start, duration), SR, start)

    for stem in args.stems:
        if stem not in SONG["stems"]:
            continue
        y = load_stem(stem, SR, start, duration)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=256)[0]
        rms_t = librosa.times_like(rms, sr=SR, hop_length=256) + start
        t0 = time.perf_counter()
        ons = (attack_onsets(y, SR) if stem == "bass" else onset_times(y, SR)) + start
        onset_cost = time.perf_counter() - t0
        for npz in sorted(run_dir.glob(f"{stem}__*.npz")):
            if "+" in npz.stem:
                continue
            c = np.load(npz)
            _, meta = load_notes_json(npz.with_suffix(".json"))
            rms_c = np.interp(c["times"], rms_t, rms)
            allowed = allowed_pitch_classes(c["times"], key, harmony)
            if stem == "bass":
                variants = {"+onset": None, "+onset+key": allowed}
            else:
                variants = {"+key": allowed}
            for suffix, allow in variants.items():
                t1 = time.perf_counter()
                if stem == "bass":
                    notes = onset_segment(c["times"], c["midi"], ons, min_dur=RANGES[stem]["min_dur"],
                                          rms=rms_c, allowed=allow)
                else:
                    midi = c["midi"]
                    notes = contour_to_notes(c["times"], midi, ~np.isnan(midi), onsets=ons,
                                             min_dur=RANGES[stem]["min_dur"], rms=rms_c, allowed=allow)
                name = meta["method"] + suffix
                runtime = meta["runtime_s"] + onset_cost + time.perf_counter() - t1
                m = meta | {"method": name, "runtime_s": round(runtime, 2),
                            "x_realtime": round(meta["audio_s"] / max(runtime, 1e-6), 1),
                            "key": key["name"] if allow is not None else None}
                base = run_dir / f"{stem}__{name}"
                save_notes_json(notes, base.with_suffix(".json"), m)
                write_midi(notes, base.with_suffix(".mid"), program=GM_PROGRAM.get(stem, 0))
                np.savez_compressed(base.with_suffix(".npz"), **{k: c[k] for k in c.files})
                print(f"{stem}/{name}: {len(notes)} notes, out-of-key time {out_of_key_pct(notes, key['scale'])}%")


if __name__ == "__main__":
    main()
