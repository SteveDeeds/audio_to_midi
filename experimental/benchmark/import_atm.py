"""Index an external ``atm_training_set`` batch as benchmark clips, without copying audio.

    python -m experimental.benchmark.import_atm E:\\atm_training_set\\expanded [--include-flagged]

For each composition in ``pairs.json`` this writes ``data/clips/<prefix><id>/truth.json`` holding
note rows from ``midi/<instrument>.mid`` (the per-instrument labels, never ``source.mid``),
quarter-note anchors from the MIDI tempo map, the batch's ``splits.json`` assignment, and
absolute paths to the batch's ``mix.wav``, Demucs ``stems/`` (source ``demucs``) and, when
present, pre-Demucs ``clean/`` renders (source ``clean``). The harness, ``ensemble.py``,
``pitch_head.py`` and ``crosssong.py`` then accept those clip folders like rendered ones.

Flags are applied per view: the batch's ``flagged`` field gates the Demucs stem, the clean
render's own energy report gates the clean view. Flagged views are skipped unless
``--include-flagged``. The prefix is ``atm_`` for the pilot and ``atmx_`` otherwise, since a
song can appear in both batches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pretty_midi
import soundfile as sf

from experimental.benchmark.prepare import CLIPS, _plain

ROLES = ("bass", "drums", "piano", "guitar")


def quarter_anchors(pm: pretty_midi.PrettyMIDI, seconds: float) -> list[list]:
    """(seconds, quarter) at every quarter note from the tempo map, covering ``seconds`` + 2."""
    times, tempi = pm.get_tempo_changes()
    if not len(tempi):
        times, tempi = [0.0], [120.0]
    anchors, t, q, segment = [], 0.0, 0, 0
    while t <= seconds + 2.0:
        anchors.append([round(t, 6), q])
        while segment + 1 < len(times) and times[segment + 1] <= t + 1e-9:
            segment += 1
        t += 60.0 / float(tempi[segment])
        q += 1
    return anchors


def _clean_flagged(root: Path, pair: dict) -> bool:
    report = json.loads((root / pair["clean_quality"]).read_text(encoding="utf-8"))
    return report["missing_notes"] > 0 and report["missing_fraction"] >= 0.1


def import_batch(root: Path, include_flagged: bool, min_notes: int, prefix: str = "atm_") -> list[str]:
    pairs = json.loads((root / "pairs.json").read_text(encoding="utf-8"))
    splits_file = root / "splits.json"
    splits = (json.loads(splits_file.read_text(encoding="utf-8"))["composition_to_split"]
              if splits_file.exists() else {})
    by_song: dict[str, list[dict]] = {}
    for pair in pairs:
        by_song.setdefault(pair["composition_id"], []).append(pair)
    written = []
    for song, items in sorted(by_song.items()):
        folder = root / song
        mix = folder / "mix.wav"
        duration = sf.info(mix).duration
        notes, programs, skipped = {}, {}, {}
        audio: dict = {"mix": str(mix.resolve()), "demucs": {}, "clean": {}}
        for pair in items:
            role = pair["instrument"]
            if role not in ROLES:
                continue
            pm = pretty_midi.PrettyMIDI(str(root / pair["midi"]))
            rows = sorted([round(n.start, 6), round(n.end, 6), int(n.pitch), int(n.velocity)]
                          for inst in pm.instruments for n in inst.notes)
            if len(rows) < min_notes:
                skipped[role] = f"{len(rows)} notes"
                continue
            # Flags are per view: a Demucs failure doesn't invalidate the clean render.
            views = {"demucs": (pair["audio"], pair["flagged"])}
            if pair.get("clean_audio"):
                views["clean"] = (pair["clean_audio"], _clean_flagged(root, pair))
            for source, (relative, flagged) in views.items():
                if flagged and not include_flagged:
                    skipped[f"{source}/{role}"] = "flagged"
                else:
                    audio[source][role] = str((root / relative).resolve())
            if any(role in audio[s] for s in views):
                notes[role] = rows
                programs[role] = sorted({inst.program for inst in pm.instruments})
        if not notes:
            continue
        timing = pretty_midi.PrettyMIDI(str(folder / "midi" / "mix.mid"))
        truth = {
            "dataset": root.name, "composition_id": song, "source": str(folder),
            "split": splits.get(song), "genre": items[0].get("genre"), "duration": duration,
            "anchors": quarter_anchors(timing, duration),
            "time_signatures": [[ts.numerator, ts.denominator, round(ts.time, 6)]
                                for ts in timing.time_signature_changes],
            "notes": notes, "programs": programs, "skipped": skipped, "audio": audio,
        }
        name = prefix + song.removeprefix("lakh_")[:12]
        (CLIPS / name).mkdir(parents=True, exist_ok=True)
        (CLIPS / name / "truth.json").write_text(json.dumps(truth, indent=1, default=_plain), encoding="utf-8")
        written.append(name)
        print(f"[atm] {name} ({truth['split']}): roles={sorted(notes)} skipped={skipped}", flush=True)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="batch folder holding pairs.json, e.g. E:\\atm_training_set\\expanded")
    parser.add_argument("--include-flagged", action="store_true")
    parser.add_argument("--min-notes", type=int, default=20)
    parser.add_argument("--prefix", default=None, help="clip name prefix (default: atm_ for pilot, atmx_ otherwise)")
    args = parser.parse_args(argv)
    prefix = args.prefix or ("atm_" if args.root.name == "pilot" else "atmx_")
    written = import_batch(args.root, args.include_flagged, args.min_notes, prefix)
    print(f"{len(written)} clips indexed under {CLIPS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
