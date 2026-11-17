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
import hashlib
import json
from pathlib import Path
import sys

import pretty_midi
import soundfile as sf

from core.midi_export import load_midi_notes
from experimental.benchmark.prepare import CLIPS, _plain

ROLES = ("bass", "drums", "piano", "guitar")
FAMILY_ROUTING_POLICY = "whole-family-energy-v1"


def balanced_index(root: Path) -> list[dict]:
    """Read the balanced batch without touching legacy clips or learned caches."""
    split = json.loads((root / "splits.json").read_text(encoding="utf-8"))["composition_to_split"]
    pairs = json.loads((root / "pairs.json").read_text(encoding="utf-8"))
    routed_pairs = json.loads((root / "training_pairs.json").read_text(encoding="utf-8"))
    routed = {(pair["composition_id"], pair["stem"]): pair for pair in routed_pairs}
    if len(routed) != len(routed_pairs):
        raise ValueError("Repeated routed stem targets in the balanced batch")
    by_song: dict[str, list[dict]] = {}
    for pair in pairs:
        by_song.setdefault(pair["composition_id"], []).append(pair)
    if set(by_song) != set(split):
        raise ValueError("Balanced batch and split composition IDs differ")
    groups: dict[str, str] = {}
    records = []
    for song, items in sorted(by_song.items()):
        assigned_split = split[song]
        if assigned_split not in ("train", "validation", "test"):
            raise ValueError(f"{song}: invalid split {assigned_split}")
        source_groups = {pair["source_group"] for pair in items}
        if len(source_groups) != 1:
            raise ValueError(f"{song}: inconsistent source groups")
        group = source_groups.pop()
        if groups.setdefault(group, assigned_split) != assigned_split:
            raise ValueError(f"Source group {group} leaks across splits")
        directory = root / song
        route = json.loads((directory / "stem-routing.json").read_text(encoding="utf-8"))
        if route.get("routing_policy") != FAMILY_ROUTING_POLICY:
            raise ValueError(f"{song}: reroute the batch using whole-family routing first")
        destinations = route["family_destinations"]
        families = {pair["instrument"] for pair in items}
        if set(destinations) != families:
            raise ValueError(f"{song}: family routing does not cover the family labels")
        for note in route["notes"]:
            if destinations[note["family"]] != note["primary_stem"]:
                raise ValueError(f"{song}: an instrument family is split across stems")
        notes, audio, flagged, legacy_flags = {}, {}, {}, {}
        for pair in items:
            role = pair["instrument"]
            if role not in ROLES or role in notes:
                raise ValueError(f"{song}: unknown or repeated family {role}")
            notes[role] = sorted(
                [start, end, pitch, velocity]
                for start, end, pitch, velocity, _ in load_midi_notes(str(root / pair["midi"]))
            )
            audio[role] = str((root / pair["clean_audio"]).resolve())
            flagged[role] = bool(routed[(song, destinations[role])]["flagged"])
            legacy_flags[role] = bool(pair["flagged"])
        other_families = sorted(name for name, stem in destinations.items() if stem == "other")
        other = routed.get((song, "other"))
        if bool(other_families) != (other is not None):
            raise ValueError(f"{song}: other-stem targets disagree with family routing")
        if "drums" in other_families:
            raise ValueError(f"{song}: mixed percussion in other requires a separate drum-class target")
        notes["other"] = []
        flagged["other"] = False
        if other is not None:
            notes["other"] = sorted(
                [start, end, pitch, velocity]
                for start, end, pitch, velocity, _ in load_midi_notes(str(root / other["midi"]))
            )
            if len(notes["other"]) != other["note_count"]:
                raise ValueError(f"{song}: other-stem MIDI note count differs from the routed index")
            if other_families != sorted(other["instrument_families"]):
                raise ValueError(f"{song}: other-stem instrument families disagree with routing")
            flagged["other"] = bool(other["flagged"])
        stems = {name: str((directory / "stems" / f"{name}.wav").resolve())
                 for name in ("bass", "drums", "piano", "guitar", "other", "vocals")}
        mix = directory / "mix.wav"
        for path in [str(mix), *audio.values(), *stems.values()]:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        records.append({
            "clip": "balanced_" + hashlib.sha256(song.encode()).hexdigest()[:20],
            "composition_id": song, "source_group": group,
            "source_composition_id": items[0]["source_composition_id"],
            "split": assigned_split, "duration": sf.info(mix).duration,
            "notes": notes, "family_destinations": destinations, "flagged": flagged,
            "legacy_family_flagged": legacy_flags,
            "other_families": other_families,
            "audio": {"mix": str(mix.resolve()), "stems": stems, "clean": audio},
            "routing_policy": FAMILY_ROUTING_POLICY,
        })
    if len({record["clip"] for record in records}) != len(records):
        raise ValueError("Balanced clip IDs collided")
    return records


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
