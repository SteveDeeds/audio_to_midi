"""Build a deterministic aligned-stem manifest without application dependencies."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parent / "babyslakh" / "babyslakh_16k"
OUTPUT = Path(__file__).parent / "babyslakh" / "manifest.json"


def split_for_track(track_number: int) -> str:
    if track_number <= 12:
        return "train"
    if track_number <= 16:
        return "validation"
    return "test"


def parse_stems(metadata_path: Path) -> dict[str, dict[str, object]]:
    stems: dict[str, dict[str, object]] = {}
    current: dict[str, object] | None = None
    for raw_line in metadata_path.read_text(encoding="utf-8").splitlines():
        if raw_line.startswith("  S") and raw_line.endswith(":"):
            stem_name = raw_line.strip()[:-1]
            current = {}
            stems[stem_name] = current
            continue
        if current is None or not raw_line.startswith("    ") or ":" not in raw_line:
            continue
        key, value = raw_line.strip().split(":", 1)
        value = value.strip()
        if value in {"true", "false"}:
            current[key] = value == "true"
        elif key == "program_num":
            current[key] = int(value)
        else:
            current[key] = value
    return stems


def main() -> None:
    entries: list[dict[str, object]] = []
    for track_dir in sorted(ROOT.glob("Track*")):
        track_number = int(track_dir.name.removeprefix("Track"))
        stem_metadata = parse_stems(track_dir / "metadata.yaml")
        for stem_name, metadata in sorted(stem_metadata.items()):
            audio_path = track_dir / "stems" / f"{stem_name}.wav"
            midi_path = track_dir / "MIDI" / f"{stem_name}.mid"
            if not audio_path.is_file() or not midi_path.is_file():
                continue
            entries.append({
                "track": track_dir.name,
                "stem": stem_name,
                "split": split_for_track(track_number),
                "audio": str(audio_path.relative_to(ROOT.parent)).replace("\\", "/"),
                "midi": str(midi_path.relative_to(ROOT.parent)).replace("\\", "/"),
                "instrument_class": metadata.get("inst_class", "Unknown"),
                "program_name": metadata.get("midi_program_name", "Unknown"),
                "program_number": metadata.get("program_num"),
                "is_drum": bool(metadata.get("is_drum", False)),
            })

    payload = {
        "schema": 1,
        "dataset": "BabySlakh v2 16k",
        "source": "https://zenodo.org/records/4603870",
        "license": "CC BY 4.0",
        "split_policy": "composition: tracks 1-12 train, 13-16 validation, 17-20 test",
        "entries": entries,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    summary: dict[tuple[str, str], int] = {}
    for entry in entries:
        key = (str(entry["split"]), str(entry["instrument_class"]))
        summary[key] = summary.get(key, 0) + 1
    print(f"Wrote {OUTPUT} with {len(entries)} aligned stems")
    for (split, instrument), count in sorted(summary.items()):
        if instrument in {"Bass", "Piano", "Drums"}:
            print(f"{split:10s} {instrument:8s} {count:2d}")


if __name__ == "__main__":
    main()