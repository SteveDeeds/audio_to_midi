"""Download a compact aligned MAESTRO subset using HTTP byte ranges."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from remotezip import RemoteZip


ROOT = Path(__file__).parent / "maestro"
CSV_PATH = ROOT / "maestro-v3.0.0.csv"
ARCHIVE_URL = (
    "https://storage.googleapis.com/magentadata/datasets/maestro/"
    "v3.0.0/maestro-v3.0.0.zip"
)


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    selected = [
        min((row for row in rows if row["split"] == split),
            key=lambda row: float(row["duration"]))
        for split in ("train", "validation", "test")
    ]
    members = [
        filename
        for row in selected
        for filename in (row["audio_filename"], row["midi_filename"])
    ]

    with RemoteZip(ARCHIVE_URL) as archive:
        for member in members:
            output = ROOT / member
            if output.is_file():
                print(f"Exists: {output}")
                continue
            print(f"Downloading: {member}")
            output.parent.mkdir(parents=True, exist_ok=True)
            archive_member = f"maestro-v3.0.0/{member}"
            with archive.open(archive_member) as source, output.open("wb") as destination:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)

    manifest = {
        "schema": 1,
        "dataset": "MAESTRO v3.0.0 compact subset",
        "source": "https://magenta.tensorflow.org/datasets/maestro",
        "license": "CC BY-NC-SA 4.0",
        "selection": "shortest piece in each official split",
        "entries": [{
            "split": row["split"],
            "composer": row["canonical_composer"],
            "title": row["canonical_title"],
            "duration": float(row["duration"]),
            "audio": row["audio_filename"],
            "midi": row["midi_filename"],
        } for row in selected],
    }
    manifest_path = ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()