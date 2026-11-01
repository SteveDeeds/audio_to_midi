"""Local-only browser GUI for cached vocal note/onset parameter experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import logging
from pathlib import Path
import re
import secrets
from urllib.parse import urlsplit

import numpy as np
import soundfile as sf

from experimental.tuning import (
    GUI_CONTROL_SPECS, KEYS, METHODS, Evidence, TuningParameters, load_evidence,
    prepare_evidence, retune, tuning_defaults,
)
from experimental.vocal_midi import export_midi, render_plots, stereo_comparison, synthesize, write_csv

ROOT = Path(__file__).parent
DEFAULT_CACHE = ROOT / "outputs" / "tuner_cache_39_to_end"


def byte_range(header: str | None, size: int) -> tuple[int, int, int]:
    if header is None:
        return 0, size - 1, 200
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    if match is None or size <= 0 or not any(match.groups()):
        raise ValueError("Unsupported byte range.")
    left, right = match.groups()
    if not left:
        count = int(right)
        if count <= 0:
            raise ValueError("Invalid suffix range.")
        return max(0, size - count), size - 1, 206
    start, end = int(left), min(int(right), size - 1) if right else size - 1
    if start >= size or end < start:
        raise ValueError("Byte range is outside the file.")
    return start, end, 206


class TunerServer(HTTPServer):
    def __init__(self, address: tuple[str, int], evidence: Evidence):
        self.evidence = evidence
        self.work = evidence.directory / "tuning_results"
        self.work.mkdir(exist_ok=True)
        self.token = secrets.token_urlsafe(32)
        self.defaults = tuning_defaults()
        super().__init__(address, TunerHandler)


class TunerHandler(SimpleHTTPRequestHandler):
    server: TunerServer

    def json_response(self, data: dict, status: int = 200) -> None:
        payload = json.dumps(data, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/state":
            result, _, _ = retune(self.server.evidence, self.server.defaults)
            self.json_response({
                "token": self.server.token, "metadata": self.server.evidence.metadata,
                "methods": METHODS, "keys": KEYS, "controls": GUI_CONTROL_SPECS,
                "defaults": asdict(self.server.defaults),
                "baseline": asdict(TuningParameters()), "result": result,
            })
            return
        if path == "/":
            file = ROOT / "tuner.html"
        elif path == "/assets/cqt.png":
            file = self.server.evidence.directory / "cqt_background.png"
        elif path.startswith("/files/"):
            file = (self.server.work / path.removeprefix("/files/")).resolve()
            if not file.is_relative_to(self.server.work.resolve()):
                self.send_error(404, "Unknown result file.")
                return
        else:
            self.send_error(404, "Unknown tuner resource.")
            return
        if not file.is_file():
            self.send_error(404, "Result file is missing.")
            return
        size = file.stat().st_size
        try:
            start, end, status = byte_range(self.headers.get("Range"), size)
        except ValueError:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(status)
        self.send_header("Content-Type", self.guess_type(str(file)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with file.open("rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            try:
                while remaining > 0:
                    chunk = handle.read(min(65536, remaining))
                    if not chunk:
                        raise OSError("Result file ended before its advertised byte range.")
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                logging.debug("Browser closed a result/media stream.")

    def do_POST(self) -> None:
        route = urlsplit(self.path).path
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 8192:
                raise ValueError("Expected a parameter request of at most 8192 bytes.")
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Expected application/json parameters.")
            payload = self.rfile.read(size)
            if self.headers.get("X-Tuner-Token") != self.server.token:
                self.json_response({"error": "Missing or invalid local tuner token."}, 403)
                return
            if route not in ("/api/tune", "/api/preview", "/api/export"):
                self.json_response({"error": "Unknown tuner action."}, 404)
                return
            values = json.loads(payload)
            parameters = TuningParameters.from_dict(values)
            result, notes, detector = retune(self.server.evidence, parameters)
            if route == "/api/tune":
                self.json_response(result)
                return
            if not notes:
                raise ValueError("No playable notes at these thresholds. Lower the gates before preview/export.")
            evidence = self.server.evidence
            sr = evidence.metadata["sample_rate"]
            synth = synthesize(notes, sr, len(evidence.arrays["vocal"]))
            stereo, gains = stereo_comparison(evidence.arrays["vocal"], synth)
            if route == "/api/preview":
                name = f"preview_{result['parameter_id']}.flac"
                sf.write(self.server.work / name, stereo, sr, subtype="PCM_24")
                self.json_response({"url": f"/files/{name}", "parameter_id": result["parameter_id"],
                                    "duration_seconds": evidence.duration, "gains": gains})
                return
            folder = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + result["parameter_id"]
            output = self.server.work / folder
            output.mkdir()
            sf.write(output / "AB.wav", stereo, sr, subtype="PCM_24")
            sf.write(output / "AB.flac", stereo, sr, subtype="PCM_24")
            export_midi(notes, output / "notes.mid", 150)
            write_csv(output / "notes.csv", list(asdict(notes[0])),
                      [asdict(note) for note in notes])
            write_csv(output / "onsets.csv", ["crop_seconds", "source_seconds", "evidence"],
                      [{"crop_seconds": float(t), "source_seconds": float(t + evidence.metadata["start_seconds"]),
                        "evidence": "SuperFlux + energy recovery" if t in result["energy_onsets"] else "detector"}
                       for t in detector.times])
            provenance = {
                "parameters": asdict(parameters), "analysis": evidence.metadata,
                "stats": result["stats"], "comparison_audio": gains,
                "notes": [asdict(note) for note in notes], "bpm": 150, "ticks_per_beat": 9600,
                "certainty_note": "CREPE periodicity and pitch agreement are not calibrated accuracy probabilities.",
                "articulation_note": "Syllable mode uses a minimum 10ms MIDI/synth reattack gap, not an estimate of acoustic silence; pitch-only changes connect.",
            }
            (output / "settings.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
            pitch = np.asarray([np.nan if p is None else p for p in result["pitch"]])
            render_plots(
                output, evidence.arrays["cqt_db"], evidence.arrays["cqt_times"],
                evidence.arrays["times"], pitch, evidence.arrays["db"],
                {parameters.method: detector}, {parameters.method: notes},
                evidence.duration, evidence.metadata["start_seconds"],
            )
            files = ["notes.mid", "AB.wav", "AB.flac", "notes.csv", "onsets.csv",
                     "settings.json", f"{parameters.method}_cqt.png"]
            self.json_response({"directory": str(output.resolve()),
                                "parameter_id": result["parameter_id"],
                                "files": {name: f"/files/{folder}/{name}" for name in files}})
        except ValueError as exc:
            logging.warning("Invalid tuner request: %s", exc)
            self.json_response({"error": str(exc)}, 400)
        except OSError as exc:
            logging.exception("Tuner preview/export failed")
            self.json_response({"error": f"Could not write tuning result: {exc}"}, 500)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--baseline", type=Path, default=ROOT / "outputs" / "vocals_39_to_end" / "report.json",
                        help="Reviewed baseline whose voiced pitch frames should remain unchanged.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    evidence = (
        prepare_evidence(args.cache, baseline=args.baseline)
        if args.rebuild or not (args.cache / "evidence.json").exists()
        else load_evidence(args.cache)
    )
    if args.prepare_only:
        return
    with TunerServer(("127.0.0.1", args.port), evidence) as server:
        print(f"Vocal tuning GUI ready: http://127.0.0.1:{server.server_port}", flush=True)
        print(f"Exports: {server.work.resolve()}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
