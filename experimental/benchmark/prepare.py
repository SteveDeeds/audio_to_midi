"""Build benchmark clips: MIDI ground truth, soundfont/rendered stems, mix and Demucs stems.

    python -m experimental.benchmark.prepare --lakh 30 --slakh 20 --groove 15

Each clip folder under ``data/clips`` holds ``truth.json`` (``[onset, offset, pitch, velocity]``
note rows in seconds per role, quarter-note timing anchors), ``mix.wav`` (stereo, 44.1 kHz), ``clean/<role>.wav``
(isolated rendered stems, mono) and ``demucs/<stem>.wav`` (htdemucs_6s, mono).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import pretty_midi
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data"
CLIPS = DATA / "clips"
FLUIDSYNTH = next((ROOT / "tools" / "fluidsynth").rglob("fluidsynth.exe"), None) if (ROOT / "tools" / "fluidsynth").exists() else None
SOUNDFONT = Path(r"C:\Program Files\JJazzLab\jjazzlab\modules\soundfont\JJazzLab-SoundFont.sf2")
SR = 44100
CLIP_SECONDS = 60.0
ROLES = ("piano", "guitar", "bass", "drums")
DEMUCS_KEEP = ("drums", "bass", "guitar", "piano", "other")
MIN_NOTES = {"piano": 40, "guitar": 40, "bass": 40, "drums": 80}
# Roland TD-11 articulations in Groove MIDI that the GM kit does not define.
GROOVE_REMAP = {22: 42, 26: 46, 58: 43}


def role_of(instrument: pretty_midi.Instrument) -> str | None:
    if instrument.is_drum:
        return "drums"
    program = instrument.program
    if program <= 7:
        return "piano"
    if 24 <= program <= 31:
        return "guitar"
    if 32 <= program <= 39:
        return "bass"
    return None


def crop(instrument: pretty_midi.Instrument, start: float, end: float) -> pretty_midi.Instrument:
    """Notes starting inside [start, end), shifted to 0; controls/bends kept for rendering."""
    out = pretty_midi.Instrument(instrument.program, instrument.is_drum, instrument.name)
    for note in instrument.notes:
        if start <= note.start < end:
            out.notes.append(pretty_midi.Note(
                note.velocity, note.pitch, note.start - start, min(note.end, end + 2.0) - start,
            ))
    for cc in instrument.control_changes:
        if cc.time < end:
            out.control_changes.append(pretty_midi.ControlChange(cc.number, cc.value, max(0.0, cc.time - start)))
    for bend in instrument.pitch_bends:
        if bend.time < end:
            out.pitch_bends.append(pretty_midi.PitchBend(bend.pitch, max(0.0, bend.time - start)))
    # Controls before the clip collapse to time 0; keep only the latest value per controller.
    latest: dict[int, pretty_midi.ControlChange] = {}
    rest = []
    for cc in sorted(out.control_changes, key=lambda c: c.time):
        if cc.time == 0.0:
            latest[cc.number] = cc
        else:
            rest.append(cc)
    out.control_changes = list(latest.values()) + rest
    bends = sorted(out.pitch_bends, key=lambda b: b.time)
    zero = [b for b in bends if b.time == 0.0]
    out.pitch_bends = zero[-1:] + [b for b in bends if b.time > 0.0]
    return out


def anchors_from(pm: pretty_midi.PrettyMIDI, start: float, seconds: float) -> list[list]:
    """(seconds, quarter) pairs relative to the clip-start downbeat, spanning the clip."""
    beats = np.asarray(pm.get_beats(), dtype=float)
    i0 = int(np.argmin(np.abs(beats - start)))
    result = []
    for i in range(i0, len(beats)):
        t = float(beats[i] - start)
        result.append([t, i - i0])
        if t > seconds + 2.0:
            break
    if len(result) < 2:
        raise ValueError("clip has no beat grid")
    if result[-1][0] < seconds:
        # Extend at the last beat period so the anchors cover the whole clip.
        period = result[-1][0] - result[-2][0]
        while result[-1][0] < seconds + 2.0:
            result.append([result[-1][0] + period, result[-1][1] + 1])
    return result


def four_four(pm: pretty_midi.PrettyMIDI) -> bool:
    return all((ts.numerator, ts.denominator) == (4, 4) for ts in pm.time_signature_changes)


def clip_start(pm: pretty_midi.PrettyMIDI, fraction: float = 0.3) -> float | None:
    downbeats = pm.get_downbeats()
    end = pm.get_end_time()
    target = fraction * end
    for t in downbeats:
        if t >= target and t + CLIP_SECONDS <= end:
            return float(t)
    return None


def render(instruments: list[pretty_midi.Instrument]) -> np.ndarray:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.instruments.extend(instruments)
    with tempfile.TemporaryDirectory() as tmp:
        mid, wav = Path(tmp) / "x.mid", Path(tmp) / "x.wav"
        pm.write(str(mid))
        subprocess.run(
            [str(FLUIDSYNTH), "-ni", "-q", "-g", "0.5", "-r", str(SR), "-O", "s16", "-T", "wav",
             "-F", str(wav), str(SOUNDFONT), str(mid)],
            check=True, capture_output=True, timeout=600,
        )
        audio, rate = sf.read(wav, dtype="float32", always_2d=True)
    if rate != SR:
        raise RuntimeError(f"fluidsynth rendered {rate} Hz")
    return audio


def fit(audio: np.ndarray, frames: int) -> np.ndarray:
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    return audio[:frames] if len(audio) >= frames else np.pad(audio, ((0, frames - len(audio)), (0, 0)))


def write_clip(folder: Path, stems: dict[str, np.ndarray], truth: dict) -> None:
    frames = int(round(CLIP_SECONDS * SR))
    stems = {role: fit(a, frames) for role, a in stems.items()}
    mix = sum(stems.values())
    scale = 0.9 / max(float(np.max(np.abs(mix))), 1e-6)
    (folder / "clean").mkdir(parents=True, exist_ok=True)
    for role, audio in stems.items():
        sf.write(folder / "clean" / f"{role}.wav", (audio * scale).mean(axis=1), SR, subtype="PCM_16")
    sf.write(folder / "mix.wav", mix * scale, SR, subtype="PCM_16")
    (folder / "truth.json").write_text(json.dumps(truth, indent=1, default=_plain), encoding="utf-8")


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def notes_of(instruments: list[pretty_midi.Instrument]) -> list[list]:
    """[onset, offset, pitch, velocity] rows in clip seconds."""
    rows = [[round(n.start, 6), round(min(n.end, CLIP_SECONDS), 6), int(n.pitch), int(n.velocity)]
            for inst in instruments for n in inst.notes if n.start < CLIP_SECONDS]
    return sorted(rows)


def refresh_lakh_truth() -> None:
    """Recompute note rows of existing Lakh clips from their source MIDI (no re-render)."""
    for folder in sorted(CLIPS.glob("lakh_*")):
        path = folder / "truth.json"
        if not path.exists():
            continue
        truth = json.loads(path.read_text(encoding="utf-8"))
        pm = pretty_midi.PrettyMIDI(str(DATA / truth["source"]))
        start = float(truth["clip_start"])
        for role in truth["notes"]:
            insts = [crop(i, start, start + CLIP_SECONDS) for i in pm.instruments if role_of(i) == role]
            truth["notes"][role] = notes_of([i for i in insts if i.notes])
        path.write_text(json.dumps(truth, indent=1, default=_plain), encoding="utf-8")
        print(f"[truth] {folder.name} refreshed", flush=True)


# -- Lakh MIDI (clean subset) rendered with the JJazzLab soundfont -------------------------------
def prepare_lakh(count: int, seed: int) -> None:
    from core.adtof_drums import drum_class

    files = []
    for dirpath, _, names in os.walk(DATA / "clean_midi"):
        files += [Path(dirpath) / n for n in names if n.lower().endswith(".mid")]
    files.sort()
    random.Random(seed).shuffle(files)
    existing = {p.name for p in CLIPS.glob("lakh_*") if (p / "truth.json").exists()}
    artists = {json.loads((CLIPS / p / "truth.json").read_text())["artist"] for p in existing}
    made = len(existing)
    for path in files:
        if made >= count:
            break
        artist = path.parent.name
        if artist in artists:
            continue
        try:
            pm = pretty_midi.PrettyMIDI(str(path))
            if not four_four(pm) or pm.get_end_time() < 100:
                continue
            start = clip_start(pm)
            if start is None:
                continue
            by_role: dict[str, list] = {}
            for inst in pm.instruments:
                role = role_of(inst)
                if role:
                    cropped = crop(inst, start, start + CLIP_SECONDS)
                    if cropped.notes:
                        by_role.setdefault(role, []).append(cropped)
            counts = {r: sum(len(i.notes) for i in v) for r, v in by_role.items()}
            if counts.get("bass", 0) < MIN_NOTES["bass"] or counts.get("drums", 0) < MIN_NOTES["drums"]:
                continue
            if max(counts.get("piano", 0), counts.get("guitar", 0)) < MIN_NOTES["piano"]:
                continue
            drums = [n.pitch for i in by_role["drums"] for n in i.notes]
            if np.mean([drum_class(p) is not None for p in drums]) < 0.8:
                continue
            by_role = {r: v for r, v in by_role.items() if counts[r] >= MIN_NOTES[r]}
            anchors = anchors_from(pm, start, CLIP_SECONDS)
        except Exception as exc:  # noqa: BLE001 - many Lakh files are malformed; skip them
            if not isinstance(exc, (ValueError, KeyError, IndexError, EOFError, OSError)):
                print(f"[lakh] skip {path.name}: {exc!r}", flush=True)
            continue
        name = f"lakh_{made:02d}"
        try:
            stems = {role: render(insts) for role, insts in by_role.items()}
            if any(np.max(np.abs(a)) < 1e-4 for a in stems.values()):
                continue
            truth = {
                "dataset": "lakh", "source": str(path.relative_to(DATA)), "artist": artist,
                "clip_start": start, "duration": CLIP_SECONDS, "anchors": anchors,
                "notes": {role: notes_of(insts) for role, insts in by_role.items()},
                "programs": {role: [i.program for i in insts] for role, insts in by_role.items()},
            }
            write_clip(CLIPS / name, stems, truth)
        except Exception:  # noqa: BLE001 - report rendering/writing failures loudly, then move on
            print(f"[lakh] failed {path}:\n{traceback.format_exc()}", flush=True)
            continue
        artists.add(artist)
        made += 1
        print(f"[lakh] {name}: {artist} / {path.stem} roles={sorted(by_role)}", flush=True)


# -- BabySlakh: professionally rendered stems, cropped and mixed from the four roles ----------
def prepare_slakh(count: int) -> None:
    import librosa
    import yaml

    root = DATA / "babyslakh_16k"
    tracks = sorted(p for p in root.glob("Track*") if p.is_dir())
    made = 0
    for track in tracks:
        if made >= count:
            break
        name = f"slakh_{track.name}"
        if (CLIPS / name / "truth.json").exists():
            made += 1
            continue
        try:
            meta = yaml.safe_load((track / "metadata.yaml").read_text(encoding="utf-8"))
            pm = pretty_midi.PrettyMIDI(str(track / "all_src.mid"))
            if not four_four(pm):
                continue
            start = clip_start(pm)
            if start is None:
                continue
            stems, notes, programs = {}, {}, {}
            for stem_id, info in meta["stems"].items():
                wav, mid = track / "stems" / f"{stem_id}.wav", track / "MIDI" / f"{stem_id}.mid"
                if not (info.get("audio_rendered", True) and wav.exists() and mid.exists()):
                    continue
                stem_pm = pretty_midi.PrettyMIDI(str(mid))
                if not stem_pm.instruments:
                    continue
                inst = stem_pm.instruments[0]
                inst.is_drum = bool(info.get("is_drum", inst.is_drum))
                if not inst.is_drum:
                    inst.program = int(info.get("program_num", inst.program))
                role = role_of(inst)
                if role is None:
                    continue
                cropped = crop(inst, start, start + CLIP_SECONDS)
                audio, rate = sf.read(wav, dtype="float32", always_2d=True)
                audio = audio[int(start * rate):int((start + CLIP_SECONDS) * rate)].mean(axis=1)
                audio = librosa.resample(audio, orig_sr=rate, target_sr=SR)
                stems[role] = stems[role] + fit(audio, len(audio)) if role in stems else fit(audio, len(audio))
                notes.setdefault(role, []).append(cropped)
                programs.setdefault(role, []).append(inst.program)
            counts = {r: sum(len(i.notes) for i in v) for r, v in notes.items()}
            keep = [r for r in stems if counts.get(r, 0) >= MIN_NOTES[r]]
            if not keep:
                continue
            frames = int(round(CLIP_SECONDS * SR))
            stems = {r: fit(stems[r], frames) for r in stems}  # mix includes every role stem
            truth = {
                "dataset": "slakh", "source": track.name, "artist": track.name,
                "clip_start": start, "duration": CLIP_SECONDS, "anchors": anchors_from(pm, start, CLIP_SECONDS),
                "notes": {r: notes_of(notes[r]) for r in keep},
                "programs": {r: programs[r] for r in keep},
            }
            write_clip(CLIPS / name, stems, truth)
            made += 1
            print(f"[slakh] {name}: roles={keep}", flush=True)
        except Exception:  # noqa: BLE001
            print(f"[slakh] skip {track.name}:\n{traceback.format_exc()}", flush=True)


# -- Groove MIDI: human drum performances rendered with the soundfont --------------------------
def prepare_groove(count: int, seed: int) -> None:
    root = DATA / "groove"
    with open(root / "info.csv", newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle)
                if r["beat_type"] == "beat" and r["time_signature"] == "4-4" and float(r["duration"]) >= 30]
    rows.sort(key=lambda r: (r["split"] != "test", r["id"]))
    random.Random(seed).shuffle(rows)
    rows.sort(key=lambda r: r["split"] != "test")
    seen: set[tuple[str, str]] = set()
    made = len(list(CLIPS.glob("groove_*")))
    for row in rows:
        if made >= count:
            break
        key = (row["drummer"], row["style"].split("/")[0])
        if key in seen:
            continue
        name = f"groove_{made:02d}"
        try:
            pm = pretty_midi.PrettyMIDI(str(root / row["midi_filename"]))
            inst = pm.instruments[0]
            inst.is_drum = True
            for note in inst.notes:
                note.pitch = GROOVE_REMAP.get(note.pitch, note.pitch)
            seconds = min(CLIP_SECONDS, float(row["duration"]))
            cropped = crop(inst, 0.0, seconds)
            truth = {
                "dataset": "groove", "source": row["midi_filename"], "artist": row["drummer"],
                "style": row["style"], "bpm": float(row["bpm"]), "clip_start": 0.0,
                "duration": seconds, "anchors": anchors_from(pm, 0.0, seconds),
                "notes": {"drums": notes_of([cropped])}, "programs": {"drums": [0]},
            }
            audio = render([cropped])
            write_groove(CLIPS / name, audio, truth, seconds)
            seen.add(key)
            made += 1
            print(f"[groove] {name}: {row['midi_filename']}", flush=True)
        except Exception:  # noqa: BLE001
            print(f"[groove] skip {row['midi_filename']}:\n{traceback.format_exc()}", flush=True)


def write_groove(folder: Path, audio: np.ndarray, truth: dict, seconds: float) -> None:
    frames = int(round(seconds * SR))
    audio = fit(audio, frames)
    scale = 0.9 / max(float(np.max(np.abs(audio))), 1e-6)
    (folder / "clean").mkdir(parents=True, exist_ok=True)
    sf.write(folder / "clean" / "drums.wav", (audio * scale).mean(axis=1), SR, subtype="PCM_16")
    sf.write(folder / "mix.wav", audio * scale, SR, subtype="PCM_16")
    (folder / "truth.json").write_text(json.dumps(truth, indent=1, default=_plain), encoding="utf-8")


# -- Demucs, exactly as the app runs it ---------------------------------------------------------
def separate(folder: Path) -> float:
    target = folder / "demucs"
    if all((target / f"{s}.wav").exists() for s in DEMUCS_KEEP):
        return 0.0
    began = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "audio_to_midi.separation_worker", str(folder / "mix.wav"), tmp],
            check=True, capture_output=True, env=env, timeout=3600,
        )
        target.mkdir(exist_ok=True)
        for stem in DEMUCS_KEEP:
            audio, rate = sf.read(Path(tmp) / f"{stem}.wav", dtype="float32", always_2d=True)
            sf.write(target / f"{stem}.wav", audio.mean(axis=1), rate, subtype="PCM_16")
    return time.perf_counter() - began


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lakh", type=int, default=30)
    parser.add_argument("--slakh", type=int, default=20)
    parser.add_argument("--groove", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-demucs", action="store_true")
    parser.add_argument("--refresh-truth", action="store_true",
                        help="only rewrite truth.json notes of existing Lakh clips (adds velocity)")
    args = parser.parse_args(argv)
    if args.refresh_truth:
        refresh_lakh_truth()
        return 0
    if FLUIDSYNTH is None or not SOUNDFONT.exists():
        raise SystemExit("FluidSynth (tools/fluidsynth) and the JJazzLab soundfont are required.")
    CLIPS.mkdir(parents=True, exist_ok=True)
    if args.groove:
        prepare_groove(args.groove, args.seed)
    if args.lakh:
        prepare_lakh(args.lakh, args.seed)
    if args.slakh and (DATA / "babyslakh_16k").exists():
        prepare_slakh(args.slakh)
    if not args.no_demucs:
        for folder in sorted(CLIPS.iterdir()):
            if folder.name.startswith("groove_") or not (folder / "truth.json").exists():
                continue
            try:
                seconds = separate(folder)
                if seconds:
                    print(f"[demucs] {folder.name}: {seconds:.0f} s", flush=True)
            except Exception:  # noqa: BLE001
                print(f"[demucs] failed {folder.name}:\n{traceback.format_exc()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
