"""Beat-native MIDI interchange with a shared tempo/meter conductor."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import os
import tempfile

import mido

from audio_to_midi.domain import Note, Project, Track
from core.midi_export import load_midi_notes
from audio_to_midi.application import record_onset


PPQ = 960


def _validate_export_path(project: Project, path: Path) -> None:
    inputs = {Path(t.audio_path).resolve() for t in project.tracks if t.audio_path}
    for source in (project.mix_path, project.stages.get("timing", {}).get("source")):
        if source:
            inputs.add(Path(source).resolve())
    preparation = project.stages.get("preparation")
    if preparation:
        from audio_to_midi.pipeline import ArtifactStore

        store = ArtifactStore(preparation["directory"])
        inputs.add(store.path)
        for revision in store.manifest["revisions"]:
            inputs.update(Path(source).resolve() for source in revision["inputs"])
            inputs.update((store.root / output["path"]).resolve() for output in revision["outputs"])
    if path.resolve() in inputs:
        raise ValueError("Export must not overwrite project audio or preparation artifacts.")


def import_notes(path: Path, project: Project, track: Track) -> int:
    if track.role == "other":
        raise ValueError("Other is harmony context only.")
    rows = load_midi_notes(str(path))
    notes = [
        Note(
            f"import:{i}:{track.id}",
            Fraction(str(start * project.grid.bpm / 60)).limit_denominator(1000000),
            Fraction(str(end * project.grid.bpm / 60)).limit_denominator(1000000),
            pitch, velocity, "imported",
        )
        for i, (start, end, pitch, velocity, _) in enumerate(rows)
    ]
    for note in notes:
        track.notes.author(note.start, note.end, note.pitch, note.velocity)
    if notes:
        boundaries = {project.grid.snap(beat) for n in notes for beat in (n.start, n.end)}
        for beat in sorted(boundaries):
            record_onset(track, beat, True)
        cursor = Fraction(0)
        duration = max(Fraction(0), project.grid.seconds_to_beat(track.duration))
        while cursor < duration:
            if cursor not in boundaries and not any(n.start <= cursor < n.end for n in notes):
                record_onset(track, cursor, False)
            cursor += project.grid.subdivision
    track.settings["imported_midi"] = str(path.resolve())
    return len(notes)


def export_midi(
    project: Project, path: Path, tracks: list[Track] | None = None,
    include_chords: bool = True,
) -> None:
    _validate_export_path(project, path)
    mid = mido.MidiFile(ticks_per_beat=PPQ)
    conductor = mido.MidiTrack()
    mid.tracks.append(conductor)
    conductor.extend([
        mido.MetaMessage("track_name", name=project.name),
        mido.MetaMessage(
            "time_signature", numerator=project.grid.numerator,
            denominator=project.grid.denominator,
        ),
    ])
    tempos = project.grid.tempos()
    first = max((i for i, (q, _) in enumerate(tempos) if q <= 0), default=0)
    tick = 0
    for i, (q, bpm) in enumerate(tempos[first:]):
        at = 0 if i == 0 else round(q * PPQ)
        conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=at - tick))
        tick = at
    channels = iter([i for i in range(16) if i != 9])
    for track in project.tracks if tracks is None else tracks:
        if track.role == "other":
            continue
        channel = 9 if track.role == "drums" else next(channels)
        _append_track(mid, track.role, track.notes.resolved(), channel)
    if include_chords and project.chords:
        from audio_to_midi.chords import segments_to_notes

        voiced = [
            Note(f"chord:{i}", start, end, pitch, velocity, "chord")
            for i, (start, end, pitch, velocity) in enumerate(segments_to_notes(project.chords))
        ]
        _append_track(mid, "chords", voiced, next(channels))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".mid")
    os.close(fd)
    try:
        mid.save(temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def export_musicxml(
    project: Project, path: Path, tracks: list[Track] | None = None,
    include_chords: bool = True,
) -> None:
    """Score for notation programs: one staff per track with chord symbols.

    With no transcribed notes among the chosen tracks, writes the chord lead
    sheet instead. A .mxl path is written compressed.
    """
    from audio_to_midi import musicxml

    _validate_export_path(project, path)
    chosen = project.tracks if tracks is None else tracks
    if any(t.role != "other" and t.notes.resolved() for t in chosen):
        text = musicxml.score_xml(project, chosen, include_chords)
    elif include_chords and project.chords:
        from audio_to_midi.chords import musicxml as lead_sheet

        text = lead_sheet(project.chords, project.grid, project.name, project.key)
    else:
        raise ValueError("There are no notes or chords to export.")
    musicxml.write(path, text)


def _append_track(mid: mido.MidiFile, name: str, notes: list[Note], channel: int) -> None:
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=name))
    mid.tracks.append(track)
    events = []
    for note in notes:
        start = round(note.start * PPQ)
        end = max(start + 1, round(note.end * PPQ))
        events.extend([
            (start, 1, note.pitch, note.velocity),
            (end, 0, note.pitch, 0),
        ])
    cursor = 0
    for tick, on, pitch, velocity in sorted(events):
        track.append(mido.Message(
            "note_on" if on else "note_off", channel=channel,
            note=pitch, velocity=velocity, time=tick - cursor,
        ))
        cursor = tick
    track.append(mido.MetaMessage("end_of_track"))
