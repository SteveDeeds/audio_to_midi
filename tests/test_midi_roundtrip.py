import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import mido

from audio_to_midi.domain import Project, Track
from audio_to_midi.midi import export_midi
from audio_to_midi.timebase import Grid
from core.midi_export import load_midi_notes


class TestMidiRoundTrip(unittest.TestCase):
    def test_project_export_then_load_preserves_touching_notes_and_velocity(self):
        project = Project.create("Round trip", Grid(120))
        track = Track("piano", "piano", "", 1.0, 8000, "basic_pitch")
        project.tracks.append(track)
        track.notes.author(Fraction(0), Fraction(1), 60, 80)
        track.notes.author(Fraction(1), Fraction(2), 60, 100)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roundtrip.mid"
            export_midi(project, path)
            rows = load_midi_notes(str(path))
        self.assertEqual(len(rows), 2)
        self.assertEqual([(row[2], row[3]) for row in rows], [(60, 80), (60, 100)])
        for row, start, end in zip(rows, (0.0, 0.5), (0.5, 1.0)):
            self.assertAlmostEqual(row[0], start)
            self.assertAlmostEqual(row[1], end)

    def test_reader_honors_tempo_changes_velocity_and_channel(self):
        midi = mido.MidiFile(ticks_per_beat=480)
        midi.tracks.append(mido.MidiTrack([
            mido.MetaMessage("set_tempo", tempo=500000),
            mido.Message("note_on", channel=2, note=60, velocity=70),
            mido.MetaMessage("set_tempo", tempo=1000000, time=480),
            mido.Message("note_off", channel=2, note=60, time=480),
        ]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tempo.mid"
            midi.save(path)
            rows = load_midi_notes(str(path))
        self.assertEqual(rows, [(0.0, 1.5, 60, 70, 2)])


if __name__ == "__main__":
    unittest.main()
