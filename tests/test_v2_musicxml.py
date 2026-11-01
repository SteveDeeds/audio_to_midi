from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile

import mido

from audio_to_midi.chords import musicxml as lead_sheet
from audio_to_midi.domain import ChordSegment, Project, Track
from audio_to_midi.midi import export_midi, export_musicxml
from audio_to_midi.musicxml import score_xml
from audio_to_midi.timebase import Grid


class MusicXMLTests(unittest.TestCase):
    def project(self, grid=None):
        project = Project.create("Export", grid or Grid(
            bpm=99, anchors=((0.5, F(0)), (2.5, F(4)), (3.5, F(8))),
        ))
        for role, pitch in (("vocals", 64), ("bass", 40), ("drums", 36)):
            track = Track(role, role, "", 3.5, 8000, "")
            track.notes.author(F(0), F(8), pitch)
            project.tracks.append(track)
        return project

    def assert_opening_tempo(self, root, bpm, displayed=None):
        self.assertEqual(len(root.findall(".//metronome")), 1)
        sounds = root.findall(".//sound[@tempo]")
        self.assertEqual(len(sounds), 1, "no hidden playback tempo changes")
        self.assertAlmostEqual(float(sounds[0].get("tempo")), bpm, places=3)
        first = root.find("part/measure")
        self.assertIsNotNone(first.find("direction/direction-type/metronome"))
        self.assertEqual(
            float(first.findtext("direction/direction-type/metronome/per-minute")),
            round(bpm) if displayed is None else displayed,
        )
        self.assertFalse(root.findall(".//metronome[@print-object='no']"))

    def test_explicit_instrument_names_and_kit_mapping(self):
        root = ET.fromstring(score_xml(self.project()))
        self.assertEqual(
            [p.findtext("part-name") for p in root.findall("part-list/score-part")],
            ["Voice", "Electric Bass", "Drum Kit"],
        )
        bass = root.find("part-list/score-part[@id='P2']")
        self.assertEqual(bass.findtext("score-instrument/instrument-name"), "Electric Bass")
        self.assertEqual(bass.findtext("score-instrument/instrument-sound"), "pluck.bass.electric")
        self.assertEqual(bass.findtext("midi-instrument/midi-program"), "34")
        drums = root.find("part-list/score-part[@id='P3']")
        self.assertEqual(drums.findtext("score-instrument/instrument-name"), "Bass Drum")
        self.assertEqual(drums.findtext("score-instrument/instrument-sound"), "drum.group.set")
        self.assertEqual(drums.findtext("midi-instrument/midi-channel"), "10")
        self.assertEqual(drums.findtext("midi-instrument/midi-unpitched"), "37")

    def test_one_elapsed_time_average_tempo_without_mutating_project(self):
        project = self.project()
        before = deepcopy(project)
        self.assert_opening_tempo(ET.fromstring(score_xml(project)), 160)
        self.assertEqual(project, before)

    def test_single_role_export_uses_whole_song_average(self):
        project = self.project()
        bass = project.tracks[1]
        bass.notes = type(bass.notes)()
        bass.notes.author(F(0), F(1), 40)
        self.assert_opening_tempo(ET.fromstring(score_xml(project, [bass])), 160)

    def test_constant_tempo_and_single_anchor(self):
        for anchors in ((), ((2.0, F(0)),)):
            with self.subTest(anchors=anchors):
                project = self.project(Grid(123.5, offset_seconds=2, anchors=anchors))
                self.assert_opening_tempo(ET.fromstring(score_xml(project)), 123.5)

    def test_compound_meter_keeps_dotted_quarter_mark(self):
        project = self.project(Grid(
            99, 6, 8, anchors=((0.5, F(0)), (2.5, F(3)), (3.5, F(6))),
        ))
        root = ET.fromstring(score_xml(project))
        self.assert_opening_tempo(root, 120, displayed=80)
        self.assertIsNotNone(root.find(".//metronome/beat-unit-dot"))

    def test_duplicate_parts_keep_explicit_numbered_names(self):
        project = self.project()
        for role, pitch in (("bass", 43), ("drums", 38)):
            track = Track(role + "2", role, "", 3.5, 8000, "")
            track.notes.author(F(0), F(8), pitch)
            project.tracks.append(track)
        root = ET.fromstring(score_xml(project))
        self.assertEqual(
            [p.findtext("part-name") for p in root.findall("part-list/score-part")],
            ["Voice", "Electric Bass 1", "Electric Bass 2", "Drum Kit 1", "Drum Kit 2"],
        )

    def test_chord_only_lead_sheet_uses_average(self):
        project = self.project()
        project.chords = [ChordSegment(F(0), F(8), "C")]
        self.assert_opening_tempo(
            ET.fromstring(lead_sheet(project.chords, project.grid, project.name)), 160,
        )
        project.tracks.clear()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chords.musicxml"
            export_musicxml(project, path)
            self.assert_opening_tempo(ET.parse(path).getroot(), 160)

    def test_compressed_export_and_midi_timing_preserved(self):
        project = self.project()
        with tempfile.TemporaryDirectory() as directory:
            score = Path(directory) / "score.mxl"
            export_musicxml(project, score)
            with zipfile.ZipFile(score) as archive:
                self.assert_opening_tempo(ET.fromstring(archive.read("score.musicxml")), 160)
            midi_path = Path(directory) / "song.mid"
            export_midi(project, midi_path)
            mid = mido.MidiFile(midi_path)
            tick, tempos = 0, []
            for message in mid.tracks[0]:
                tick += message.time
                if message.type == "set_tempo":
                    tempos.append((tick, round(mido.tempo2bpm(message.tempo))))
            self.assertEqual(tempos, [(0, 120), (4 * mid.ticks_per_beat, 240)])


if __name__ == "__main__":
    unittest.main()
