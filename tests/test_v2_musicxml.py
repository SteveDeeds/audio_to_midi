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
        self.assertEqual(drums.findtext("score-instrument/instrument-name"), "Drum Kit")
        self.assertEqual(drums.findtext("score-instrument/instrument-sound"), "drum.group.set")
        self.assertEqual(drums.findtext("midi-instrument/midi-channel"), "10")
        self.assertEqual(drums.findtext("midi-instrument/midi-unpitched"), "37")

    def test_drum_kit_identity_precedes_piece_names_without_changing_hit_mapping(self):
        for pitches in ((36, 38, 42), (38, 42), (42,)):
            with self.subTest(pitches=pitches):
                project = Project.create("Drums", Grid())
                track = Track("drums", "drums", "", 4, 8000, "")
                for pitch in pitches:
                    track.notes.author(F(0), F(1), pitch)
                project.tracks.append(track)
                root = ET.fromstring(score_xml(project))
                instruments = root.findall("part-list/score-part/score-instrument")
                self.assertEqual(instruments[0].findtext("instrument-name"), "Drum Kit")
                piece_names = {38: "Snare", 42: "Closed Hi-Hat"}
                self.assertEqual(
                    [i.findtext("instrument-name") for i in instruments[1:]],
                    [piece_names[p] for p in pitches[1:]],
                )
                mappings = {
                    i.get("id"): int(i.findtext("midi-unpitched")) - 1
                    for i in root.findall("part-list/score-part/midi-instrument")
                }
                self.assertCountEqual(
                    [mappings[n.find("instrument").get("id")]
                     for n in root.findall(".//note[unpitched]")],
                    list(pitches),
                )

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

    def vocal_project(self, rows, grid=None):
        project = Project.create("Rhythm", grid or Grid())
        track = Track("voice", "vocals", "", 4, 8000, "")
        for start, end, pitch in rows:
            track.notes.author(start, end, pitch)
        project.tracks.append(track)
        return project

    def test_half_grid_releases_do_not_turn_straight_onsets_into_triplets(self):
        project = self.vocal_project([
            (F(2), F(19, 8), 60),
            (F(5, 2), F(21, 8), 62),
            (F(11, 4), F(3), 64),
        ])
        before = deepcopy(project)
        root = ET.fromstring(score_xml(project))
        self.assertFalse(root.findall(".//time-modification"))
        self.assertFalse(root.findall(".//tuplet"))
        self.assertFalse(root.findall(".//slur"))
        sounding = root.findall(".//note[pitch]")
        self.assertEqual([n.findtext("type") for n in sounding], ["eighth", "16th", "16th"])
        divisions = int(root.findtext(".//divisions"))
        cursor, starts = F(0), []
        for note in root.findall("part/measure/note"):
            if note.find("pitch") is not None:
                starts.append(cursor)
            cursor += F(int(note.findtext("duration")), divisions)
        self.assertEqual(starts, [F(2), F(5, 2), F(11, 4)])
        self.assertEqual(project, before)

    def test_slurs_use_original_touching_boundaries_not_quantized_gaps(self):
        for pitch in (60, 62):
            for gap in (F(0), F(1, 8), F(1, 4)):
                with self.subTest(pitch=pitch, gap=gap):
                    project = self.vocal_project([
                        (F(0), F(1) - gap, 60), (F(1), F(2), pitch),
                    ])
                    root = ET.fromstring(score_xml(project))
                    self.assertEqual(len(root.findall(".//slur")), 2 if gap == 0 else 0)
                    rests = root.findall(".//note[rest]")
                    self.assertEqual(len(rests), 2 if gap == F(1, 4) else 1)
                    self.assertFalse(root.findall(".//tie"))

    def test_short_gap_across_barline_is_articulated_not_slurred(self):
        project = self.vocal_project([
            (F(15, 4), F(31, 8), 60), (F(4), F(17, 4), 62),
        ])
        root = ET.fromstring(score_xml(project))
        self.assertFalse(root.findall(".//slur"))
        first = root.find("part/measure")
        self.assertEqual(first.findall("note")[-1].findtext("type"), "16th")
        self.assertIsNotNone(first.findall("note")[-1].find("pitch"))

    def test_quantization_is_independent_of_editor_grid_and_explicit_triplets_work(self):
        project = self.vocal_project([
            (F(0), F(1, 6), 60), (F(1, 3), F(1, 2), 62), (F(2, 3), F(1), 64),
        ], Grid(subdivision=F(1, 6)))
        self.assertFalse(ET.fromstring(score_xml(project)).findall(".//time-modification"))
        root = ET.fromstring(score_xml(project, quantization=F(1, 3)))
        self.assertTrue(root.findall(".//time-modification"))
        self.assertEqual(
            [n.findtext("type") for n in root.findall(".//note[pitch]")],
            ["eighth", "eighth", "eighth"],
        )
        self.assertFalse(root.findall(".//slur"))

    def test_selected_quantization_controls_gap_threshold_and_compressed_export(self):
        project = self.vocal_project([
            (F(0), F(3, 8), 60), (F(1, 2), F(1), 62),
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "score.mxl"
            export_musicxml(project, path, quantization=F(1, 8))
            with zipfile.ZipFile(path) as archive:
                root = ET.fromstring(archive.read("score.musicxml"))
        self.assertEqual(root.findtext(".//note[rest]/type"), "32nd")
        self.assertFalse(root.findall(".//slur"))

    def test_export_grid_applies_to_every_instrument_and_measures_stay_full(self):
        for numerator, denominator in ((4, 4), (3, 4), (6, 8), (7, 8)):
            for step in (F(1), F(1, 2), F(1, 4), F(1, 8)):
                with self.subTest(meter=(numerator, denominator), step=step):
                    project = Project.create("Score", Grid(120, numerator, denominator))
                    for role, pitch in (("vocals", 60), ("guitar", 64), ("bass", 40),
                                        ("piano", 72), ("drums", 42)):
                        track = Track(role, role, "", 4, 8000, "")
                        for start, end in ((F(0), F(3, 8)), (F(1, 2), F(1)),
                                           (F(5, 4), F(19, 8))):
                            track.notes.author(start, end, pitch)
                        project.tracks.append(track)
                    root = ET.fromstring(score_xml(project, quantization=step))
                    divisions = int(root.findtext(".//divisions"))
                    self.assertFalse(root.findall(".//time-modification"))
                    for measure in root.findall("part/measure"):
                        voices = {}
                        for note in measure.findall("note"):
                            if note.find("chord") is None:
                                voice = note.findtext("voice")
                                voices[voice] = voices.get(voice, F(0)) + F(
                                    int(note.findtext("duration")), divisions,
                                )
                        self.assertTrue(all(v == project.grid.bar_beats for v in voices.values()))

    def test_invalid_quantization_is_reported(self):
        for step in (F(0), F(-1, 4), F(1, 5)):
            with self.assertRaisesRegex(ValueError, "quantization"):
                score_xml(self.project(), quantization=step)
        with self.assertRaisesRegex(ValueError, "quarter-note beats"):
            score_xml(self.project(Grid(120, 6, 8)), quantization=F(1, 6))


if __name__ == "__main__":
    unittest.main()
