import tempfile
import unittest
from pathlib import Path

import mido

from core.audio_engine import AudioMarkers, TimeGrid
from core.midi_export import build_midi, load_midi_notes, load_midi_timing
from core.models import CorrectionModel


class TestMidiRoundTrip(unittest.TestCase):
    def test_midi_boundaries_train_positive_and_silence_candidates_negative(self) -> None:
        markers = AudioMarkers.__new__(AudioMarkers)
        markers._current_min_dur_sec = 0.25
        markers._candidate_raw_times = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75]
        markers._candidate_features = {
            markers._cand_key(t): (0.2, 0.1, -20.0)
            for t in markers._candidate_raw_times
        }
        markers._candidate_percussive_spikes = {}
        markers._candidate_active_keys = set(markers._candidate_features)
        markers._candidate_forced_on_keys = set()
        markers._candidate_forced_off_keys = set()
        markers._pitch_cache = {}
        markers._pyin_seeded_keys = set()
        markers.chroma_times = []
        markers.model = CorrectionModel()

        positives, negatives = markers._train_note_boundaries_from_midi([
            (0.25, 0.75, 60, 100, 0),
        ])

        self.assertEqual((positives, negatives), (2, 4))
        user_rows = {
            row["key"]: row["label"]
            for row in markers.model.to_dict(user_only=True)["cand"]
        }
        self.assertEqual(user_rows[markers._cand_key(0.25)], 1)
        self.assertNotIn(markers._cand_key(0.50), user_rows)
        self.assertEqual(user_rows[markers._cand_key(0.75)], 1)
        self.assertEqual(user_rows[markers._cand_key(1.00)], 0)

    def test_midi_timing_and_compound_meter_grid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "timing.mid"
            mid = mido.MidiFile()
            track = mido.MidiTrack()
            mid.tracks.append(track)
            track.append(mido.MetaMessage("set_tempo", tempo=750000, time=0))
            track.append(mido.MetaMessage(
                "time_signature", numerator=6, denominator=8, time=0
            ))
            mid.save(out_path)

            timing = load_midi_timing(str(out_path))

        self.assertAlmostEqual(float(timing["bpm"]), 80.0)
        self.assertEqual((timing["beats_per_bar"], timing["beat_unit"]), (6, 8))
        grid = TimeGrid(80.0, 6, 8)
        self.assertAlmostEqual(grid.beat_dur, 0.375)
        self.assertAlmostEqual(grid.bar_dur, 2.25)

    def test_build_then_load_preserves_notes_velocity_and_channel(self) -> None:
        notes = [
            (0.00, 0.50, 60, 70),
            (0.25, 0.75, 64, 95),
            (0.50, 1.00, 67, 120),
        ]

        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "roundtrip.mid"
            build_midi(notes, bpm=120.0, output_path=str(out_path), channel=2)

            parsed = load_midi_notes(str(out_path))

        self.assertEqual(len(parsed), 3)

        parsed_simple = [(row[2], row[3], row[4]) for row in parsed]
        self.assertEqual(parsed_simple, [(60, 70, 2), (64, 95, 2), (67, 120, 2)])

        starts = [row[0] for row in parsed]
        ends = [row[1] for row in parsed]
        self.assertTrue(abs(starts[0] - 0.00) < 0.01)
        self.assertTrue(abs(starts[1] - 0.25) < 0.01)
        self.assertTrue(abs(starts[2] - 0.50) < 0.01)
        self.assertTrue(abs(ends[0] - 0.50) < 0.01)
        self.assertTrue(abs(ends[1] - 0.75) < 0.01)
        self.assertTrue(abs(ends[2] - 1.00) < 0.01)

    def test_touching_same_pitch_notes_remain_separate(self) -> None:
        notes = [
            (0.00, 0.50, 60, 80),
            (0.50, 1.00, 60, 100),
        ]

        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "retrigger.mid"
            build_midi(notes, bpm=120.0, output_path=str(out_path), channel=0)
            parsed = load_midi_notes(str(out_path))

        self.assertEqual(len(parsed), 2)
        self.assertTrue(abs(parsed[0][0] - 0.00) < 0.01)
        self.assertTrue(abs(parsed[0][1] - 0.50) < 0.01)
        self.assertTrue(abs(parsed[1][0] - 0.50) < 0.01)
        self.assertTrue(abs(parsed[1][1] - 1.00) < 0.01)


if __name__ == "__main__":
    unittest.main()
