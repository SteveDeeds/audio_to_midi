import threading
import unittest
from unittest.mock import patch

import numpy as np

from core.audio_engine import AudioMarkers
from core.mode_registry import get_mode
from core.pitch_models import POLY_PITCH_MODEL_TYPES, create_pitch_model
from core.poly_pitch import PolyNote, PolyTranscriber


def transcriber_with_notes(stem, tracker, notes, duration=4.0):
    """Cache `notes` (onset, end, midi, velocity[, struck]) as one analysed batch."""
    transcriber = PolyTranscriber(np.zeros(int(100 * duration), dtype=np.float32), 100)
    transcriber._chunks[(stem, tracker, (0.0, duration))] = [
        PolyNote(n[0], n[1], n[2], n[3], n[4] if len(n) > 4 else True) for n in notes
    ]
    transcriber._rebuild(stem, tracker)
    return transcriber


class FakeCandidateModel:
    _cand_trained = True

    def __init__(self, probs=None):
        self.probs = dict(probs or {})

    def candidate_keep_prob(self, transient, *_):
        return self.probs.get(round(transient, 3), 0.5)


# C major chord at 0 held to 2 s, melody E5/F5 on top, then a G chord at 2 s.
CHORDS = [
    (0.0, 2.0, 48, 0.8), (0.0, 2.0, 52, 0.7), (0.0, 2.0, 55, 0.7),
    (0.02, 0.95, 76, 0.9), (1.0, 1.9, 77, 0.9),
    (2.01, 3.4, 43, 0.8), (2.01, 3.4, 59, 0.6),
]


def make_markers(notes=CHORDS, onsets=(1.0, 2.0, 3.0), policy="piano__basic_pitch", probs=None, polyphony=6):
    markers = AudioMarkers.__new__(AudioMarkers)
    markers.sr = 100
    markers.audio = np.zeros(400, dtype=np.float32)
    markers.duration = 4.0
    markers.wav_path = "piano.wav"
    markers._pitch_training_lock = threading.Lock()
    markers._pitch_model = create_pitch_model("standard")
    markers._pitch_models = {"standard": markers._pitch_model}
    markers._nnls_max_polyphony = polyphony
    markers._pitch_cache = {}
    markers._pyin_seeded_keys = set()
    markers._pitch_edit_generation = 0
    markers._pitch_interval_labels = {}
    markers._rejected_pitches = {}
    markers._authored_note_spans = {}
    markers._next_authored_note_id = 1
    markers._generated_note_spans_cache = None
    markers._generated_note_spans_cache_window = None
    markers._published_note_spans = ()
    markers._pending_pitch_training = []
    markers._pending_pitch_silence_keys = []
    markers._note_spans_revision = 0
    markers._onset_training_lock = threading.Lock()
    markers._pending_onset_training = []
    markers._onset_edit_generation = 0
    markers._current_min_dur_sec = 0.25
    markers.onset_times = np.array([])
    markers.chroma_times = np.array([])
    markers._candidate_raw_times = [float(t) for t in np.arange(0.25, 4.0, 0.25)]
    markers._candidate_features = {
        markers._cand_key(t): (round(t / 10, 3), 0.0, -20.0) for t in markers._candidate_raw_times
    }
    markers._candidate_percussive_spikes = {}
    markers._candidate_active_keys = {markers._cand_key(t) for t in onsets}
    markers._candidate_forced_on_keys = set()
    markers._candidate_forced_off_keys = set()
    markers._mono_baseline_active_keys = None
    markers.model = FakeCandidateModel(probs)
    markers._update_active_times()
    model = create_pitch_model(policy)
    markers._poly_transcriber = transcriber_with_notes(model.poly_stem, model.poly_tracker, notes)
    markers.set_pitch_model(policy)
    return markers


def notes(markers, t_start=0.0, t_end=4.0):
    return sorted(
        (round(s.onset, 3), round(s.release, 3), s.midi)
        for s in markers.get_pitched_note_spans(t_start, t_end)
    )


def onset_marks(markers):
    return [round(t, 3) for t, _ in markers.visible_onset_markers(0, 4)]


class TestBlockNotes(unittest.TestCase):
    def test_struck_and_held_pitches_per_block(self):
        transcriber = transcriber_with_notes("piano", "basic_pitch", CHORDS)
        first = transcriber.block_notes(0.0, 1.0, "piano", "basic_pitch", 0.125)
        self.assertEqual({(n.midi, n.continues) for n in first},
                         {(48, False), (52, False), (55, False), (76, False)})
        self.assertAlmostEqual(next(n for n in first if n.midi == 76).release, 0.95)
        second = transcriber.block_notes(1.0, 2.0, "piano", "basic_pitch", 0.125)
        self.assertEqual({(n.midi, n.continues) for n in second},
                         {(48, True), (52, True), (55, True), (77, False)})
        self.assertFalse(second[0].continues)

    def test_note_struck_mid_block_is_left_out(self):
        transcriber = transcriber_with_notes("piano", "basic_pitch", CHORDS)
        merged = transcriber.block_notes(0.0, 2.0, "piano", "basic_pitch", 0.125)
        self.assertNotIn(77, {n.midi for n in merged})

    def test_blip_is_ignored_and_sounding_fraction_unions_pitches(self):
        transcriber = transcriber_with_notes("guitar", "piano_hr", [(0.0, 0.02, 60, 0.5), (0.5, 0.75, 62, 0.5),
                                                                    (0.6, 1.0, 64, 0.5)])
        self.assertEqual(transcriber.block_notes(0.0, 0.5, "guitar", "piano_hr", 0.125), [])
        self.assertAlmostEqual(transcriber.sounding_fraction(0.5, 1.0, "guitar", "piano_hr"), 1.0)
        self.assertAlmostEqual(transcriber.sounding_fraction(0.0, 0.5, "guitar", "piano_hr"), 0.04)
        np.testing.assert_allclose(transcriber.onset_times("guitar", "piano_hr"), [0.0, 0.5, 0.6])


class TestPolyBatches(unittest.TestCase):
    def test_batches_own_onsets_and_continue_held_notes(self):
        transcriber = PolyTranscriber(np.ones(100 * 20, dtype=np.float32), 100)
        transcriber.CHUNK_SECONDS = {"basic_pitch": 4.0}
        transcriber.CONTEXT_SECONDS = {"basic_pitch": 1.0}

        def fake(y, sr, stem, tracker):
            start = calls[-1] if calls else 0.0
            return [(t - start, 9.0 - start, 60, 0.8) for t in (1.0, 3.5) if start <= t < start + len(y) / sr] + (
                [(0.0, 9.0 - start, 60, 0.8)] if start > 1.0 else []
            )

        calls = []

        def wrapped(y, sr, stem, tracker):
            result = fake(y, sr, stem, tracker)
            calls.append(calls[-1] + 4.0 if calls else 3.0)
            return result

        with patch("core.poly_pitch.transcribe_notes", side_effect=wrapped) as model:
            self.assertFalse(transcriber.prepare(0, 8, "piano", "basic_pitch", max_batches=1))
            self.assertTrue(transcriber.prepare(0, 8, "piano", "basic_pitch"))
            transcriber.prepare(2, 6, "piano", "basic_pitch")
        self.assertEqual(model.call_count, 2)
        np.testing.assert_allclose(transcriber.onset_times("piano", "basic_pitch"), [1.0, 3.5])
        held = transcriber.block_notes(4.0, 8.0, "piano", "basic_pitch", 0.125)
        self.assertEqual([(n.midi, n.continues) for n in held], [(60, True)])
        self.assertAlmostEqual(held[0].release, 8.0)

    def test_cancel_keeps_finished_batch(self):
        transcriber = PolyTranscriber(np.ones(100 * 30, dtype=np.float32), 100)
        done = []

        def model(*_):
            done.append(1)
            return []

        with patch("core.poly_pitch.transcribe_notes", side_effect=model):
            self.assertFalse(transcriber.prepare(0, 20, "piano", "basic_pitch", lambda: bool(done)))
        self.assertEqual(len(done), 1)
        self.assertTrue(transcriber.covers(0, 4, "piano", "basic_pitch"))


class TestPolyOnsetGrid(unittest.TestCase):
    def test_chord_holds_across_melody_onsets(self):
        markers = make_markers()
        self.assertEqual(notes(markers), [
            (0, 1, 76), (0, 2, 48), (0, 2, 52), (0, 2, 55), (1, 2, 77),
            (2, 3.5, 43), (2, 3.5, 59),
        ])

    def test_deleting_an_onset_removes_the_notes_struck_there(self):
        markers = make_markers()
        self.assertTrue(markers.remove_nearest_candidate(1.0))
        self.assertEqual(notes(markers, 0, 2), [(0, 1, 76), (0, 2, 48), (0, 2, 52), (0, 2, 55)])

    def test_polyphony_slider_caps_notes_per_block_keeping_strongest(self):
        markers = make_markers(polyphony=2)
        self.assertEqual(notes(markers, 0, 1), [(0, 1, 48), (0, 1, 76)])

    def test_release_stops_at_view_edge_and_rejections_apply(self):
        markers = make_markers()
        markers._rejected_pitches[(0, 100)] = {76}
        markers._invalidate_note_spans()
        self.assertEqual(notes(markers, 0, 0.6), [(0, 0.6, 48), (0, 0.6, 52), (0, 0.6, 55)])

    def test_model_note_starts_become_grid_onsets(self):
        markers = make_markers(onsets=(0.5, 1.0), probs={0.05: 0.1})
        markers.compute_visible_block_pitches(0, 4)
        self.assertEqual(onset_marks(markers), [1.0, 2.0])

    def test_worker_prepares_the_selected_model(self):
        markers = make_markers(policy="guitar__piano_hr")
        with patch.object(markers._poly_transcriber, "prepare", return_value=True) as prepare:
            markers.compute_visible_block_pitches(0, 4)
        self.assertEqual(prepare.call_args.args[2:4], ("guitar", "piano_hr"))
        self.assertFalse(markers.has_pending_visible_analysis(0, 4))


class TestPolyPresets(unittest.TestCase):
    def test_piano_and_guitar_modes_are_registered(self):
        expected = {
            "piano__basic_pitch": ("piano", "basic_pitch"),
            "piano__piano_hr": ("piano", "piano_hr"),
            "guitar__basic_pitch": ("guitar", "basic_pitch"),
            "guitar__piano_hr": ("guitar", "piano_hr"),
        }
        self.assertEqual({m.name for m in POLY_PITCH_MODEL_TYPES}, set(expected))
        for name, (stem, tracker) in expected.items():
            policy = get_mode(name).create_pitch_policy()
            self.assertEqual((policy.name, policy.poly_stem, policy.poly_tracker), (name, stem, tracker))
            self.assertEqual((policy.visible_stem, policy.visible_tracker), (stem, tracker))
            self.assertIsNone(policy.max_polyphony)
            self.assertIsNone(policy.mono_stem)


if __name__ == "__main__":
    unittest.main()
