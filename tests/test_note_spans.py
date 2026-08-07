import unittest
import threading
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core.audio_engine import AudioMarkers, PitchedNoteSpan
from core.pitch_models import PitchModelPolicy


class _PitchModelStub:
    def __init__(self):
        self.events = []

    def observe_confirmation(self, key, midi):
        self.events.append((key, midi, True))

    def observe_rejection(self, key, midi):
        self.events.append((key, midi, False))


class _CorrectionModelStub:
    def __init__(self):
        self.batches = []

    def record_pitch_gate_batch(self, examples):
        self.batches.append(list(examples))

    def record_candidate_batch(self, examples):
        self.batches.append(list(examples))


class _LearnedChordModelStub:
    PITCH_GATE_THRESHOLD = 0.55

    def pitch_suggestions(self, _fingerprint):
        return [55, 60, 64], 1.0

    def pitch_gate_prob_with_features(self, **_features):
        return None


class _MultiAtomModelStub:
    has_learned_atoms = True

    def predict(self, _t0, _t1):
        return {57, 62}


class _LegacyModelMustNotRun(_LearnedChordModelStub):
    def pitch_suggestions(self, _fingerprint):
        raise AssertionError("legacy tuple inference should not run")


class TestNoteSpans(unittest.TestCase):
    def test_drum_feedback_queues_training_without_feature_extraction(self) -> None:
        markers = AudioMarkers.__new__(AudioMarkers)
        markers.model = object()
        markers.duration = 10.0
        markers._manual_drums = {}
        markers._manual_drum_rejects = {}
        markers._pitch_training_lock = threading.Lock()
        markers._pending_drum_training = []
        markers._pitch_edit_generation = 0
        markers.beat_index_to_time = lambda *_args: 2.0
        markers._extract_drum_slice_features = lambda *_args: self.fail(
            "drum features must be extracted by the worker"
        )

        added = markers.add_manual_drum(4, 36, 120.0, 4, "1/8")

        self.assertTrue(added)
        self.assertEqual(markers._manual_drums, {4: {36}})
        self.assertEqual(
            markers._pending_drum_training,
            [(4, 36, 120.0, 4, "1/8", True)],
        )

    def make_markers(self) -> AudioMarkers:
        markers = AudioMarkers.__new__(AudioMarkers)
        markers.sr = 100
        markers.audio = np.zeros(400, dtype=np.float32)
        markers.duration = 4.0
        markers.onset_times = np.array([], dtype=float)
        markers.chroma_times = np.array([1.0, 2.0, 3.0], dtype=float)
        markers._manual_pitches = {}
        markers._explicit_note_spans = {}
        markers._next_explicit_note_id = 1
        markers._pitch_training_lock = threading.Lock()
        markers._pending_pitch_training = []
        markers._pending_pitch_silence_keys = []
        markers._pitch_edit_generation = 0
        markers._note_spans_revision = 0
        markers._onset_training_lock = threading.Lock()
        markers._pending_onset_training = []
        markers._onset_edit_generation = 0
        markers._pitch_model = _PitchModelStub()
        markers._record_pitch_gate_example_with_nnls = lambda *args, **kwargs: None
        return markers

    def test_contiguous_pitch_blocks_merge_across_chroma_boundaries(self) -> None:
        markers = self.make_markers()
        rows = [
            (0.0, 1.0, 60, "nnls", 0.2),
            (1.0, 2.0, 60, "nnls", 0.4),
            (1.0, 2.0, 64, "nnls", 0.5),
        ]

        spans = markers._assemble_note_spans(rows)

        self.assertEqual(
            [(span.onset, span.release, span.midi) for span in spans],
            [(0.0, 2.0, 60), (1.0, 2.0, 64)],
        )

    def test_unrelated_onset_does_not_retrigger_held_pitch(self) -> None:
        markers = self.make_markers()
        markers.onset_times = np.array([1.0], dtype=float)

        spans = markers._assemble_note_spans([
            (0.0, 1.0, 60, "nnls", 0.2),
            (1.0, 2.0, 60, "nnls", 0.4),
            (1.0, 2.0, 64, "nnls", 0.5),
        ])

        self.assertEqual(
            [(span.onset, span.release, span.midi) for span in spans],
            [(0.0, 2.0, 60), (1.0, 2.0, 64)],
        )

    def test_release_snaps_and_clamps_to_next_same_pitch_onset(self) -> None:
        markers = self.make_markers()
        first = PitchedNoteSpan("explicit:1", 0.0, 1.0, 60, "manual", 0.5)
        second = PitchedNoteSpan("explicit:2", 2.0, 3.0, 60, "manual", 0.5)
        markers._explicit_note_spans = {first.note_id: first, second.note_id: second}
        markers._next_explicit_note_id = 3
        markers._collect_block_note_rows = lambda *_: []
        cached = (PitchedNoteSpan("generated:64:0", 0.0, 4.0, 64, "nnls", 0.4),)
        markers._generated_note_spans_cache = cached

        release = markers.set_note_release(first.note_id, 2.8)

        self.assertEqual(release, 2.0)
        self.assertEqual(markers._explicit_note_spans[first.note_id].release, 2.0)
        self.assertIs(markers._generated_note_spans_cache, cached)

    def test_repeated_release_edit_before_redraw_reuses_explicit_span(self) -> None:
        markers = self.make_markers()
        generated = PitchedNoteSpan(
            "generated:60:0", 0.0, 1.0, 60, "nnls", 0.5, ((0, 100),)
        )
        markers._published_note_spans = (generated,)
        markers._release_edit_note_ids = {}

        first_release = markers.set_note_release(generated.note_id, 1.8)
        second_release = markers.set_note_release(generated.note_id, 2.8)

        self.assertEqual((first_release, second_release), (2.0, 3.0))
        self.assertEqual(len(markers._explicit_note_spans), 1)
        self.assertEqual(len(markers._pending_pitch_training), 1)
        explicit = next(iter(markers._explicit_note_spans.values()))
        self.assertEqual(explicit.release, 3.0)

    def test_explicit_note_suppresses_generated_tail_from_same_blocks(self) -> None:
        markers = self.make_markers()
        generated = PitchedNoteSpan(
            "generated:60:0", 0.0, 2.0, 60, "nnls", 0.5,
            ((0, 100), (100, 200)),
        )
        manual = PitchedNoteSpan(
            "explicit:1", 0.0, 1.0, 60, "manual", 0.5,
            ((0, 100), (100, 200)),
        )
        markers._explicit_note_spans = {manual.note_id: manual}

        spans = markers._overlay_explicit_note_spans([generated])

        self.assertEqual(spans, [manual])

    def test_touching_explicit_same_pitch_notes_remain_separate(self) -> None:
        markers = self.make_markers()
        first = PitchedNoteSpan("explicit:1", 0.0, 1.0, 60, "manual", 0.5)
        second = PitchedNoteSpan("explicit:2", 1.0, 2.0, 60, "manual", 0.5)
        markers._explicit_note_spans = {first.note_id: first, second.note_id: second}

        spans = markers._overlay_explicit_note_spans([])

        self.assertEqual(spans, [first, second])

    def test_learned_chord_precedes_nnls_and_uses_onset_block_duration(self) -> None:
        markers = self.make_markers()
        markers._analysis_boundaries = lambda: np.array([0.0, 1.0])
        markers._cqt_fingerprint_for_block = lambda *_: np.ones(5)
        markers._nnls_feature_vectors_for_block = lambda *_: None
        markers._nnls_notes_for_block = lambda *_: [(36, 1.0), (32, 0.8)]
        markers._harmonic_block_energy = lambda *_: 1.0
        markers._pitch_model = PitchModelPolicy()
        markers._nnls_max_polyphony = 3
        markers._nnls_midi_to_idx = {}
        markers._current_min_dur_sec = 0.125
        markers.model = _LearnedChordModelStub()

        rows = markers._collect_block_note_rows(0.0, 1.0)

        self.assertEqual(
            [(row[0], row[1], row[2], row[3]) for row in rows],
            [
                (0.0, 1.0, 55, "ml"),
                (0.0, 1.0, 60, "ml"),
                (0.0, 1.0, 64, "ml"),
            ],
        )

    def test_multi_atom_predictions_precede_legacy_tuple_inference(self) -> None:
        markers = self.make_markers()
        markers._analysis_boundaries = lambda: np.array([0.0, 1.0])
        markers._multi_atom_model = _MultiAtomModelStub()
        markers._nnls_feature_vectors_for_block = lambda *_: None
        markers._nnls_notes_for_block = lambda *_: [(36, 1.0)]
        markers._harmonic_block_energy = lambda *_: 1.0
        markers._pitch_model = PitchModelPolicy()
        markers._nnls_max_polyphony = 3
        markers._nnls_midi_to_idx = {}
        markers._current_min_dur_sec = 0.25
        markers.model = _LegacyModelMustNotRun()

        rows = markers._collect_block_note_rows(0.0, 1.0)

        self.assertEqual(
            [(row[0], row[1], row[2], row[3]) for row in rows],
            [(0.0, 1.0, 57, "ml"), (0.0, 1.0, 62, "ml")],
        )

    def test_pitch_feedback_is_queued_and_generation_increments(self) -> None:
        markers = self.make_markers()
        markers.onset_times = np.array([1.0], dtype=float)

        self.assertTrue(markers.add_manual_pitch(0.5, 60))
        self.assertFalse(markers.remove_manual_pitch(1.5, 64))

        self.assertEqual(markers.pitch_edit_generation, 2)
        self.assertTrue(markers.has_pending_pitch_training())
        self.assertEqual(
            markers._pending_pitch_training,
            [((0, 100), 60, True), ((100, 200), 64, False)],
        )

    def test_pending_pitch_feedback_trains_as_one_batch(self) -> None:
        markers = self.make_markers()
        markers.model = _CorrectionModelStub()
        markers._pending_pitch_training = [
            ((0, 100), 60, True),
            ((100, 200), 64, False),
        ]
        markers._pitch_edit_generation = 2
        markers._pitch_gate_example_with_nnls = (
            lambda key, midi, keep, source: (key, midi, keep, source)
        )
        trained_keys = []
        markers._train_pitch_model_for_manual_blocks = (
            lambda keys: trained_keys.extend(sorted(keys))
        )

        generation = markers.process_pending_pitch_training()

        self.assertEqual(generation, 2)
        self.assertFalse(markers.has_pending_pitch_training())
        self.assertEqual(len(markers.model.batches), 1)
        self.assertEqual(trained_keys, [(0, 100)])

    def test_midi_import_queues_pitch_training_without_fitting_inline(self) -> None:
        markers = self.make_markers()
        markers.grid = SimpleNamespace(bpm=120.0, beats_per_bar=4)
        markers._current_min_dur_label = "1/8"
        markers._current_min_dur_sec = 0.25
        markers._candidate_raw_times = []
        markers._candidate_features = {}
        markers._candidate_percussive_spikes = {}
        markers._candidate_active_keys = set()
        markers._candidate_forced_on_keys = set()
        markers._candidate_forced_off_keys = set()
        markers._manual_drums = {}
        markers._manual_drum_rejects = {}
        markers._pending_drum_training = []
        markers._harmonic_block_energy = lambda *_args: 1.0
        markers.model = _CorrectionModelStub()

        with patch("core.midi_export.load_midi_notes", return_value=[
            (0.0, 1.0, 60, 100, 0),
            (1.0, 2.0, 64, 100, 0),
        ]):
            result = markers.load_user_notes_from_midi("companion.mid")

        self.assertEqual(result[0], 2)
        self.assertEqual(markers._pitch_model.events, [])
        self.assertEqual(markers.model.batches, [])
        self.assertTrue(markers.has_pending_pitch_training())
        self.assertEqual(len(markers._pending_pitch_training), 2)
        self.assertEqual(markers._pending_pitch_silence_keys, [(200, 300), (300, 400)])

    def test_onset_edit_is_queued_without_training_inline(self) -> None:
        markers = self.make_markers()
        markers.model = _CorrectionModelStub()
        candidate_key = markers._cand_key(1.0)
        markers._candidate_raw_times = [1.0]
        markers._candidate_features = {candidate_key: (0.8, 0.4, -12.0)}
        markers._candidate_percussive_spikes = {candidate_key: 0.2}
        markers._candidate_active_keys = set()
        markers._candidate_forced_on_keys = set()
        markers._candidate_forced_off_keys = set()
        markers._pitch_cache = {}
        markers._pyin_seeded_keys = set()
        markers._generated_note_spans_cache = None

        self.assertTrue(markers.restore_nearest_candidate(1.0))

        self.assertEqual(markers.model.batches, [])
        self.assertTrue(markers.has_pending_onset_training())
        self.assertIn(candidate_key, markers._candidate_active_keys)

    def test_pending_onset_feedback_retrains_candidate_model_once(self) -> None:
        markers = self.make_markers()
        markers.model = _CorrectionModelStub()
        markers._pending_onset_training = [
            (0.8, 0.4, -12.0, 0.2, True, 1000, "user"),
            (0.2, 0.1, -40.0, 0.0, False, 2000, "user"),
        ]
        markers._onset_edit_generation = 2

        generation = markers.process_pending_onset_training()

        self.assertEqual(generation, 2)
        self.assertFalse(markers.has_pending_onset_training())
        self.assertEqual(len(markers.model.batches), 1)
        self.assertEqual(len(markers.model.batches[0]), 2)

    def test_manual_pitch_survives_changed_onset_boundaries(self) -> None:
        markers = self.make_markers()
        markers._manual_pitches = {(0, 200): (60,)}
        markers.onset_times = np.array([1.0], dtype=float)
        markers.chroma_times = np.array([], dtype=float)
        markers._harmonic_block_energy = lambda *_: 0.5
        markers._cqt_fingerprint_for_block = lambda *_: np.zeros(5)
        markers._nnls_feature_vectors_for_block = lambda *_: None
        markers._nnls_notes_for_block = lambda *_: []
        markers._pitch_model = PitchModelPolicy()
        markers.model = type(
            "SuggestionStub", (), {"pitch_suggestions": lambda *_: ([], 0.0)}
        )()

        rows = markers._collect_block_note_rows(0.0, 4.0)

        self.assertIn((0.0, 2.0, 60, "manual", 0.5), rows)

    def test_manual_pitch_renders_until_next_onset(self) -> None:
        markers = self.make_markers()
        markers.onset_times = np.array([2.0], dtype=float)
        markers._manual_pitches = {(0, 100): (60,)}
        markers._current_min_dur_sec = 0.25
        markers._harmonic_block_energy = lambda *_: 0.5
        markers._cqt_fingerprint_for_block = lambda *_: np.zeros(5)
        markers._nnls_feature_vectors_for_block = lambda *_: None
        markers._nnls_notes_for_block = lambda *_: []
        markers._pitch_model = PitchModelPolicy()
        markers.model = type(
            "SuggestionStub", (), {"pitch_suggestions": lambda *_: ([], 0.0)}
        )()

        rows = markers._collect_block_note_rows(0.0, 4.0)

        self.assertIn((0.0, 2.0, 60, "manual", 0.5), rows)

    def test_new_click_prevents_stale_cache_publication(self) -> None:
        markers = self.make_markers()
        markers._generated_note_spans_cache = None
        stale = PitchedNoteSpan("stale", 0.0, 1.0, 60, "nnls", 0.5)
        current = PitchedNoteSpan("current", 0.0, 1.0, 64, "manual", 0.5)
        calls = []

        def collect_rows(*_):
            calls.append(True)
            if len(calls) == 1:
                markers.add_manual_pitch(0.5, 64)
                return [(0.0, 1.0, 60, "nnls", 0.5)]
            return [(0.0, 1.0, 64, "manual", 0.5)]

        markers._collect_block_note_rows = collect_rows
        markers._assemble_note_spans = lambda rows: [
            stale if rows[0][2] == 60 else current
        ]

        spans = markers.get_pitched_note_spans(0.0, 1.0)

        self.assertEqual(spans, [current])
        self.assertEqual(markers._generated_note_spans_cache, (current,))

    def test_reject_last_manual_pitch_keeps_rest_override(self) -> None:
        markers = self.make_markers()
        markers._manual_pitches = {(0, 200): (60,)}
        markers.onset_times = np.array([1.0], dtype=float)
        markers.chroma_times = np.array([], dtype=float)

        removed = markers.remove_manual_pitch(0.5, 60)

        self.assertTrue(removed)
        self.assertIsNone(markers._manual_pitches[(0, 200)])
        self.assertEqual(
            markers._pending_pitch_training[-1], ((0, 200), 60, False)
        )

    def test_reject_finds_manual_pitch_after_onset_boundary_change(self) -> None:
        markers = self.make_markers()
        markers._manual_pitches = {(0, 200): (60,)}
        markers.onset_times = np.array([1.0], dtype=float)
        markers.chroma_times = np.array([], dtype=float)

        self.assertTrue(markers.remove_manual_pitch(1.5, 60))

        self.assertIsNone(markers._manual_pitches[(0, 200)])

    def test_reject_removes_release_edited_explicit_span(self) -> None:
        markers = self.make_markers()
        span = PitchedNoteSpan(
            "explicit:1", 0.0, 2.0, 60, "manual", 0.5, ((0, 200),)
        )
        markers._explicit_note_spans = {span.note_id: span}

        removed = markers.remove_manual_pitch(1.5, 60)

        self.assertTrue(removed)
        self.assertEqual(markers._explicit_note_spans, {})
        self.assertIsNone(markers._manual_pitches[(0, 200)])


if __name__ == "__main__":
    unittest.main()