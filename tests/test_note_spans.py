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
        markers._pitch_interval_labels = {}
        markers._rejected_pitches = {}
        markers._authored_note_spans = {}
        markers._next_authored_note_id = 1
        markers._pitch_training_lock = threading.Lock()
        markers._pending_pitch_training = []
        markers._pending_pitch_silence_keys = []
        markers._pitch_edit_generation = 0
        markers._note_spans_revision = 0
        markers._generated_note_spans_cache = None
        markers._generated_note_spans_cache_window = None
        markers._published_note_spans = ()
        markers._onset_training_lock = threading.Lock()
        markers._pending_onset_training = []
        markers._onset_edit_generation = 0
        markers._pitch_model = _PitchModelStub()
        markers._record_pitch_gate_example_with_nnls = lambda *args, **kwargs: None
        markers._current_min_dur_sec = 0.25
        markers._harmonic_block_energy = lambda *_: 0.5
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

    def test_visible_worker_rows_preserve_note_ids_for_release_dragging(self) -> None:
        markers = self.make_markers()
        expected = [(0.0, 1.0, 60, "ml", 90, "generated:60:0")]
        markers.get_pitched_note_rows = lambda *_args: expected

        rows = markers.compute_visible_block_pitches(0.0, 1.0)

        self.assertEqual(rows, expected)
        self.assertEqual(rows[0][5], "generated:60:0")

    def test_nnls_note_selection_reuses_precomputed_features(self) -> None:
        markers = self.make_markers()
        markers._nnls_target_midis = [60, 64]
        markers._nnls_gain_threshold = 0.1
        markers._nnls_max_polyphony = 4
        markers._nnls_feature_vectors_for_block = lambda *_: self.fail(
            "precomputed features must not be recomputed"
        )
        features = {
            "combined": np.array([0.8, 0.2]),
            "combined_abs": np.array([0.8, 0.2]),
            "active": np.array([True, False]),
        }

        notes = markers._nnls_notes_for_block(0.0, 1.0, features)

        self.assertEqual(notes, [(60, 0.8)])

    def test_manual_span_does_not_relabel_adjacent_generated_span(self) -> None:
        markers = self.make_markers()

        spans = markers._assemble_note_spans([
            (0.0, 1.0, 60, "manual", 0.4),
            (1.0, 2.0, 60, "nnls", 0.3),
        ])

        self.assertEqual(
            [(span.onset, span.release, span.source) for span in spans],
            [(0.0, 1.0, "manual"), (1.0, 2.0, "nnls")],
        )

    def test_onset_boundary_separates_same_pitch_notes(self) -> None:
        markers = self.make_markers()
        markers.onset_times = np.array([1.0], dtype=float)

        spans = markers._assemble_note_spans([
            (0.0, 1.0, 60, "nnls", 0.2),
            (1.0, 2.0, 60, "nnls", 0.4),
            (1.0, 2.0, 64, "nnls", 0.5),
        ])

        self.assertEqual(
            [(span.onset, span.release, span.midi) for span in spans],
            [(0.0, 1.0, 60), (1.0, 2.0, 60), (1.0, 2.0, 64)],
        )

    def test_click_creates_manual_span_for_exact_onset_interval(self) -> None:
        markers = self.make_markers()
        markers.onset_times = np.array([1.0, 2.0], dtype=float)
        markers.chroma_times = np.array([1.5], dtype=float)
        markers._harmonic_block_energy = lambda *_: 0.5

        added = markers.add_manual_pitch(1.6, 60)

        self.assertTrue(added)
        spans = list(markers._authored_note_spans.values())
        self.assertEqual(len(spans), 1)
        self.assertEqual(
            (spans[0].onset, spans[0].release, spans[0].midi),
            (1.0, 2.0, 60),
        )

    def test_release_snaps_and_clamps_to_next_same_pitch_onset(self) -> None:
        markers = self.make_markers()
        first = PitchedNoteSpan("explicit:1", 0.0, 1.0, 60, "manual", 0.5)
        second = PitchedNoteSpan("explicit:2", 2.0, 3.0, 60, "manual", 0.5)
        markers._authored_note_spans = {first.note_id: first, second.note_id: second}
        markers._next_authored_note_id = 3
        markers._collect_block_note_rows = lambda *_: []
        cached = (PitchedNoteSpan("generated:64:0", 0.0, 4.0, 64, "nnls", 0.4),)
        markers._generated_note_spans_cache = cached

        release = markers.set_note_release(first.note_id, 2.8)

        self.assertEqual(release, 2.0)
        self.assertEqual(markers._authored_note_spans[first.note_id].release, 2.0)
        self.assertIs(markers._generated_note_spans_cache, cached)

    def test_release_override_snaps_to_grid_without_moving_onset(self) -> None:
        markers = self.make_markers()
        markers._current_min_dur_sec = 0.25
        note = PitchedNoteSpan("explicit:1", 1.0, 2.0, 60, "manual", 0.5)
        markers._authored_note_spans = {note.note_id: note}

        release = markers.set_note_release(note.note_id, 1.36)

        self.assertEqual(release, 1.25)
        edited = markers._authored_note_spans[note.note_id]
        self.assertEqual((edited.onset, edited.release), (1.0, 1.25))

    def test_imported_note_release_ignores_hidden_generated_duplicate(self) -> None:
        markers = self.make_markers()
        imported = PitchedNoteSpan(
            "explicit:1", 1.1, 2.1, 60, "manual", 0.5, ((100, 200),)
        )
        duplicate = PitchedNoteSpan(
            "generated:60:120", 1.2, 2.0, 60, "ml", 0.5, ((100, 200),)
        )
        markers._authored_note_spans = {imported.note_id: imported}
        markers._published_note_spans = (duplicate,)
        markers.current_grid_duration = lambda: 0.25

        release = markers.set_note_release(imported.note_id, 2.76)

        self.assertEqual(release, 2.75)
        self.assertEqual(markers._authored_note_spans[imported.note_id].release, 2.75)

    def test_imported_note_release_still_clamps_at_later_same_pitch_note(self) -> None:
        markers = self.make_markers()
        imported = PitchedNoteSpan(
            "explicit:1", 1.1, 2.1, 60, "manual", 0.5, ((100, 200),)
        )
        later = PitchedNoteSpan(
            "explicit:2", 2.5, 3.0, 60, "manual", 0.5, ((250, 300),)
        )
        markers._authored_note_spans = {
            imported.note_id: imported,
            later.note_id: later,
        }
        markers.current_grid_duration = lambda: 0.25

        release = markers.set_note_release(imported.note_id, 2.76)

        self.assertEqual(release, 2.5)

    def test_repeated_release_edit_before_redraw_reuses_authored_span(self) -> None:
        markers = self.make_markers()
        generated = PitchedNoteSpan(
            "generated:60:0", 0.0, 1.0, 60, "nnls", 0.5, ((0, 100),)
        )
        markers._published_note_spans = (generated,)

        first_release = markers.set_note_release(generated.note_id, 1.8)
        second_release = markers.set_note_release(generated.note_id, 2.8)

        self.assertEqual((first_release, second_release), (1.75, 2.75))
        self.assertEqual(len(markers._authored_note_spans), 1)
        self.assertEqual(len(markers._pending_pitch_training), 1)
        authored = next(iter(markers._authored_note_spans.values()))
        self.assertEqual(authored.note_id, generated.note_id)
        self.assertEqual(authored.release, 2.75)
        self.assertEqual(
            markers._pitch_interval_labels[(0, 100)], frozenset({60})
        )

    def test_authored_note_suppresses_generated_tail_from_same_blocks(self) -> None:
        markers = self.make_markers()
        generated = PitchedNoteSpan(
            "generated:60:0", 0.0, 2.0, 60, "nnls", 0.5,
            ((0, 100), (100, 200)),
        )
        manual = PitchedNoteSpan(
            "explicit:1", 0.0, 1.0, 60, "manual", 0.5,
            ((0, 100), (100, 200)),
        )
        markers._authored_note_spans = {manual.note_id: manual}

        spans = markers._overlay_authored_note_spans([generated])

        self.assertEqual(spans, [manual])

    def test_touching_authored_same_pitch_notes_remain_separate(self) -> None:
        markers = self.make_markers()
        first = PitchedNoteSpan("explicit:1", 0.0, 1.0, 60, "manual", 0.5)
        second = PitchedNoteSpan("explicit:2", 1.0, 2.0, 60, "manual", 0.5)
        markers._authored_note_spans = {first.note_id: first, second.note_id: second}

        spans = markers._overlay_authored_note_spans([])

        self.assertEqual(spans, [first, second])

    def test_manual_note_preserves_other_suggestions_in_onset_interval(self) -> None:
        markers = self.make_markers()
        markers._analysis_boundaries = lambda: np.array([0.0, 1.0])
        markers._multi_atom_model = _MultiAtomModelStub()
        markers._nnls_feature_vectors_for_block = lambda *_: None
        markers._nnls_notes_for_block = lambda *_: []
        markers._pitch_model = PitchModelPolicy()
        markers._nnls_max_polyphony = 3
        markers._nnls_midi_to_idx = {}
        markers.model = _LegacyModelMustNotRun()
        manual = PitchedNoteSpan(
            "explicit:1", 0.0, 1.0, 57, "manual", 0.5, ((0, 100),)
        )
        markers._authored_note_spans = {manual.note_id: manual}
        markers._pitch_interval_labels = {(0, 100): frozenset({57})}

        generated = markers._assemble_note_spans(
            markers._collect_block_note_rows(0.0, 1.0)
        )
        spans = markers._overlay_authored_note_spans(generated)

        self.assertEqual(
            [(span.midi, span.source) for span in spans],
            [(57, "manual"), (62, "ml")],
        )

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
            [((0, 100), 60, True), ((100, 400), 64, False)],
        )

    def test_repeated_rejection_is_a_no_op(self) -> None:
        markers = self.make_markers()
        markers.onset_times = np.array([1.0], dtype=float)

        self.assertFalse(markers.remove_manual_pitch(1.5, 64))
        generation = markers.pitch_edit_generation
        revision = markers._note_spans_revision
        pending = list(markers._pending_pitch_training)

        self.assertFalse(markers.remove_manual_pitch(1.5, 64))

        self.assertEqual(markers.pitch_edit_generation, generation)
        self.assertEqual(markers._note_spans_revision, revision)
        self.assertEqual(markers._pending_pitch_training, pending)

    def test_repeated_confirmation_returns_same_span_without_retraining(self) -> None:
        markers = self.make_markers()
        markers.onset_times = np.array([1.0], dtype=float)

        first = markers.add_manual_pitch(0.5, 60)
        generation = markers.pitch_edit_generation
        revision = markers._note_spans_revision
        pending = list(markers._pending_pitch_training)
        second = markers.add_manual_pitch(0.5, 60)

        self.assertIs(second, first)
        self.assertEqual(markers.pitch_edit_generation, generation)
        self.assertEqual(markers._note_spans_revision, revision)
        self.assertEqual(markers._pending_pitch_training, pending)

    def test_latest_pending_feedback_replaces_stale_state(self) -> None:
        markers = self.make_markers()
        markers.onset_times = np.array([1.0], dtype=float)

        markers.add_manual_pitch(0.5, 60)
        markers.remove_manual_pitch(0.5, 60)

        self.assertEqual(
            markers._pending_pitch_training,
            [((0, 100), 60, False)],
        )

    def test_rejected_published_suggestion_stays_out_of_inference(self) -> None:
        markers = self.make_markers()
        markers._analysis_boundaries = lambda: np.array([0.0, 1.0])
        markers._multi_atom_model = _MultiAtomModelStub()
        markers._nnls_feature_vectors_for_block = lambda *_: None
        markers._nnls_notes_for_block = lambda *_: []
        markers._harmonic_block_energy = lambda *_: 1.0
        markers._pitch_model = PitchModelPolicy()
        markers._nnls_max_polyphony = 3
        markers._nnls_midi_to_idx = {}
        markers._current_min_dur_sec = 0.25
        markers.model = _LegacyModelMustNotRun()
        suggested = PitchedNoteSpan(
            "generated:57:0", 0.0, 1.0, 57, "ml", 1.0, ((0, 100),)
        )
        markers._published_note_spans = (suggested,)

        removed = markers.remove_manual_pitch(0.5, 57)
        rows = markers._collect_block_note_rows(0.0, 1.0)

        self.assertTrue(removed)
        self.assertEqual(markers._published_note_spans, ())
        self.assertEqual(markers._rejected_pitches, {(0, 100): {57}})
        self.assertNotIn(57, [row[2] for row in rows])

    def test_confirming_rejected_pitch_restores_manual_note(self) -> None:
        markers = self.make_markers()
        markers.onset_times = np.array([1.0], dtype=float)
        markers._rejected_pitches = {(0, 100): {60}}

        added = markers.add_manual_pitch(0.5, 60)

        self.assertTrue(added)
        self.assertEqual(markers._rejected_pitches, {})
        self.assertEqual(markers._pitch_interval_labels[(0, 100)], frozenset({60}))

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

    def test_unchanged_midi_is_imported_only_once_per_session(self) -> None:
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
        ]) as load_notes:
            first = markers.load_user_notes_from_midi("companion.mid")
            second = markers.load_user_notes_from_midi("companion.mid")

        self.assertEqual(first[0], 1)
        self.assertEqual(second, (0, 0, 0, 0))
        self.assertEqual(load_notes.call_count, 1)
        self.assertEqual(len(markers._pending_pitch_training), 1)

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

    def test_manual_span_survives_changed_onset_boundaries(self) -> None:
        markers = self.make_markers()
        manual = PitchedNoteSpan(
            "explicit:1", 0.0, 2.0, 60, "manual", 0.5, ((0, 200),)
        )
        markers._authored_note_spans = {manual.note_id: manual}
        markers.onset_times = np.array([1.0], dtype=float)
        markers.chroma_times = np.array([], dtype=float)
        markers._collect_block_note_rows = lambda *_: []

        spans = markers.get_pitched_note_spans(0.0, 4.0)

        self.assertEqual(spans, [manual])

    def test_manual_pitch_renders_until_next_onset(self) -> None:
        markers = self.make_markers()
        markers.onset_times = np.array([2.0], dtype=float)
        markers._collect_block_note_rows = lambda *_: []

        markers.add_manual_pitch(0.5, 60)
        spans = markers.get_pitched_note_spans(0.0, 4.0)

        self.assertEqual(
            [(span.onset, span.release, span.midi, span.source) for span in spans],
            [(0.0, 2.0, 60, "manual")],
        )

    def test_new_click_prevents_stale_cache_publication(self) -> None:
        markers = self.make_markers()
        markers.onset_times = np.array([1.0], dtype=float)
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

        self.assertEqual(len(spans), 1)
        self.assertEqual(
            (spans[0].onset, spans[0].release, spans[0].midi, spans[0].source),
            (0.0, 1.0, 64, "manual"),
        )
        self.assertEqual(markers._generated_note_spans_cache, (current,))

    def test_note_inference_collects_only_requested_window(self) -> None:
        markers = self.make_markers()
        current = PitchedNoteSpan("current", 1.0, 2.0, 64, "nnls", 0.5)
        calls = []
        markers._collect_block_note_rows = lambda t0, t1: (
            calls.append((t0, t1)) or [(1.0, 2.0, 64, "nnls", 0.5)]
        )
        markers._assemble_note_spans = lambda _: [current]

        first = markers.get_pitched_note_spans(1.0, 2.0)
        second = markers.get_pitched_note_spans(1.0, 2.0)

        self.assertEqual(first, [current])
        self.assertEqual(second, [current])
        self.assertEqual(calls, [(1.0, 2.0)])

    def test_visible_inference_replaces_only_published_window(self) -> None:
        markers = self.make_markers()
        earlier = PitchedNoteSpan("earlier", 0.0, 1.0, 60, "nnls", 0.4)
        stale = PitchedNoteSpan("stale", 2.0, 3.0, 62, "nnls", 0.4)
        current = PitchedNoteSpan("current", 2.0, 3.0, 64, "nnls", 0.5)
        markers._published_note_spans = (earlier, stale)
        markers._collect_block_note_rows = lambda *_: [
            (2.0, 3.0, 64, "nnls", 0.5)
        ]
        markers._assemble_note_spans = lambda _: [current]

        markers.get_pitched_note_spans(2.0, 3.0)

        self.assertEqual(markers._published_note_spans, (earlier, current))

    def test_midi_export_uses_published_notes_without_inference(self) -> None:
        markers = self.make_markers()
        published = PitchedNoteSpan("seen", 1.0, 2.0, 64, "nnls", 0.5)
        markers._published_note_spans = (published,)
        markers._collect_block_note_rows = lambda *_: self.fail(
            "export must not run inference"
        )

        notes = markers.build_midi_notes()

        self.assertEqual(notes, [(1.0, 2.0, 64, 90)])

    def test_reject_last_manual_pitch_keeps_rest_override(self) -> None:
        markers = self.make_markers()
        markers._pitch_interval_labels = {(0, 200): frozenset({60})}
        markers.onset_times = np.array([1.0], dtype=float)
        markers.chroma_times = np.array([], dtype=float)

        removed = markers.remove_manual_pitch(0.5, 60)

        self.assertTrue(removed)
        self.assertEqual(markers._pitch_interval_labels[(0, 200)], frozenset())
        self.assertEqual(
            markers._pending_pitch_training[-1], ((0, 200), 60, False)
        )

    def test_reject_finds_manual_pitch_after_onset_boundary_change(self) -> None:
        markers = self.make_markers()
        markers._pitch_interval_labels = {(0, 200): frozenset({60})}
        markers.onset_times = np.array([1.0], dtype=float)
        markers.chroma_times = np.array([], dtype=float)

        self.assertTrue(markers.remove_manual_pitch(1.5, 60))

        self.assertEqual(markers._pitch_interval_labels[(0, 200)], frozenset())

    def test_reject_tie_removes_lower_midi_deterministically(self) -> None:
        markers = self.make_markers()
        markers._pitch_interval_labels = {(0, 100): frozenset({60, 62})}
        markers.onset_times = np.array([1.0], dtype=float)

        self.assertTrue(markers.remove_manual_pitch(0.5, 61))

        self.assertEqual(markers._pitch_interval_labels[(0, 100)], frozenset({62}))
        self.assertEqual(markers._rejected_pitches, {(0, 100): {60}})

    def test_reject_authored_span_clears_labels_from_all_covered_blocks(self) -> None:
        markers = self.make_markers()
        span = PitchedNoteSpan(
            "explicit:1", 0.0, 2.0, 60, "manual", 0.5,
            ((0, 100), (100, 200)),
        )
        markers._authored_note_spans = {span.note_id: span}
        markers._pitch_interval_labels = {
            (0, 100): frozenset({60, 64}),
            (100, 200): frozenset({60, 67}),
        }

        self.assertTrue(markers.remove_manual_pitch(0.5, 60))

        self.assertEqual(
            markers._pitch_interval_labels,
            {
                (0, 100): frozenset({64}),
                (100, 200): frozenset({67}),
            },
        )

    def test_reject_removes_release_edited_authored_span(self) -> None:
        markers = self.make_markers()
        span = PitchedNoteSpan(
            "explicit:1", 0.0, 2.0, 60, "manual", 0.5, ((0, 200),)
        )
        markers._authored_note_spans = {span.note_id: span}

        removed = markers.remove_manual_pitch(1.5, 60)

        self.assertTrue(removed)
        self.assertEqual(markers._authored_note_spans, {})
        self.assertEqual(markers._pitch_interval_labels[(0, 200)], frozenset())


if __name__ == "__main__":
    unittest.main()