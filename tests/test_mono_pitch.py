import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import librosa
import numpy as np

from core.audio_engine import AudioMarkers, PitchedNoteSpan
from core.mono_pitch import MonoTranscriber, clean_key_contour, track_pitch
from core.pitch_models import MONO_PITCH_MODEL_TYPES, create_pitch_model

HOP = 0.01
CONTEXT = set(range(12)), np.array([]), np.zeros((12, 0))


def flat_contour(y, sample_rate, _stem, _tracker):
    times = np.arange(0, len(y) / sample_rate, HOP)
    return times, np.full(len(times), 60.0)


def transcriber_with_contour(stem, tracker, pitch, onsets=(), duration=4.0):
    transcriber = MonoTranscriber(np.zeros(int(22050 * duration), dtype=np.float32), 22050, f"{stem}.wav")
    times = np.arange(len(pitch)) * HOP
    pitch = np.asarray(pitch, dtype=float)
    transcriber._contours[(stem, tracker, (0.0, duration))] = times, pitch, np.ones((12, len(pitch)), bool)
    transcriber._clean[(stem, tracker)] = ((times, pitch),)
    transcriber._onsets[stem] = np.asarray(onsets, dtype=float)
    return transcriber


class TestKeyCleaningAndTracking(unittest.TestCase):
    def test_short_chromatic_slide_folds_but_sustained_note_survives(self):
        raw = np.array([60.0] * 30 + [61.0] * 10 + [62.0] * 30 + [63.0] * 30)
        allowed = np.zeros((12, len(raw)), dtype=bool)
        allowed[[0, 2]] = True
        cleaned = clean_key_contour(raw, raw, allowed, 0.01)
        np.testing.assert_array_equal(cleaned[30:40], 62.0)
        np.testing.assert_array_equal(cleaned[70:], 63.0)

    def test_local_harmony_protects_short_chromatic_note(self):
        raw = np.array([60.0] * 30 + [61.0] * 10 + [62.0] * 30)
        allowed = np.ones((12, len(raw)), dtype=bool)
        np.testing.assert_array_equal(clean_key_contour(raw, raw, allowed, 0.01), raw)

    def test_real_pyin_tracks_a440(self):
        sr = 22050
        audio = (0.3 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
        _, midi = track_pitch(audio, sr, "vocals", "pyin")
        self.assertGreater(np.isfinite(midi).mean(), 0.9)
        self.assertAlmostEqual(float(np.nanmedian(midi)), 69.0, delta=0.15)


class TestBlockPitch(unittest.TestCase):
    def test_attack_is_skipped_and_body_votes(self):
        pitch = [55.0] * 5 + [60.0] * 95
        transcriber = transcriber_with_contour("bass", "crepe_tiny", pitch, duration=1.0)
        ranked, release = transcriber.block_pitch(0.0, 1.0, "bass", "crepe_tiny")
        self.assertEqual(ranked[0], 60)
        self.assertNotIn(55, ranked)
        self.assertEqual(release, 1.0)

    def test_precise_attack_moves_the_body_window(self):
        pitch = [np.nan] * 10 + [50.0] * 6 + [60.0] * 32
        transcriber = transcriber_with_contour("bass", "crepe_tiny", pitch, duration=0.48)
        ranked, _ = transcriber.block_pitch(0.0, 0.48, "bass", "crepe_tiny", attack_start=0.1)
        self.assertEqual(ranked[0], 60)

    def test_lost_pitch_releases_early_but_short_dropouts_do_not(self):
        pitch = [60.0] * 50 + [np.nan] * 5 + [60.0] * 20 + [np.nan] * 25
        transcriber = transcriber_with_contour("vocals", "pyin", pitch, duration=1.0)
        ranked, release = transcriber.block_pitch(0.0, 1.0, "vocals", "pyin")
        self.assertEqual(ranked[0], 60)
        self.assertAlmostEqual(release, 0.75)

    def test_unvoiced_or_barely_voiced_block_has_no_pitch(self):
        transcriber = transcriber_with_contour("bass", "crepe_full", [np.nan] * 100, duration=1.0)
        self.assertIsNone(transcriber.block_pitch(0.0, 1.0, "bass", "crepe_full"))
        transcriber = transcriber_with_contour("bass", "crepe_full", [60.0] * 2 + [np.nan] * 98, duration=1.0)
        self.assertIsNone(transcriber.block_pitch(0.0, 1.0, "bass", "crepe_full"))

    def test_held_pitch_changes_are_reported_but_blips_and_gaps_are_not(self):
        pitch = [60.0] * 30 + [62.0] * 30 + [64.0] * 5 + [62.0] * 30 + [np.nan] * 10 + [65.0] * 30
        transcriber = transcriber_with_contour("vocals", "pyin", pitch, duration=1.35)
        changes = transcriber.pitch_changes(0.0, 1.35, "vocals", "pyin")
        self.assertEqual(len(changes), 1)
        self.assertAlmostEqual(changes[0], 0.3)


class TestTranscriberCache(unittest.TestCase):
    def make(self, seconds, path="vocals.wav"):
        transcriber = MonoTranscriber(np.ones(22050 * seconds, dtype=np.float32), 22050, path)
        transcriber._onsets = {"vocals": np.array([]), "bass": np.array([])}
        return transcriber

    def test_chunk_cache_does_not_rerun_tracker(self):
        transcriber = self.make(1)
        with patch.object(transcriber, "_key_context", return_value=CONTEXT):
            with patch("core.mono_pitch.track_pitch", side_effect=flat_contour) as tracker:
                self.assertTrue(transcriber.prepare(0, 1, "vocals", "pyin"))
                transcriber.prepare(0.2, 0.8, "vocals", "pyin")
                tracker.assert_called_once()
        self.assertEqual(transcriber.block_pitch(0, 1, "vocals", "pyin")[0][0], 60)
        self.assertIsNone(transcriber.block_pitch(0, 1, "bass", "crepe_full"))

    def test_inference_is_bounded_to_the_visible_window(self):
        transcriber = self.make(120)
        calls = []

        def contour(y, sample_rate, stem, tracker):
            calls.append(len(y) / sample_rate)
            return flat_contour(y, sample_rate, stem, tracker)

        with patch.object(transcriber, "_key_context", return_value=CONTEXT):
            with patch("core.mono_pitch.track_pitch", side_effect=contour):
                transcriber.prepare(61, 64, "vocals", "pyin")
                first_calls = list(calls)
                transcriber.prepare(61, 64, "vocals", "pyin")
        self.assertEqual(calls, first_calls)
        self.assertLessEqual(max(calls), 2.6)
        self.assertTrue(transcriber.covers(61, 64, "vocals", "pyin"))
        self.assertFalse(transcriber.covers(0, 120, "vocals", "pyin"))

    def test_context_reads_only_visible_sibling_audio(self):
        sr = 22050
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "other.wav").touch()
            transcriber = MonoTranscriber(np.ones(sr * 90, dtype=np.float32), sr, str(root / "vocals.wav"))
            with patch("core.mono_pitch.librosa.load", return_value=(np.ones(sr * 3), sr)) as load:
                with patch("core.mono_pitch.librosa.feature.chroma_cqt", return_value=np.ones((12, 10))):
                    transcriber._key_context(61, 64)
            self.assertEqual(load.call_args.kwargs.get("offset"), 61)
            self.assertEqual(load.call_args.kwargs.get("duration"), 3)

    def test_cancel_keeps_the_finished_batch_and_skips_the_rest(self):
        transcriber = self.make(30)
        cancelled = False

        def contour(y, sample_rate, stem, tracker):
            nonlocal cancelled
            cancelled = True
            return flat_contour(y, sample_rate, stem, tracker)

        with patch.object(transcriber, "_key_context", return_value=CONTEXT):
            with patch("core.mono_pitch.track_pitch", side_effect=contour) as tracker:
                completed = transcriber.prepare(0, 20, "vocals", "pyin", lambda: cancelled)
        self.assertFalse(completed)
        tracker.assert_called_once()
        self.assertTrue(transcriber.covers(0, 2, "vocals", "pyin"))

    def test_max_batches_publishes_progress_between_slow_batches(self):
        transcriber = self.make(10)
        with patch.object(transcriber, "_key_context", return_value=CONTEXT):
            with patch("core.mono_pitch.track_pitch", side_effect=flat_contour) as tracker:
                self.assertFalse(transcriber.prepare(0, 6, "vocals", "pyin", max_batches=1))
                self.assertFalse(transcriber.prepare(0, 6, "vocals", "pyin", max_batches=1))
                self.assertTrue(transcriber.prepare(0, 6, "vocals", "pyin", max_batches=1))
        self.assertEqual(tracker.call_count, 3)

    def test_attack_detector_normalises_over_the_whole_stem_once(self):
        sr = 22050
        audio = np.zeros(sr * 4, dtype=np.float32)
        n = np.arange(int(0.3 * sr))
        for t, amplitude in ((0.5, 1.0), (1.5, 1.0), (2.5, 0.002), (3.5, 1.0)):
            start = int(t * sr)
            audio[start:start + len(n)] = amplitude * np.sin(2 * np.pi * 110 * n / sr) * np.exp(-n / sr * 4)
        transcriber = MonoTranscriber(audio, sr, "bass.wav")
        with patch("core.mono_pitch.librosa.onset.onset_detect", wraps=librosa.onset.onset_detect) as detect:
            onsets = transcriber.onset_times("bass")
            transcriber.onset_times("bass")
        detect.assert_called_once()
        for expected in (0.5, 1.5, 3.5):
            self.assertLess(np.min(np.abs(onsets - expected)), 0.05)
        self.assertGreater(np.min(np.abs(onsets - 2.5)), 0.05)


class FakeCandidateModel:
    _cand_trained = True

    def __init__(self, probs=None):
        self.probs = dict(probs or {})

    def candidate_keep_prob(self, transient, *_):
        return self.probs.get(round(transient, 3), 0.5)


def make_markers(pitch=None, onsets=(1.0, 2.0, 3.0), detector=(1.0, 2.0, 3.0),
                 policy="bass__crepe_tiny+onset+key", probs=None):
    """4 s engine on a 0.25 s grid; contour is 60 for 0-2 s and 62 for 2-4 s unless given.

    Candidate transient feature = t / 10, so ``probs`` is keyed by t / 10.
    """
    markers = AudioMarkers.__new__(AudioMarkers)
    markers.sr = 100
    markers.audio = np.zeros(400, dtype=np.float32)
    markers.duration = 4.0
    markers.wav_path = "bass.wav"
    markers._pitch_training_lock = threading.Lock()
    markers._pitch_model = create_pitch_model("standard")
    markers._pitch_models = {"standard": markers._pitch_model}
    markers._nnls_max_polyphony = 4
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
    markers._harmonic_block_energy = lambda *_: 0.5
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
    if pitch is None:
        pitch = [60.0] * 200 + [62.0] * 200
    model = create_pitch_model(policy)
    markers._mono_transcriber = transcriber_with_contour(model.mono_stem, model.mono_tracker, pitch, detector)
    markers.set_pitch_model(policy)
    return markers


def notes(markers, t_start=0.0, t_end=4.0):
    return [(round(s.onset, 3), round(s.release, 3), s.midi) for s in markers.get_pitched_note_spans(t_start, t_end)]


def onset_marks(markers):
    return [round(t, 3) for t, _ in markers.visible_onset_markers(0, 4)]


class TestOnsetGridIntegration(unittest.TestCase):
    def test_one_pretrained_pitch_per_trained_onset_block(self):
        markers = make_markers()
        self.assertEqual(notes(markers), [(0, 1, 60), (1, 2, 60), (2, 3, 62), (3, 4, 62)])
        self.assertTrue(all(row[3] == "ml" for row in markers.get_pitched_note_rows(0, 4)))

    def test_deleting_an_onset_extends_the_previous_note_to_the_next_onset(self):
        markers = make_markers()
        self.assertTrue(markers.remove_nearest_candidate(1.0))
        self.assertEqual(notes(markers), [(0, 2, 60), (2, 3, 62), (3, 4, 62)])

    def test_adding_an_onset_splits_the_note(self):
        markers = make_markers()
        self.assertTrue(markers.restore_nearest_candidate(1.5))
        self.assertEqual(notes(markers, 0, 2), [(0, 1, 60), (1, 1.5, 60), (1.5, 2, 60)])

    def test_lost_pitch_releases_early_on_the_grid(self):
        pitch = [60.0] * 200 + [62.0] * 130 + [np.nan] * 70
        markers = make_markers(pitch=pitch)
        self.assertEqual(notes(markers)[-1], (3, 3.5, 62))

    def test_release_stops_at_the_right_edge_of_the_view(self):
        markers = make_markers()
        self.assertEqual(notes(markers, 0, 2.6)[-1], (2, 2.6, 62))

    def test_rejected_pitch_falls_back_and_rests_are_kept(self):
        pitch = [60.0] * 70 + [59.0] * 30 + [60.0] * 100 + [62.0] * 200
        markers = make_markers(pitch=pitch)
        markers._rejected_pitches[(0, 100)] = {60}
        markers._mark_pitch_interval_rest((100, 200))
        markers._invalidate_note_spans()
        self.assertEqual(notes(markers, 0, 3), [(0, 1, 59), (2, 3, 62)])

    def test_ui_reads_do_not_start_tracker_inference(self):
        markers = make_markers()
        with patch.object(markers._mono_transcriber, "prepare", side_effect=AssertionError("UI inference")):
            self.assertTrue(markers.get_pitched_note_rows(0, 4))

    def test_worker_runs_one_batch_per_pass_and_reports_remaining_work(self):
        markers = make_markers()
        with patch.object(markers._mono_transcriber, "prepare", return_value=False) as prepare:
            rows = markers.compute_visible_block_pitches(0, 4)
        self.assertEqual(prepare.call_args.kwargs.get("max_batches"), 1)
        self.assertEqual(len(rows), 4)
        self.assertFalse(markers.has_pending_visible_analysis(0, 4))
        self.assertTrue(markers.has_pending_visible_analysis(0, 6) or markers.duration < 6)


class TestOnsetEvidence(unittest.TestCase):
    def test_attacks_the_model_accepts_become_onsets_and_unsupported_guesses_go(self):
        # 2.25 has an attack but the onset model rejects it; 0.5 / 3.0 have no attack.
        markers = make_markers(onsets=(0.5, 1.0, 3.0), detector=(1.02, 2.0, 2.26), probs={0.225: 0.1})
        markers.compute_visible_block_pitches(0, 4)
        self.assertEqual(onset_marks(markers), [1.0, 2.0])

    def test_very_confident_model_keeps_an_onset_without_an_attack(self):
        markers = make_markers(detector=(), probs={0.1: 0.95})
        markers.compute_visible_block_pitches(0, 4)
        self.assertEqual(onset_marks(markers), [1.0])

    def test_no_onset_where_there_is_no_pitch(self):
        pitch = [60.0] * 200 + [np.nan] * 200
        markers = make_markers(pitch=pitch, onsets=(), detector=(1.0, 3.0))
        markers.compute_visible_block_pitches(0, 4)
        self.assertEqual(onset_marks(markers), [1.0])

    def test_user_onsets_survive_reapplication(self):
        markers = make_markers(detector=(2.0,))
        self.assertTrue(markers.restore_nearest_candidate(1.5))
        self.assertTrue(markers.remove_nearest_candidate(2.0))
        markers.compute_visible_block_pitches(0, 4)
        self.assertEqual(onset_marks(markers), [1.5])
        self.assertEqual(notes(markers), [(0, 1.5, 60), (1.5, 4, 62)])

    def test_vocal_legato_change_adds_a_grid_onset_but_bass_does_not(self):
        low = {round(t / 10, 3): 0.1 for t in np.arange(0.25, 4.0, 0.25)}
        for policy, expected in (("vocals__pyin+key", [2.0]), ("bass__crepe_tiny+onset+key", [])):
            markers = make_markers(onsets=(), detector=(), policy=policy, probs=low)
            markers.compute_visible_block_pitches(0, 4)
            self.assertEqual(onset_marks(markers), expected, policy)

    def test_onset_model_feedback_reapplies_evidence_in_mono_mode(self):
        markers = make_markers(detector=(1.0,))
        self.assertEqual(markers.apply_ml_suggestions_window(0, 4), (2, 0))
        self.assertEqual(onset_marks(markers), [1.0])

    def test_leaving_mono_mode_restores_trained_onsets_plus_user_edits(self):
        markers = make_markers(detector=(1.0,))
        markers.compute_visible_block_pitches(0, 4)
        self.assertTrue(markers.restore_nearest_candidate(0.5))
        markers.set_pitch_model("standard")
        self.assertEqual(onset_marks(markers), [0.5, 1.0, 2.0, 3.0])


class TestPresets(unittest.TestCase):
    def test_all_presets_roundtrip(self):
        for model_type in MONO_PITCH_MODEL_TYPES:
            model = model_type()
            restored = create_pitch_model(model.name, model.to_dict())
            self.assertEqual((restored.name, restored.mono_stem, restored.mono_tracker),
                             (model.name, model.mono_stem, model.mono_tracker))
        self.assertTrue(create_pitch_model("vocals__crepe_full+key").mono_legato_split)
        self.assertFalse(create_pitch_model("bass__crepe_full+onset+key").mono_legato_split)


class TestMonoControls(unittest.TestCase):
    def test_dropdown_emits_real_model_and_locks_polyphony(self):
        # Coordinator tests own a QCoreApplication, which cannot host widgets.
        code = """
from PySide6.QtWidgets import QApplication
from core.pitch_models import MONO_PITCH_MODEL_TYPES
from ui.controls import ControlPanel
app = QApplication([])
panel = ControlPanel()
models = []
panel.pitch_model_changed.connect(models.append)
for model in MONO_PITCH_MODEL_TYPES:
    panel.set_pitch_model(model.name)
    assert panel.pitch_model() == model.name
    assert panel.polyphony() == 1
    assert not panel._polyphony_slider.isEnabled()
assert models == [model.name for model in MONO_PITCH_MODEL_TYPES]
panel.set_pitch_model("standard")
assert panel._polyphony_slider.isEnabled()
panel.close()
"""
        env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
        subprocess.run([sys.executable, "-c", code], check=True, env=env,
                       cwd=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    unittest.main()
