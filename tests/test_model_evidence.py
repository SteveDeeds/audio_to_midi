from copy import deepcopy
from fractions import Fraction as F
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from audio_to_midi import application as app
from audio_to_midi.adapters import Cancelled, Prediction
from audio_to_midi.domain import Note, Project
from audio_to_midi.evidence_cache import CacheError, EvidenceCache, checksum
from audio_to_midi.model_choices import MODELS
from audio_to_midi.model_evidence import replay_pitch, threshold, validate_parameters
from audio_to_midi.storage import InvalidProjectError, SchemaVersionError, load_project, save_project
from audio_to_midi.timebase import Grid
from audio_to_midi.yourmt3 import select_notes, validate_notes


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = EvidenceCache(self.root / "cache")
        self.descriptor = self.cache.descriptor(
            {"sha256": "audio"}, "stem:piano", {"name": "model", "checkpoint": "one"}, {"sr": 16000})
        self.arrays = {"activation": np.ones((3, 2), dtype=np.float32)}

    def valid(self, arrays):
        self.assertEqual(arrays["activation"].shape, (3, 2))

    def test_exact_rankings_no_mrmt3_and_vocals_unchanged(self):
        self.assertEqual(MODELS["bass"], ("piano_hr", "madmom_brnn_full", "madmom_cnn_full"))
        self.assertEqual(MODELS["guitar"], ("basic_pitch", "yourmt3", "madmom_cnn_full"))
        self.assertEqual(MODELS["piano"], ("piano_hr", "yourmt3", "basic_pitch"))
        self.assertEqual(MODELS["drums"], ("adtof", "yourmt3"))
        self.assertEqual(MODELS["other"], ())
        self.assertEqual(MODELS["vocals"], ("pyin", "crepe_full", "crepe_tiny", "madmom_cnn", "madmom_brnn"))
        self.assertFalse(any("mr_mt3" in choices for choices in MODELS.values()))

    def test_cached_basic_pitch_preserves_note_confidence_for_slider(self):
        from audio_to_midi.evidence_providers import EvidencePolyTranscriber
        from audio_to_midi.model_evidence import replay_poly
        arrays = {name: np.ones((100, 88)) for name in ("onset", "note", "contour")}
        modules = {
            "basic_pitch": SimpleNamespace(),
            "basic_pitch.constants": SimpleNamespace(AUDIO_SAMPLE_RATE=22050, FFT_HOP=256),
            "basic_pitch.note_creation": SimpleNamespace(
                model_output_to_notes=lambda *args, **kwargs: (None, [(0, .5, 60, .81, [])])),
        }
        with patch.dict(sys.modules, modules):
            events = replay_poly("basic_pitch", arrays, "piano", {})
        self.assertEqual(events, [(0.0, .5, 60, .81, .81)])
        provider = EvidencePolyTranscriber(np.ones(8000), 8000, self.root / "stem.wav", {})
        with patch("audio_to_midi.evidence_providers.file_evidence", return_value=arrays), \
                patch("audio_to_midi.evidence_providers.replay_poly", return_value=events):
            provider.prepare(0, 1, "piano", "basic_pitch")
        self.assertAlmostEqual(provider.block_notes(0, .5, "piano", "basic_pitch", .01)[0].confidence, .81)

    def test_cached_piano_hr_confidence_uses_frames_not_velocity(self):
        from audio_to_midi.model_evidence import replay_poly
        class Processor:
            def __init__(self, *args, **kwargs):
                pass

            def output_dict_to_midi_events(self, arrays):
                return ([{"onset_time": 0, "offset_time": .5, "midi_note": 60,
                          "velocity": 12}], [])

        arrays = {"t": np.arange(100) / 100, "frame_output": np.full((100, 88), .93)}
        with patch.dict(sys.modules, {
            "piano_transcription_inference": SimpleNamespace(),
            "piano_transcription_inference.utilities": SimpleNamespace(RegressionPostProcessor=Processor),
        }):
            events = replay_poly("piano_hr", arrays, "piano", {})
        self.assertAlmostEqual(events[0][3], 12 / 127)
        self.assertAlmostEqual(events[0][4], .93)

    def test_reopen_reuses_disk_and_deleted_folder_regenerates_quietly(self):
        calls = []
        def compute():
            calls.append(1)
            return self.arrays
        self.cache.get(self.descriptor, compute, self.valid)
        EvidenceCache(self.cache.root).get(self.descriptor, compute, self.valid)
        self.assertEqual(len(calls), 1)
        shutil.rmtree(self.cache.root)
        EvidenceCache(self.cache.root).get(self.descriptor, compute, self.valid)
        self.assertEqual(len(calls), 2)

    def test_every_identity_dimension_invalidates_and_thresholds_do_not(self):
        self.cache.save(self.descriptor, self.arrays, self.valid)
        for field, value in (
            ("audio", {"sha256": "other"}), ("source", "mix"),
            ("model", {"name": "model", "checkpoint": "two"}), ("parameters", {"sr": 22050}),
            ("schema", 99),
        ):
            changed = {**self.descriptor, field: value}
            self.assertNotEqual(self.cache.path(changed), self.cache.path(self.descriptor))
            self.assertIsNone(self.cache.load(changed, self.valid))
        self.assertNotIn("threshold", json.dumps(self.descriptor))

    def test_file_evidence_keys_cover_content_checkpoint_device_and_sample_rate(self):
        from audio_to_midi.model_evidence import file_evidence
        wav = self.root / "input.wav"
        sf.write(wav, np.zeros(8000), 8000)
        model = {"name": "piano_hr", "checkpoint": "first"}
        with patch("audio_to_midi.model_evidence.model_identity", return_value=model), \
                patch.object(self.cache, "get", return_value=self.arrays) as get:
            with patch.dict(os.environ, ATM_DEVICE="cpu"):
                file_evidence(str(wav), "stem:piano", "piano_hr", cache=self.cache)
            first = get.call_args.args[0]
            self.assertEqual(first["parameters"]["sample_rate"], 16000)
            self.assertEqual(first["parameters"]["precision"], "float32")
            with patch.dict(os.environ, ATM_DEVICE="auto"):
                file_evidence(str(wav), "stem:piano", "piano_hr", cache=self.cache)
            self.assertNotEqual(self.cache.path(first), self.cache.path(get.call_args.args[0]))
            sf.write(wav, np.ones(8000) * .1, 8000)
            with patch.dict(os.environ, ATM_DEVICE="cpu"):
                file_evidence(str(wav), "stem:piano", "piano_hr", cache=self.cache)
            self.assertNotEqual(first["audio"], get.call_args.args[0]["audio"])

    def test_worker_backend_invalidates_even_if_audio_size_and_mtime_are_preserved(self):
        wav = self.root / "input.wav"
        sf.write(wav, np.zeros(8000), 8000)
        project = Project.create("Content identity", Grid())
        project.key = "C"
        track = app.add_stem(project, wav, "piano")
        before = app.media_identity(str(wav))
        service = app.TranscriptionService()
        with patch("audio_to_midi.application.TrackBackend.create") as create:
            create.return_value.predict.return_value = Prediction(F(0), F(1), (), ())
            service.predict(project, track, F(0), F(1))
            service.predict(project, track, F(0), F(1))
            self.assertEqual(create.call_count, 1)
            sf.write(wav, np.ones(8000) * .1, 8000)
            os.utime(wav, ns=(before[2], before[2]))
            self.assertEqual(before, app.media_identity(str(wav)))
            service.predict(project, track, F(0), F(1))
            self.assertEqual(create.call_count, 2)

    def test_corruption_is_explicit_and_never_overwritten(self):
        path = self.cache.path(self.descriptor)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"corrupt")
        with self.assertRaises(CacheError), patch.object(self.cache, "save") as save:
            self.cache.get(self.descriptor, lambda: self.arrays, self.valid)
        save.assert_not_called()
        self.assertEqual(path.read_bytes(), b"corrupt")

    def test_atomic_write_failure_keeps_existing_entry_and_cleans_temporary(self):
        self.cache.save(self.descriptor, self.arrays, self.valid)
        before = checksum(self.cache.path(self.descriptor))
        with patch("audio_to_midi.evidence_cache.os.replace", side_effect=PermissionError("locked")):
            with self.assertRaises(PermissionError):
                self.cache.save(self.descriptor, {"activation": np.zeros((3, 2))}, self.valid)
        self.assertEqual(checksum(self.cache.path(self.descriptor)), before)
        self.assertEqual(list(self.cache.path(self.descriptor).parent.glob(".evidence-*")), [])

    def test_duplicate_threads_compute_once_and_waiting_can_cancel(self):
        entered, release = threading.Event(), threading.Event()
        calls, failures = [], []
        def compute():
            calls.append(1)
            entered.set()
            release.wait(5)
            return self.arrays
        def run():
            try:
                self.cache.get(self.descriptor, compute, self.valid)
            except Exception as exc:
                failures.append(exc)
        first, second = threading.Thread(target=run), threading.Thread(target=run)
        first.start()
        self.assertTrue(entered.wait(5))
        second.start()
        with self.assertRaises(Cancelled):
            self.cache.get(self.descriptor, compute, self.valid, lambda: True)
        release.set()
        first.join(5)
        second.join(5)
        self.assertFalse(failures)
        self.assertEqual(len(calls), 1)

    def test_cancelled_compute_does_not_publish_cache(self):
        cancelled = threading.Event()
        def compute():
            cancelled.set()
            return self.arrays
        with self.assertRaises(Cancelled):
            self.cache.get(self.descriptor, compute, self.valid, cancelled.is_set)
        self.assertFalse(self.cache.path(self.descriptor).exists())

    def test_gpu_contention_uses_cpu_and_waiting_gpu_work_can_cancel(self):
        from core.inference_runtime import neural_device, torch_device
        from audio_to_midi.evidence_cache import file_lock
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        with patch.dict(sys.modules, torch=torch), patch.dict(os.environ, ATM_DEVICE="auto"):
            with file_lock("audio-to-midi-gpu"):
                with neural_device() as device:
                    self.assertEqual(device, "cpu")
                with self.assertRaises(Cancelled):
                    with neural_device(wait=True, cancelled=lambda: True):
                        self.fail("Cancelled worker acquired the GPU")
        torch.cuda.is_available = lambda: False
        with patch.dict(sys.modules, torch=torch), patch.dict(os.environ, ATM_DEVICE="cuda"):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                torch_device()

    def test_subprocess_cancellation_terminates_inflight_work(self):
        from audio_to_midi.pipeline import Cancelled as PreparationCancelled, _run_process
        cancelled = threading.Event()
        reports = []
        def report(text):
            reports.append(text)
            cancelled.set()
        started = time.monotonic()
        with self.assertRaises(PreparationCancelled):
            _run_process([sys.executable, "-u", "-c",
                          "import time; print('inference started', flush=True); time.sleep(60)"],
                         cancel=cancelled.is_set, progress=report, name="Test inference")
        self.assertLess(time.monotonic() - started, 10)
        self.assertEqual(reports, ["inference started"])

    def test_confidence_replay_keeps_unmasked_evidence_and_velocity_is_not_confidence(self):
        arrays = dict(t=np.arange(4) * .01, f0_hz=np.full(4, 440.0),
                      db=np.zeros(4), smoothed_periodicity=np.array([0, .2, .5, 1]))
        original = deepcopy(arrays)
        _, low = replay_pitch("crepe_full", arrays, {"voicing_threshold": .1})
        _, high = replay_pitch("crepe_full", arrays, {"voicing_threshold": .8})
        self.assertEqual(np.isfinite(low).sum(), 3)
        self.assertEqual(np.isfinite(high).sum(), 1)
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], original[key])
        notes = dict(notes=np.array([[0, 1, 60, 1, 0, 0], [0, 1, 64, 127, 24, 0],
                                    [0, .1, 36, 100, 0, 1]], dtype=float))
        validate_notes(notes)
        self.assertEqual(select_notes(notes, "piano", "mix", {})[0, 3], 1)
        self.assertEqual(len(select_notes(notes, "guitar", "stem", {})), 2)
        self.assertEqual(len(select_notes(notes, "drums", "mix", {})), 1)
        with self.assertRaisesRegex(ValueError, "confidence"):
            select_notes(notes, "piano", "mix", {"onset_threshold": 0})

    def test_invalid_or_unsupported_thresholds_fail_instead_of_silent_defaults(self):
        for value in (-1, 2, float("nan"), True, "0.5"):
            with self.assertRaises(ValueError):
                threshold({"onset_threshold": value}, "onset_threshold", .5)
        for model in ("yourmt3", "pyin", "basic_pitch"):
            with self.assertRaises(ValueError):
                validate_parameters(model, {"invented_confidence": .5})

    def test_yourmt3_provision_checks_the_validated_adapter_and_transformers(self):
        from audio_to_midi import provision
        checkpoint = self.root / "checkpoint.ckpt"
        checkpoint.write_bytes(b"test")
        with patch("audio_to_midi.yourmt3.interpreter", return_value=sys.executable), \
                patch("audio_to_midi.yourmt3.checkpoint", return_value=checkpoint), \
                patch.object(provision.subprocess, "run") as run:
            run.return_value = SimpleNamespace(returncode=0, stdout="0.2.0 4.45.1\n")
            self.assertEqual(provision.missing("yourmt3"), [])
            run.return_value = SimpleNamespace(returncode=0, stdout="0.2.0 5.0.0\n")
            self.assertEqual(provision.missing("yourmt3"), ["isolated mt3-infer==0.2.0 environment"])

    def test_saved_projects_round_trip_rankings_and_accept_schema_3(self):
        wav = self.root / "stem.wav"
        sf.write(wav, np.zeros(8000), 8000)
        project = Project.create("Models", Grid())
        for role in MODELS:
            track = app.add_stem(project, wav, role)
            for model in MODELS[role] or ("",):
                track.model = model
                if model == "yourmt3":
                    track.settings["inference_source"] = "mix"
                else:
                    track.settings.pop("inference_source", None)
                save_project(project, self.root / "project")
                self.assertEqual(load_project(self.root / "project"), project)
        path = self.root / "project" / "project.json"
        data = json.loads(path.read_text())
        older = deepcopy(data)
        older["schema_version"] = 3
        for track in older["tracks"]:
            track["notes"].pop("detection_level")
            for kind in ("generated", "authored"):
                for note in track["notes"][kind]:
                    note.pop("confidence")
        path.write_text(json.dumps(older))
        self.assertEqual(load_project(path.parent), project)
        path.write_text(json.dumps({**data, "schema_version": 2}))
        with self.assertRaises(SchemaVersionError):
            load_project(path.parent)
        data["tracks"][0]["model"] = "mr_mt3"
        path.write_text(json.dumps(data))
        with self.assertRaises(InvalidProjectError):
            load_project(path.parent)

    def test_stale_source_mix_audio_project_or_edits_cannot_publish(self):
        wav = self.root / "audio.wav"
        sf.write(wav, np.zeros(8000), 8000)
        project = Project.create("Stale", Grid())
        project.mix_path = str(wav)
        track = app.add_stem(project, wav, "piano")
        track.model = "yourmt3"
        prediction = Prediction(F(0), F(1), (Note("n", F(0), F(1), 60),), ())
        for change in (
            lambda: track.settings.update(inference_source="mix"),
            lambda: setattr(track, "audio_path", str(self.root / "replacement.wav")),
            lambda: setattr(project, "id", "different"),
            lambda: track.notes.author(F(0), F(1), 61),
        ):
            sf.write(self.root / "replacement.wav", np.ones(8000) * .1, 8000)
            captured = app.token(project, track)
            change()
            self.assertFalse(app.publish(project, captured, prediction))
