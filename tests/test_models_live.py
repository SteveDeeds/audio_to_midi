"""Opt-in real checkpoint tests: ATM_RUN_MODEL_TESTS=1, provision models first."""

from fractions import Fraction as F
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from audio_to_midi import application as app
from audio_to_midi.evidence_cache import EvidenceCache
from audio_to_midi.domain import Project
from audio_to_midi.model_choices import MODELS
from audio_to_midi.storage import load_project, save_project
from audio_to_midi.timebase import Grid


@unittest.skipUnless(os.environ.get("ATM_RUN_MODEL_TESTS") == "1", "requires installed checkpoints")
class LiveModels(unittest.TestCase):
    def test_exposed_models_run_reopen_and_replay_without_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, ATM_CACHE_DIR=str(root / "cache")):
                sr = 22050
                t = np.arange(sr * 2) / sr
                envelope = np.where((t % .5) < .35, 1.0, 0.0)
                audio = (.2 * envelope * (np.sin(2 * np.pi * 130.81 * t) +
                                         .5 * np.sin(2 * np.pi * 261.63 * t))).astype(np.float32)
                wav = root / "audio.wav"
                sf.write(wav, audio, sr)
                project = Project.create("Live model smoke", Grid(120))
                project.key = "C"
                project.mix_path = str(wav)
                service = app.TranscriptionService()
                def load_only(store, descriptor, compute, validate, cancelled=lambda: False):
                    arrays = store.load(descriptor, validate)
                    if arrays is None:
                        self.fail(f"Replay/reopen attempted inference: {descriptor}")
                    return arrays
                for role, models in MODELS.items():
                    if role == "other":
                        continue
                    track = app.add_stem(project, wav, role)
                    track.settings.update(license_acknowledged=True, madmom_license_acknowledged=True)
                    for model in models:
                        sources = ("stem", "mix") if model == "yourmt3" else ("stem",)
                        for source in sources:
                            with self.subTest(role=role, model=model, source=source):
                                track.model = model
                                if model == "yourmt3":
                                    track.settings["inference_source"] = source
                                else:
                                    track.settings.pop("inference_source", None)
                                track.settings.pop("decoder_parameters", None)
                                print(f"LIVE {role}/{model}/{source}", flush=True)
                                prediction = service.predict(project, track, F(0), F(4))
                                self.assertEqual((prediction.start, prediction.end), (F(0), F(4)))
                                for note in prediction.notes:
                                    self.assertTrue(0 <= note.pitch <= 127 and 1 <= note.velocity <= 127)
                                self.assertTrue(app.publish(project, app.token(project, track), prediction))
                                track.notes.validate()
                                save_project(project, root / "project")
                                reopened = load_project(root / "project")
                                restored = reopened.track(track.id)
                                with patch.object(EvidenceCache, "get", new=load_only):
                                    replayed = app.TranscriptionService().predict(reopened, restored, F(0), F(4))
                                    self.assertEqual(prediction, replayed)
                                    if model in ("piano_hr", "basic_pitch", "adtof",
                                                 "madmom_cnn_full", "madmom_brnn_full",
                                                 "pyin", "crepe_full", "crepe_tiny"):
                                        name = "voicing_threshold" if model in (
                                            "pyin", "crepe_full", "crepe_tiny") else "onset_threshold"
                                        for level in (0.0, .8, 1.0):
                                            restored.settings["decoder_parameters"] = {name: level}
                                            result = app.TranscriptionService().predict(reopened, restored, F(0), F(4))
                                            self.assertEqual(result.end, F(4))
