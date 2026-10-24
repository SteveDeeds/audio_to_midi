"""Synthetic/offline contracts for the preparation store and separation."""

import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from audio_to_midi.pipeline import (
    ArtifactStore, Cancelled, PreparationError, RevisionConflict, _run_process, separate_mix,
)


ROOT = Path(__file__).resolve().parents[1]


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix=".test-preparation-", dir=ROOT)
        self.root = Path(self.directory.name)
        self.rate = 16000
        t = np.arange(self.rate * 3) / self.rate
        tone = 0.25 * np.sin(2 * np.pi * 440 * t)
        self.mix = self.root / "original.wav"
        sf.write(self.mix, np.column_stack((tone, tone * 0.5)), self.rate, subtype="FLOAT")

    def tearDown(self):
        self.directory.cleanup()

    def store_stage(self, store, stage="analysis", content="one", dependencies=(), **kwargs):
        def builder(staging):
            (staging / "output.txt").write_text(content, encoding="utf-8")
            return {"result": "output.txt"}, {"content": content}
        return store.run(stage, inputs=[self.mix], parameters={"content": content},
                         backend="synthetic-v1", builder=builder, dependencies=dependencies, **kwargs)

    def test_durable_cache_hashes_parameters_and_backend(self):
        store = ArtifactStore(self.root / "artifacts")
        first = self.store_stage(store)
        same = self.store_stage(ArtifactStore(store.root))
        self.assertEqual(first["id"], same["id"])
        second = self.store_stage(store, content="two")
        self.assertNotEqual(first["id"], second["id"])
        self.assertTrue(store.output(first, "result").exists())
        self.assertTrue(store.revision(first["id"])["stale"])
        store.output(second, "result").write_text("tampered", encoding="utf-8")
        with self.assertRaises(RevisionConflict):
            self.store_stage(store, content="two")
        third = self.store_stage(store, content="two", force_new_revision=True)
        self.assertEqual(store.output(second, "result").read_text(), "tampered")
        self.assertNotEqual(second["id"], third["id"])
        # Same pathname, changed bytes: it must not reuse a previous key.
        sf.write(self.mix, np.zeros((1000, 2)), self.rate)
        fourth = self.store_stage(store, content="two")
        self.assertNotEqual(third["key"], fourth["key"])

    def test_upstream_rerun_protects_authored_and_tampered_downstream(self):
        for use_marker in (True, False):
            store = ArtifactStore(self.root / str(use_marker))
            upstream = self.store_stage(store)
            downstream = self.store_stage(store, "separation", dependencies=[upstream["id"]])
            if use_marker:
                store.mark_edited(downstream["id"])
            else:
                store.output(downstream, "result").write_text("manual change")
            with self.assertRaises(RevisionConflict):
                self.store_stage(store, content="changed")
            self.assertEqual(len(store.manifest["revisions"]), 2)
            replacement = self.store_stage(store, content="changed", force_new_revision=True)
            self.assertTrue(store.revision(downstream["id"])["stale"])
            self.assertTrue(store.output(downstream, "result").exists())
            self.assertFalse(replacement["stale"])
            with self.assertRaises(RevisionConflict):
                self.store_stage(store, "notes", dependencies=[downstream["id"]])

    def test_failed_cancelled_and_interrupted_jobs_not_done(self):
        store = ArtifactStore(self.root / "artifacts")

        def fail(staging):
            (staging / "partial").write_text("partial")
            raise Cancelled("test")

        with self.assertRaises(Cancelled):
            store.run("analysis", inputs=[self.mix], parameters={}, backend="test", builder=fail)
        self.assertEqual(store.manifest["revisions"][-1]["status"], "cancelled")
        self.assertEqual(store.manifest["active"], {})
        self.assertFalse(list(store.root.glob(".stage-*")))
        manifest = store.manifest
        manifest["revisions"][-1]["status"] = "running"
        store.path.write_text(json.dumps(manifest), encoding="utf-8")
        recovered = ArtifactStore(store.root)
        self.assertEqual(recovered.manifest["revisions"][-1]["status"], "interrupted")
        self.assertFalse(recovered.valid(recovered.manifest["revisions"][-1]))
        self.assertEqual(json.loads(store.path.read_text())["revisions"][-1]["status"], "interrupted")

    def test_editing_during_stage_preserves_live_job_and_revision_protection(self):
        store = ArtifactStore(self.root / "artifacts")
        upstream = self.store_stage(store)
        editor = ArtifactStore(store.root)

        def builder(staging):
            observer = ArtifactStore(store.root)
            self.assertEqual(observer.manifest["revisions"][-1]["status"], "running")
            editor.mark_edited(upstream["id"])
            store.mark_edited(upstream["id"])
            (staging / "output.txt").write_text("separated")
            return {"result": "output.txt"}, {}

        result = store.run(
            "separation", inputs=[self.mix], parameters={}, backend="test",
            builder=builder, dependencies=[upstream["id"]],
        )
        recovered = ArtifactStore(store.root)
        self.assertEqual(recovered.revision(result["id"])["status"], "done")
        self.assertTrue(recovered.revision(upstream["id"])["edited"])
        with self.assertRaises(RevisionConflict):
            self.store_stage(recovered, content="changed")

    def test_atomic_manifest_failure_keeps_previous_revision(self):
        store = ArtifactStore(self.root / "artifacts")
        original = self.store_stage(store)
        previous = store.path.read_bytes()
        with patch("audio_to_midi.pipeline.os.replace", side_effect=OSError("interrupted write")):
            with self.assertRaises(OSError):
                store.mark_edited(original["id"])
        self.assertEqual(store.path.read_bytes(), previous)
        self.assertFalse(ArtifactStore(store.root).revision(original["id"])["edited"])
        self.assertFalse(list(store.root.glob(".state-*")))

    def test_backend_identity_is_in_cache_key(self):
        store = ArtifactStore(self.root / "artifacts")
        original = self.store_stage(store)

        def builder(staging):
            (staging / "output.txt").write_text("one")
            return {"result": "output.txt"}, {}

        newer = store.run("analysis", inputs=[self.mix], parameters={"content": "one"},
                          backend="synthetic-v2", builder=builder)
        self.assertNotEqual(original["key"], newer["key"])

    def test_input_mutation_during_job_rejected(self):
        store = ArtifactStore(self.root / "artifacts")

        def mutate(staging):
            sf.write(self.mix, np.zeros((100, 2)), self.rate)
            (staging / "x").write_text("x")
            return {"x": "x"}, {}

        with self.assertRaises(RevisionConflict):
            store.run("analysis", inputs=[self.mix], parameters={}, backend="test", builder=mutate)
        self.assertFalse(store.manifest["active"])
        self.assertEqual(store.manifest["revisions"][-1]["status"], "failed")

    def test_process_cancellation_and_progress_real_subprocess(self):
        started = time.monotonic()
        progress = []
        with self.assertRaises(Cancelled):
            _run_process(
                [sys.executable, "-u", "-c",
                 "import time; print('ready', flush=True); time.sleep(30)"],
                cancel=lambda: bool(progress), progress=progress.append,
            )
        self.assertIn("ready", progress)
        self.assertLess(time.monotonic() - started, 10)
        with self.assertRaises(PreparationError):
            _run_process([sys.executable, "-c", "raise SystemExit(3)"])

    def demucs_ready(self):
        return patch("audio_to_midi.provision.ensure"), patch("importlib.metadata.version", return_value="test")

    def test_demucs_install_failure_creates_no_revision(self):
        store = ArtifactStore(self.root / "artifacts")
        with patch("audio_to_midi.provision.ensure", side_effect=PreparationError("Installing demucs failed")), \
                patch("audio_to_midi.pipeline._run_process") as process:
            with self.assertRaisesRegex(PreparationError, "Installing demucs"):
                separate_mix(store, self.mix)
        process.assert_not_called()
        self.assertNotIn("separation", store.manifest["active"])

    def test_demucs_validates_outputs_and_caches_without_inference(self):
        store = ArtifactStore(self.root / "artifacts")
        calls = []
        bad = [False]

        def fake_process(argv, **kwargs):
            calls.append(argv)
            self.assertEqual(argv[0], sys.executable)
            self.assertEqual(argv[1:5], ["-X", "utf8", "-m", "audio_to_midi.separation_worker"])
            self.assertEqual(Path(argv[-2]), self.mix)
            output = Path(argv[-1])
            output.mkdir(parents=True)
            audio, rate = sf.read(self.mix, always_2d=True)
            for role in ("drums", "bass", "other", "vocals", "guitar", "piano"):
                sf.write(output / (role + ".wav"), audio[:-10] if bad[0] else audio,
                         rate, subtype="FLOAT")

        ensure, version = self.demucs_ready()
        with ensure, version, patch("audio_to_midi.pipeline._run_process", side_effect=fake_process):
            result = separate_mix(store, self.mix)
            self.assertEqual(len(result["outputs"]), 6)
            self.assertEqual(result["id"], separate_mix(store, self.mix)["id"])
            self.assertEqual(len(calls), 1)
            bad[0] = True
            with self.assertRaisesRegex(PreparationError, "mismatch"):
                separate_mix(store, self.mix, force_new_revision=True)
        self.assertEqual(store.manifest["active"]["separation"], result["id"])
        self.assertEqual(store.manifest["revisions"][-1]["status"], "failed")
        self.assertFalse(list(store.root.glob(".stage-*")))

    def test_demucs_accepts_stems_matching_a_decoded_mp3_mix(self):
        # MP3 headers can overstate the length; stems match what decoders return.
        mp3 = self.root / "mix.mp3"
        audio, rate = sf.read(self.mix, always_2d=True)
        sf.write(mp3, audio, rate, format="MP3")
        mp3.write_bytes(mp3.read_bytes()[: mp3.stat().st_size * 3 // 4])
        decoded, _ = sf.read(mp3, always_2d=True)
        self.assertNotEqual(sf.info(mp3).frames, len(decoded))
        store = ArtifactStore(self.root / "artifacts")

        def fake_process(argv, **kwargs):
            output = Path(argv[-1])
            output.mkdir(parents=True)
            for role in ("drums", "bass", "other", "vocals", "guitar", "piano"):
                sf.write(output / (role + ".wav"), decoded, rate, subtype="FLOAT")

        ensure, version = self.demucs_ready()
        with ensure, version, patch("audio_to_midi.pipeline._run_process", side_effect=fake_process):
            self.assertEqual(len(separate_mix(store, mp3)["outputs"]), 6)

    def test_demucs_cancellation_cleans_staging_and_records_state(self):
        store = ArtifactStore(self.root / "artifacts")

        def cancelled_process(argv, **kwargs):
            output = Path(argv[-1])
            output.mkdir()
            (output / "partial.wav").write_bytes(b"incomplete")
            raise Cancelled("specific process was stopped")

        ensure, version = self.demucs_ready()
        with ensure, version, patch("audio_to_midi.pipeline._run_process", side_effect=cancelled_process):
            with self.assertRaises(Cancelled):
                separate_mix(store, self.mix)
        reopened = ArtifactStore(store.root)
        self.assertEqual(reopened.manifest["revisions"][-1]["status"], "cancelled")
        self.assertNotIn("separation", reopened.manifest["active"])
        self.assertFalse(list(store.root.glob(".stage-*")))


if __name__ == "__main__":
    unittest.main()
