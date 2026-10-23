"""Synthetic/offline contracts for the optional experimental preparation API."""

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
    ArtifactStore, Cancelled, PreparationError, RevisionConflict, WarpMap,
    _run_process, analyze_mix, build_warp_map, prepare_mix, render_full_mix,
    separate_mix,
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

    def analysis(self):
        return {
            "confirmed_bpm": 120, "confirmed_meter": [12, 8],
            "downbeat_seconds": 0.5, "duration_seconds": 3,
            "beat_anchors": [(0.5, 0), (1.25, 1), (2, 2)],
        }

    def test_mapping_strict_forward_inverse_and_exact_anchors(self):
        mapping = WarpMap((0.5, 1.25, 2, 3), (0, 0.5, 1, 2))
        for source, target in zip(mapping.source_seconds, mapping.target_seconds):
            self.assertLess(abs(mapping.forward(source) - target), 1 / self.rate)
            self.assertLess(abs(mapping.inverse(target) - source), 1 / self.rate)
        points = np.linspace(0.5, 3, 500)
        np.testing.assert_allclose(mapping.inverse(mapping.forward(points)), points)
        for source, target in [
            ((0, 0), (0, 1)), ((1, 0), (0, 1)),
            ((0, 1), (0, float("nan"))), ((0, 1), (1, 0)),
            ((0,), (0,)), ((-1, 1), (0, 1)),
        ]:
            with self.assertRaises(ValueError):
                WarpMap(source, target)
        for value in (-1, 4, float("nan")):
            with self.assertRaises(ValueError):
                mapping.forward(value)

    def test_lead_in_and_compound_meter_quarter_note_bpm(self):
        trimmed = build_warp_map(self.analysis(), lead_in="trim")
        padded = build_warp_map(self.analysis(), lead_in="pad")
        self.assertEqual(trimmed.source_seconds[0], 0.5)
        self.assertEqual(trimmed.forward(0.5), 0)
        self.assertAlmostEqual(trimmed.forward(1.25), 0.5)
        self.assertEqual(padded.source_seconds[0], 0)
        self.assertAlmostEqual(padded.forward(0.5), 1 / 3)
        self.assertAlmostEqual(padded.forward(1.25) - padded.forward(0.5), 0.5)
        with self.assertRaises(ValueError):
            build_warp_map(self.analysis(), lead_in="silent-trim")
        data = self.analysis()
        data["confirmed_meter"] = None
        with self.assertRaises(ValueError):
            build_warp_map(data, lead_in="trim")

    def test_candidate_analysis_does_not_invent_downbeats(self):
        result = analyze_mix(self.mix)
        self.assertIsNone(result["candidate_downbeats"])
        self.assertIsNone(result["confirmed_bpm"])
        self.assertIsNone(result["downbeat_seconds"])
        confirmed = analyze_mix(self.mix, bpm=120, meter=(12, 8), downbeat_seconds=0.5,
                                beat_anchors=[(1.25, 1), (2, 2)])
        self.assertEqual(confirmed["beat_anchors"][0], (0.5, 0))
        self.assertEqual(confirmed["confirmed_meter"], (12, 8))
        for kwargs in [
            {"bpm": 0}, {"meter": (4, 3)}, {"downbeat_seconds": 4},
            {"beat_anchors": [(1, 2), (2, 1)]},
            {"beat_anchors": [(1, 0)], "downbeat_seconds": 1.5},
        ]:
            with self.assertRaises(ValueError):
                analyze_mix(self.mix, **kwargs)

    def test_variable_continuous_warp_pitch_and_stereo(self):
        mapping = WarpMap((0, 1, 2, 3), (0, 0.8, 2.2, 3.5))
        output = self.root / "warped.wav"
        progress = []
        metadata = render_full_mix(self.mix, output, mapping, lead_in="pad", progress=progress.append)
        audio, rate = sf.read(output, always_2d=True)
        self.assertEqual(rate, self.rate)
        self.assertEqual(len(audio), round(3.5 * rate))
        self.assertEqual(metadata["quality"], "experimental")
        self.assertEqual(progress[-1], 1)
        np.testing.assert_allclose(audio[:, 1], audio[:, 0] * 0.5, atol=1e-6)
        # Inspect every local-rate region, not just the dominant whole-file FFT.
        for start, stop in ((0.2, 0.6), (1.1, 1.8), (2.5, 3.2)):
            segment = audio[round(start * rate):round(stop * rate), 0]
            frequencies = np.fft.rfftfreq(len(segment), 1 / rate)
            peak = frequencies[np.argmax(abs(np.fft.rfft(segment * np.hanning(len(segment)))))]
            self.assertLess(abs(peak - 440), 4)
        # A continuous oscillator must not reset at either anchor.
        for anchor in (0.8, 2.2):
            index = round(anchor * rate)
            self.assertLess(np.max(abs(np.diff(audio[index - 50:index + 50, 0]))), 0.08)

    def test_stereo_quadrature_and_memory_bound(self):
        t = np.arange(self.rate * 3) / self.rate
        sf.write(self.mix, np.column_stack((0.2 * np.sin(2 * np.pi * 440 * t),
                                           0.2 * np.cos(2 * np.pi * 440 * t))),
                 self.rate, subtype="FLOAT")
        output = self.root / "quadrature.wav"
        mapping = WarpMap((0, 1.5, 3), (0, 1, 4))
        render_full_mix(self.mix, output, mapping, lead_in="pad")
        audio, _ = sf.read(output, always_2d=True)
        interior = audio[self.rate:3 * self.rate]
        self.assertLess(abs(np.corrcoef(interior.T)[0, 1]), 0.02)
        rms = np.sqrt(np.mean(interior ** 2, axis=0))
        self.assertLess(abs(rms[0] - rms[1]), 0.002)
        with self.assertRaises(PreparationError):
            render_full_mix(self.mix, output, mapping, lead_in="pad", max_working_bytes=1)

    def test_render_cancellation_never_overwrites_destination(self):
        output = self.root / "existing.wav"
        output.write_bytes(b"old artifact")
        calls = []

        def cancel():
            calls.append(True)
            return len(calls) >= 3

        with self.assertRaises(Cancelled):
            render_full_mix(self.mix, output, WarpMap((0, 3), (0, 4)),
                            lead_in="pad", cancel=cancel)
        self.assertEqual(output.read_bytes(), b"old artifact")
        self.assertFalse(list(self.root.glob(".render-*")))

    def test_streaming_identity_including_boundaries(self):
        rate = 8000
        generator = np.random.default_rng(17)
        audio = generator.normal(0, 0.1, size=(rate * 2 + 177, 2))
        sf.write(self.mix, audio, rate, subtype="FLOAT")
        duration = len(audio) / rate
        output = self.root / "identity.wav"
        budget = 16 * 1024 * 1024
        metadata = render_full_mix(
            self.mix, output, WarpMap((0, duration), (0, duration)), lead_in="pad",
            max_working_bytes=budget,
        )
        actual, _ = sf.read(output, always_2d=True)
        expected, _ = sf.read(self.mix, always_2d=True)
        np.testing.assert_allclose(actual, expected, atol=2e-6)
        self.assertLessEqual(metadata["working_bytes_reserved"], budget)
        # A rejected budget must neither replace media nor leak staging files.
        previous = output.read_bytes()
        with self.assertRaises(PreparationError):
            render_full_mix(
                self.mix, output, WarpMap((0, duration), (0, duration)), lead_in="pad",
                max_working_bytes=metadata["working_bytes_reserved"] - 1,
            )
        self.assertEqual(output.read_bytes(), previous)
        self.assertFalse(list(self.root.glob(".render-*")))

    def test_full_song_streaming_small_budget_pitch_stereo_and_peak_allocations(self):
        import tracemalloc

        rate, duration = 8000, 120
        # Build and inspect via blocks as well: the fixture must not mask an
        # accidental full-file allocation by the renderer.
        with sf.SoundFile(self.mix, "w", samplerate=rate, channels=2, subtype="FLOAT") as source:
            for second in range(duration):
                t = (np.arange(rate) + second * rate) / rate
                tone = 0.2 * np.sin(2 * np.pi * 440 * t)
                source.write(np.column_stack((tone, tone * 0.5)))
        output = self.root / "long.wav"
        budget = 16 * 1024 * 1024
        mapping = WarpMap((0, 40, 80, 120), (0, 30, 85, 130))
        progress = []
        tracemalloc.start()
        try:
            metadata = render_full_mix(
                self.mix, output, mapping, lead_in="pad", max_working_bytes=budget,
                progress=progress.append,
            )
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, budget)
        self.assertLessEqual(metadata["working_bytes_reserved"], budget)
        self.assertEqual(sf.info(output).frames, 130 * rate)
        self.assertEqual(progress[-1], 1)
        self.assertTrue(all(a <= b for a, b in zip(progress, progress[1:])))
        with sf.SoundFile(output) as rendered:
            for second in (5, 35, 70, 100, 128):
                rendered.seek(second * rate)
                block = rendered.read(rate, always_2d=True)
                np.testing.assert_allclose(block[:, 1], block[:, 0] * 0.5, atol=1e-6)
                frequencies = np.fft.rfftfreq(len(block), 1 / rate)
                peak_frequency = frequencies[np.argmax(abs(np.fft.rfft(
                    block[:, 0] * np.hanning(len(block)),
                )))]
                self.assertLess(abs(peak_frequency - 440), 3)
        self.assertFalse(list(self.root.glob(".render-*")))

    def test_streaming_cancel_during_render_and_nonfinite_cleanup(self):
        destination = self.root / "previous.wav"
        destination.write_bytes(b"keep previous")
        progress = []
        with self.assertRaises(Cancelled):
            render_full_mix(
                self.mix, destination, WarpMap((0, 3), (0, 4)), lead_in="pad",
                progress=progress.append, cancel=lambda: bool(progress) and progress[-1] > 0.4,
                max_working_bytes=16 * 1024 * 1024,
            )
        self.assertEqual(destination.read_bytes(), b"keep previous")
        self.assertFalse(list(self.root.glob(".render-*")))
        audio = np.zeros((self.rate, 2))
        audio[-1, 0] = np.nan
        sf.write(self.mix, audio, self.rate, subtype="FLOAT")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            render_full_mix(
                self.mix, destination, WarpMap((0, 1), (0, 1)), lead_in="pad",
            )
        self.assertEqual(destination.read_bytes(), b"keep previous")
        self.assertFalse(list(self.root.glob(".render-*")))

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
            downstream = self.store_stage(store, "warp", dependencies=[upstream["id"]])
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

    def test_prepare_persists_map_and_reuses_audio(self):
        kwargs = dict(bpm=120, meter=(4, 4), downbeat_seconds=0.5,
                      beat_anchors=[(1.25, 1), (2, 2)], lead_in="trim")
        result = prepare_mix(self.mix, self.root / "project", **kwargs)
        warp = result["warp_revision"]
        self.assertEqual(warp["metadata"]["bar_one_seconds"], 0)
        self.assertEqual(warp["metadata"]["removed_lead_in_seconds"], 0.5)
        again = prepare_mix(self.mix, self.root / "project", **kwargs)
        self.assertEqual(warp["id"], again["warp_revision"]["id"])
        self.assertTrue(again["store"].valid(again["warp_revision"]))
        inspected = prepare_mix(self.mix, self.root / "inspect")
        self.assertIsNone(inspected["warp_revision"])

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

    def fake_warp(self, store):
        def builder(staging):
            audio, rate = sf.read(self.mix, always_2d=True)
            sf.write(staging / "mix.wav", audio, rate, subtype="FLOAT")
            return {"full_mix": "mix.wav"}, {}
        return store.run("warp", inputs=[self.mix], parameters={}, backend="test", builder=builder)

    def demucs_ready(self):
        return patch("audio_to_midi.provision.ensure"), patch("importlib.metadata.version", return_value="test")

    def test_demucs_install_failure_creates_no_revision(self):
        store = ArtifactStore(self.root / "artifacts")
        warp = self.fake_warp(store)
        with patch("audio_to_midi.provision.ensure", side_effect=PreparationError("Installing demucs failed")), \
                patch("audio_to_midi.pipeline._run_process") as process:
            with self.assertRaisesRegex(PreparationError, "Installing demucs"):
                separate_mix(store, warp["id"])
        process.assert_not_called()
        self.assertNotIn("separation", store.manifest["active"])

    def test_demucs_validates_outputs_and_caches_without_inference(self):
        store = ArtifactStore(self.root / "artifacts")
        warp = self.fake_warp(store)
        calls = []
        bad = [False]

        def fake_process(argv, **kwargs):
            calls.append(argv)
            self.assertEqual(argv[0], sys.executable)
            self.assertEqual(argv[1:5], ["-X", "utf8", "-m", "audio_to_midi.separation_worker"])
            self.assertEqual(Path(argv[-2]), store.output(warp, "full_mix"))
            output = Path(argv[-1])
            output.mkdir(parents=True)
            audio, rate = sf.read(self.mix, always_2d=True)
            for role in ("drums", "bass", "other", "vocals", "guitar", "piano"):
                sf.write(output / (role + ".wav"), audio[:-10] if bad[0] else audio,
                         rate, subtype="FLOAT")

        ensure, version = self.demucs_ready()
        with ensure, version, patch("audio_to_midi.pipeline._run_process", side_effect=fake_process):
            result = separate_mix(store, warp["id"])
            self.assertEqual(len(result["outputs"]), 6)
            self.assertEqual(result["id"], separate_mix(store, warp["id"])["id"])
            self.assertEqual(len(calls), 1)
            bad[0] = True
            with self.assertRaisesRegex(PreparationError, "mismatch"):
                separate_mix(store, warp["id"], force_new_revision=True)
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

        def builder(staging):
            (staging / "mix.mp3").write_bytes(mp3.read_bytes())
            return {"full_mix": "mix.mp3"}, {}

        warp = store.run("warp", inputs=[mp3], parameters={}, backend="test", builder=builder)

        def fake_process(argv, **kwargs):
            output = Path(argv[-1])
            output.mkdir(parents=True)
            for role in ("drums", "bass", "other", "vocals", "guitar", "piano"):
                sf.write(output / (role + ".wav"), decoded, rate, subtype="FLOAT")

        ensure, version = self.demucs_ready()
        with ensure, version, patch("audio_to_midi.pipeline._run_process", side_effect=fake_process):
            self.assertEqual(len(separate_mix(store, warp["id"])["outputs"]), 6)

    def test_demucs_cancellation_cleans_staging_and_records_state(self):
        store = ArtifactStore(self.root / "artifacts")
        warp = self.fake_warp(store)

        def cancelled_process(argv, **kwargs):
            output = Path(argv[-1])
            output.mkdir()
            (output / "partial.wav").write_bytes(b"incomplete")
            raise Cancelled("specific process was stopped")

        ensure, version = self.demucs_ready()
        with ensure, version, patch("audio_to_midi.pipeline._run_process", side_effect=cancelled_process):
            with self.assertRaises(Cancelled):
                separate_mix(store, warp["id"])
        reopened = ArtifactStore(store.root)
        self.assertEqual(reopened.manifest["revisions"][-1]["status"], "cancelled")
        self.assertNotIn("separation", reopened.manifest["active"])
        self.assertFalse(list(store.root.glob(".stage-*")))


if __name__ == "__main__":
    unittest.main()
