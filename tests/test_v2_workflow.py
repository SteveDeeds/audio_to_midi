import contextlib
from copy import deepcopy
from fractions import Fraction as F
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mido
import numpy as np
import soundfile as sf

from audio_to_midi import application as app
from audio_to_midi.adapters import Cancelled, Prediction, TrackBackend
from audio_to_midi.domain import Note, NoteStore, Project
from audio_to_midi.midi import export_midi, import_notes
from audio_to_midi.storage import load_project, save_project
from audio_to_midi.timebase import Grid
from audio_to_midi.__main__ import main


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.wav = self.root / "stem.wav"
        sf.write(self.wav, np.zeros(16000, dtype=np.float32), 8000)
        self.project = Project.create("Test", Grid(120, 6, 8, F(1, 6)))
        self.project.key = "C"
        self.track = app.add_stem(self.project, self.wav, "bass")

    def test_role_models_and_alignment(self):
        with self.assertRaisesRegex(ValueError, "already exists"):
            app.add_stem(self.project, self.wav, "bass")
        other = app.add_stem(self.project, self.wav, "other")
        self.assertEqual(other.model, "")
        self.assertTrue(app.complete(self.project, other))
        sf.write(self.root / "short.wav", np.zeros(8000), 8000)
        with self.assertRaisesRegex(ValueError, "duration differs"):
            app.add_stem(self.project, self.root / "short.wav", "vocals")

    def test_publication_preserves_manual_and_rejections(self):
        self.track.notes.author(F(0), F(1), 60)
        self.track.notes.reject(F(1), F(2), 62)
        snapshot = app.token(self.project, self.track)
        prediction = Prediction(F(0), F(4), (
            Note("generated", F(0), F(4), 60),
            Note("rejected", F(1), F(2), 62),
        ), (F(0), F(1)))
        self.assertTrue(app.publish(self.project, snapshot, prediction))
        self.assertEqual([(n.start, n.end, n.pitch) for n in self.track.notes.resolved()], [
            (F(0), F(1), 60), (F(1), F(4), 60),
        ])
        self.assertTrue(app.complete(self.project, self.track))

    def test_replacing_stem_validates_alignment_before_mutating(self):
        before = deepcopy(self.track)
        for duration, rate in ((1, 8000), (2, 16000)):
            with self.subTest(duration=duration, rate=rate):
                replacement = self.root / "replacement.wav"
                sf.write(replacement, np.zeros(duration * rate), rate)
                with self.assertRaisesRegex(ValueError, "duration differs"):
                    app.replace_stem(self.project, self.track, replacement)
                self.assertEqual(self.track, before)
        self.project.grid = Grid(offset_seconds=2)
        with self.assertRaisesRegex(ValueError, "downbeat"):
            app.replace_stem(self.project, self.track, self.wav)
        self.assertEqual(self.track, before)

    def test_stale_results_after_edit_settings_asset_or_project_change(self):
        prediction = Prediction(F(0), F(4), (), ())
        for change in (
            lambda: self.track.notes.author(F(0), F(1), 60),
            lambda: setattr(self.track, "model", "pyin"),
            lambda: self.track.training.append({"domain": "onset", "beat": "1", "keep": False}),
            lambda: setattr(self.project, "id", "different"),
        ):
            snapshot = app.token(self.project, self.track)
            change()
            self.assertFalse(app.publish(self.project, snapshot, prediction))

    def test_empty_coverage_is_not_unprocessed_and_survives_reopen(self):
        self.assertFalse(app.complete(self.project, self.track))
        result = Prediction(F(0), F(4), (), ())
        self.assertTrue(app.publish(self.project, app.token(self.project, self.track), result))
        self.assertTrue(app.complete(self.project, self.track))
        save_project(self.project, self.root / "project")
        loaded = load_project(self.root / "project")
        self.assertTrue(app.complete(loaded, loaded.tracks[0]))
        loaded.tracks[0].model = "pyin"
        self.assertFalse(app.complete(loaded, loaded.tracks[0]))

    def test_bounded_coverage_merges_without_false_full_completion(self):
        for a, b in [(F(2), F(3)), (F(0), F(2))]:
            app.publish(self.project, app.token(self.project, self.track), Prediction(a, b, (), ()))
        self.assertFalse(app.complete(self.project, self.track))
        app.publish(self.project, app.token(self.project, self.track), Prediction(F(3), F(4), (), ()))
        self.assertTrue(app.complete(self.project, self.track))

    def test_midi_triplets_meter_velocity_channels_and_restrikes(self):
        self.track.notes.author(F(0), F(1, 3), 60, 76)
        self.track.notes.author(F(1, 3), F(2, 3), 60, 111)
        drum = app.add_stem(self.project, self.wav, "drums")
        drum.notes.author(F(0), F(1, 6), 36, 100)
        other = app.add_stem(self.project, self.wav, "other")
        out = self.root / "song.mid"
        export_midi(self.project, out)
        midi = mido.MidiFile(out)
        self.assertEqual((midi.tracks[0][2].numerator, midi.tracks[0][2].denominator), (6, 8))
        on = [m for track in midi.tracks for m in track if m.type == "note_on"]
        self.assertEqual([m.velocity for m in on], [76, 111, 100])
        self.assertEqual(on[-1].channel, 9)
        self.assertTrue(all(m.channel != 9 for m in on[:-1]), "only drums use GM channel 10")
        drums_only = self.root / "drums.mid"
        export_midi(self.project, drums_only, tracks=[drum], include_chords=False)
        self.assertEqual({m.channel for t in mido.MidiFile(drums_only).tracks for m in t if m.type == "note_on"}, {9})
        self.assertEqual(len(midi.tracks), 3)
        events = [(m.type, m.time) for m in midi.tracks[1] if m.type in ("note_on", "note_off")]
        self.assertEqual(events, [("note_on", 0), ("note_off", 320), ("note_on", 0), ("note_off", 320)])
        self.assertFalse(any(m.type == "pitchwheel" for track in midi.tracks for m in track))
        self.assertTrue(app.complete(self.project, other))

    def test_starting_midi_is_authoritative_and_honors_tempo_events(self):
        midi = mido.MidiFile()
        lane = mido.MidiTrack()
        lane.extend([
            mido.MetaMessage("set_tempo", tempo=500000),
            mido.Message("note_on", note=60, velocity=91),
            mido.MetaMessage("set_tempo", tempo=1000000, time=480),
            mido.Message("note_off", note=60, time=480),
        ])
        midi.tracks.append(lane)
        path = self.root / "start.mid"
        midi.save(path)
        self.assertEqual(import_notes(path, self.project, self.track), 1)
        self.assertEqual(self.track.notes.resolved()[0].end, F(3))
        positives = {row["beat"] for row in self.track.training if row["keep"]}
        negatives = {row["beat"] for row in self.track.training if not row["keep"]}
        self.assertEqual(positives, {"0", "3"})
        self.assertIn("7/2", negatives)
        self.assertNotIn("1", negatives)
        self.assertTrue(app.complete(self.project, self.track))
        result = Prediction(F(0), F(4), (Note("g", F(0), F(3), 60),), ())
        app.publish(self.project, app.token(self.project, self.track), result)
        self.assertEqual(len(self.track.notes.resolved()), 1)

    def test_delete_different_pitch_onset_extends_preceding_note_bounded(self):
        notes = NoteStore(generated=[
            Note("a", F(0), F(1), 60), Note("b", F(1), F(2), 64),
            Note("c", F(3), F(4), 67),
        ], onsets=[F(0), F(1), F(3)])
        notes.delete_onset(F(1), F(2))
        self.assertEqual([(n.start, n.end, n.pitch) for n in notes.resolved()], [
            (F(0), F(2), 60), (F(3), F(4), 67),
        ])
        self.assertTrue(notes.undo())
        self.assertEqual(len(notes.resolved()), 3)

    def test_transcriber_checks_cancel_before_loading_models(self):
        backend = TrackBackend.create(self.track, self.project)
        with self.assertRaises(Cancelled):
            backend.predict(self.track, F(0), F(4), lambda: True)
        self.assertIsNone(backend.provider)

    def test_mono_adapter_uses_onset_blocks_not_model_note_boundaries(self):
        class Mono:
            def prepare(self, *args, **kwargs):
                return True
            def onset_times(self, *args):
                return np.array([0, .5, 1])
            def pitch_changes(self, *args):
                return []
            def block_pitch(self, a, b, *args):
                return [60], b
        backend = TrackBackend.create(self.track, self.project)
        backend.provider = Mono()
        prediction = backend.predict(self.track, F(0), F(4), lambda: False)
        self.assertEqual([n.start for n in prediction.notes], [F(0), F(1), F(2)])
        self.assertEqual([n.end for n in prediction.notes], [F(1), F(2), F(4)])

    def test_stable_mono_changes_split_bass_but_deleted_onsets_stay_deleted(self):
        class Mono:
            def prepare(self, *args, **kwargs):
                return True
            def onset_times(self, *args):
                return np.array([0])
            def pitch_changes(self, *args):
                return [.5]
            def block_pitch(self, a, b, *args):
                return [60 if a < .5 else 62], b
        backend = TrackBackend.create(self.track, self.project)
        backend.provider = Mono()
        result = backend.predict(self.track, F(0), F(4), lambda: False)
        self.assertEqual([n.pitch for n in result.notes], [60, 62])
        app.record_onset(self.track, F(1), False)
        with patch.object(backend, "_candidate_features", return_value=(.1, .1, -10., .5)):
            result = backend.predict(self.track, F(0), F(4), lambda: False)
        self.assertNotIn(F(1), result.onsets)

    def test_visible_batches_merge_continuation_without_false_restrike(self):
        class Mono:
            def prepare(self, *args, **kwargs):
                return True
            def onset_times(self, *args):
                return np.array([0])
            def pitch_changes(self, *args):
                return []
            def block_pitch(self, a, b, *args):
                return [60], b
        backend = TrackBackend.create(self.track, self.project)
        backend.provider = Mono()
        for a, b in ((F(0), F(2)), (F(2), F(4))):
            snapshot = app.token(self.project, self.track)
            result = backend.predict(self.track, a, b, lambda: False)
            app.publish(self.project, snapshot, result)
        self.assertEqual([(n.start, n.end) for n in self.track.notes.resolved()], [(F(0), F(4))])

    def test_repeated_author_and_rejection_are_idempotent(self):
        a = self.track.notes.author(F(0), F(1), 60)
        revision = self.track.notes.revision
        b = self.track.notes.author(F(0), F(1), 60)
        self.assertEqual(a.id, b.id)
        self.assertEqual(self.track.notes.revision, revision)
        self.track.notes.reject(F(0), F(1), 60)
        revision = self.track.notes.revision
        self.track.notes.reject(F(0), F(1), 60)
        self.assertEqual(self.track.notes.revision, revision)

    def test_key_override_parsing(self):
        from audio_to_midi.adapters import normalize_key
        self.assertEqual(normalize_key("A minor"), "Am")
        self.assertEqual(normalize_key("Bb major"), "Bb")
        self.assertEqual(normalize_key("Am (C)"), "Am")
        with self.assertRaises(ValueError):
            normalize_key("unknown")

    def test_prepare_attach_and_edited_revision_protection(self):
        from audio_to_midi.pipeline import RevisionConflict, prepare_mix
        args = dict(bpm=120, meter=(4, 4), downbeat_seconds=0,
                    beat_anchors=[(0, 0), (1, 2), (2, 4)], lead_in="trim")
        result = prepare_mix(self.wav, self.root / "artifacts", **args)
        project = Project.create("Prepared", Grid())
        app.attach_prepared(project, self.root / "artifacts")
        self.assertEqual(project.grid.bpm, 120)
        self.assertTrue(Path(project.mix_path).is_file())
        with self.assertRaisesRegex(ValueError, "overwrite"):
            export_midi(project, self.wav)
        app.mark_project_edited(project)
        with self.assertRaises(RevisionConflict):
            prepare_mix(self.wav, self.root / "artifacts", **{**args, "bpm": 100})
        self.assertTrue(result["store"].output(result["warp_revision"], "full_mix").is_file())

    def test_cannot_render_a_fixed_tempo_from_a_downbeat_alone(self):
        from audio_to_midi.pipeline import prepare_mix
        with self.assertRaisesRegex(ValueError, "at least two"):
            prepare_mix(self.wav, self.root / "artifacts", bpm=120, meter=(4, 4),
                        downbeat_seconds=0, lead_in="trim")

    def test_external_warp_registration_never_rewarps_or_invents_a_map(self):
        from audio_to_midi.pipeline import register_warped_mix
        with patch("audio_to_midi.pipeline.render_full_mix", side_effect=AssertionError("must not rewarp")):
            result = register_warped_mix(self.wav, self.root / "external", bpm=75, meter=(12, 8), offset_seconds=.1)
        copied = result["store"].output(result["warp_revision"], "full_mix")
        self.assertEqual(copied.read_bytes(), self.wav.read_bytes())
        self.assertIsNone(result["warp_revision"]["metadata"]["warp_map"])
        project = Project.create("External", Grid())
        app.attach_prepared(project, self.root / "external")
        self.assertEqual((project.grid.bpm, project.grid.numerator, project.grid.denominator), (75, 12, 8))
        self.assertEqual(project.grid.offset_seconds, .1)

    def test_outputs_cannot_overwrite_original_audio(self):
        from audio_to_midi.pipeline import WarpMap, render_full_mix
        original = self.wav.read_bytes()
        with self.assertRaisesRegex(ValueError, "overwrite"):
            export_midi(self.project, self.wav)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            render_full_mix(self.wav, self.wav, WarpMap((0, 2), (0, 2)), lead_in="trim")
        self.assertEqual(self.wav.read_bytes(), original)

    def test_exports_cannot_overwrite_original_mix_after_warping(self):
        from audio_to_midi.domain import ChordSegment
        from audio_to_midi.midi import export_musicxml

        original = self.root / "original.wav"
        original.write_bytes(self.wav.read_bytes())
        self.project.mix_path = str(self.wav)
        self.project.stages["timing"] = {"source": str(original)}
        self.project.chords = [ChordSegment(F(0), F(4), "C")]
        before = original.read_bytes()
        for export in (export_midi, export_musicxml):
            with self.subTest(export=export.__name__):
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    export(self.project, original)
                self.assertEqual(original.read_bytes(), before)

    def test_harmony_audio_changes_invalidate_cache_and_pending_results(self):
        other_path = self.root / "other.wav"
        sf.write(other_path, np.zeros(16000), 8000)
        other = app.add_stem(self.project, other_path, "other")
        before = app.token(self.project, self.track)
        import os
        stat = other_path.stat()
        os.utime(other_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        self.assertNotEqual(before, app.token(self.project, self.track))
        self.assertFalse(app.publish(self.project, before, Prediction(F(0), F(4), (), ())))

    def test_environment_doctor_does_not_load_optional_models(self):
        from audio_to_midi.environment import inspect_environment
        with patch("audio_to_midi.environment.import_module") as imports:
            report = inspect_environment()
        self.assertTrue(report["base_ready"])
        self.assertEqual(len(imports.call_args_list), 7)
        self.assertFalse(any(call.args[0] in ("torchcrepe", "basic_pitch", "demucs") for call in imports.call_args_list))
        import json
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["doctor"]), 0)
        self.assertTrue(json.loads(output.getvalue())["base_ready"])

    def test_cli_exports_individual_role_and_chords_without_other_tracks(self):
        from audio_to_midi.domain import ChordSegment
        self.project.chords = [ChordSegment(F(0), F(1), "E")]
        self.track.notes.author(F(0), F(1), 60)
        self.track.settings["imported_midi"] = "source.mid"
        folder = self.root / "project"
        save_project(self.project, folder)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["export", str(folder), str(self.root / "bass.mid"), "--role", "bass"]), 0)
            self.assertEqual(main(["export", str(folder), str(self.root / "chords.mid"), "--role", "chords"]), 0)
            self.assertEqual(main(["export", str(folder), str(self.root / "chords.musicxml"), "--role", "chords"]), 0)
        for name, expected in (("bass", [60]), ("chords", [40, 64, 68, 71])):
            midi = mido.MidiFile(self.root / (name + ".mid"))
            pitches = [msg.note for lane in midi.tracks for msg in lane if msg.type == "note_on"]
            self.assertEqual(sorted(pitches), expected)
        sheet = (self.root / "chords.musicxml").read_text(encoding="utf-8")
        self.assertIn("<root-step>E</root-step>", sheet)
        self.assertIn("<kind text=\"\">major</kind>", sheet)

    def test_basic_pitch_explicitly_uses_onnx_even_with_other_backends_installed(self):
        import sys
        from types import ModuleType, SimpleNamespace
        from core import poly_pitch

        package = ModuleType("basic_pitch")
        package.FilenameSuffix = SimpleNamespace(onnx="onnx")
        package.build_icassp_2022_model_path = lambda suffix: self.root / (suffix + ".model")
        inference = ModuleType("basic_pitch.inference")
        loaded = []
        inference.Model = lambda path: loaded.append(path) or object()
        inference.predict = lambda *args, **kwargs: (None, None, [(0, .5, 60, .8, [])])
        with patch.dict(sys.modules, {"basic_pitch": package, "basic_pitch.inference": inference}):
            with patch.object(poly_pitch, "_BASIC_PITCH_MODEL", None):
                notes = poly_pitch.transcribe_notes(np.ones(22050, dtype=np.float32), 22050, "piano", "basic_pitch")
        self.assertEqual(loaded, [self.root / "onnx.model"])
        self.assertEqual(notes, [(0., .5, 60, .8)])

    def test_cached_polyphony_carries_voice_and_restrike_splits(self):
        from core.poly_pitch import BlockNote
        class Poly:
            def prepare(self, *args, **kwargs):
                return True
            def onset_times(self, *args):
                return np.array([0, .5, 1])
            def block_notes(self, a, b, *args):
                return [BlockNote(60, b, .8, a == .5)]
        track = app.add_stem(self.project, self.wav, "piano")
        backend = TrackBackend.create(track, self.project)
        backend.provider = Poly()
        result = backend.predict(track, F(0), F(4), lambda: False)
        self.assertEqual([(n.start, n.end) for n in result.notes], [(F(0), F(2)), (F(2), F(4))])

    def test_profile_import_reuses_compatible_legacy_rows_and_reports_extras(self):
        import json
        path = self.root / "profile.json"
        path.write_text(json.dumps({"cand_X": [[.1, .2, -20, .3]], "cand_y": [1], "fp": []}))
        self.assertIn("fp", app.import_profile(self.track, path))
        self.assertIn("onset_profile", self.track.settings)

    def test_cli_create_import_export_info_and_refuse_incomplete(self):
        folder = self.root / "cli"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["create", str(folder), "--bpm", "120", "--meter", "12/8"]), 0)
            self.assertEqual(main(["add-stem", str(folder), "bass", str(self.wav)]), 0)
            self.assertEqual(main(["info", str(folder)]), 0)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(["export", str(folder), str(self.root / "bad.mid")])
            self.assertEqual(error.exception.code, 1)
            self.assertEqual(main(["export", str(folder), str(self.root / "partial.mid"), "--allow-partial"]), 0)
        self.assertTrue((self.root / "partial.mid").exists())

    def test_shared_legacy_features_retain_expected_units(self):
        from core import onset_features
        times = np.arange(0, 1, .1)
        envelope = np.array([0, 0, 0, 0, .1, .5, .2, .1, 0, 0])
        self.assertAlmostEqual(onset_features.transient_intensity(times, envelope, .3, .7), .45)
        self.assertAlmostEqual(onset_features.rms_db(np.ones(100), 100, 0, 1), 0, places=8)
        magnitude = np.ones((83, 10))
        value = onset_features.percussive_spike(times, magnitude, .5, .4)
        self.assertTrue(0 <= value <= 1)
        self.assertAlmostEqual(value, onset_features.percussive_spike(times, magnitude * 100, .5, .4))

    def test_real_pyin_synthetic_tone_end_to_end(self):
        samples = np.arange(16000) / 8000
        audio = .4 * np.sin(2 * np.pi * 440 * samples)
        audio[:400] *= np.linspace(0, 1, 400)
        audio[-400:] *= np.linspace(1, 0, 400)
        sf.write(self.wav, audio, 8000)
        vocals = app.add_stem(self.project, self.wav, "vocals")
        snapshot = app.token(self.project, vocals)
        prediction = app.TranscriptionService().predict(self.project, vocals, F(0), F(4))
        self.assertTrue(prediction.notes)
        self.assertTrue(all(n.pitch == 69 for n in prediction.notes))
        self.assertTrue(app.publish(self.project, snapshot, prediction))
        save_project(self.project, self.root / "tone-project")
        loaded = load_project(self.root / "tone-project")
        output = self.root / "tone.mid"
        export_midi(loaded, output, [loaded.track(vocals.id)], include_chords=False)
        parsed = mido.MidiFile(output)
        self.assertTrue(any(m.type == "note_on" and m.note == 69 for lane in parsed.tracks for m in lane))


if __name__ == "__main__":
    unittest.main()
