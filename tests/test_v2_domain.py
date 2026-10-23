"""Public-behavior specification for the headless v2 domain and manifest."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as F
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

from audio_to_midi import Grid, Note, NoteStore, Project, Track
from audio_to_midi.storage import (
    InvalidProjectError, MissingMediaError, MissingProjectError,
    SCHEMA_VERSION, SchemaVersionError, load_project, save_project,
)


def note(identity="g", start=0, end=4, pitch=60, velocity=100):
    return Note(identity, F(start), F(end), pitch, velocity)


def spans(store):
    return [(n.start, n.end, n.pitch) for n in store.resolved()]


class GridTests(unittest.TestCase):
    def test_defaults_and_frozen(self):
        grid = Grid()
        self.assertEqual(grid.bar_beats, F(4))
        self.assertEqual(grid.subdivision, F(1, 4))
        with self.assertRaises(FrozenInstanceError):
            grid.bpm = 90

    def test_compound_meter_is_quarter_units(self):
        for numerator, expected in ((6, F(3)), (12, F(6))):
            grid = Grid(120, numerator, 8)
            self.assertEqual(grid.bar_beats, expected)
            self.assertEqual(grid.beat_to_seconds(expected), float(expected) / 2)

    def test_triplets_offset_and_round_trip(self):
        grid = Grid(137.25, 6, 8, F(1, 3), 0.173)
        self.assertEqual(grid.seconds_to_beat(0.173), F(0))
        for beat in (F(-1, 3), F(1, 3), F(1000, 7), F(100000, 3)):
            seconds = grid.beat_to_seconds(beat)
            result = grid.seconds_to_beat(seconds)
            self.assertLess(abs(grid.beat_to_seconds(result) - seconds), 1 / 192000)
        self.assertEqual(grid.snap(F(34, 100)), F(1, 3))
        self.assertEqual(grid.snap(F(1, 6)), F(1, 3))
        self.assertEqual(grid.snap(F(-1, 6)), F(0))
        self.assertEqual(Grid().snap(F(3, 8)), F(1, 2))

    def test_invalid_grid(self):
        cases = [
            {"bpm": 0}, {"bpm": -1}, {"bpm": float("nan")},
            {"bpm": float("inf")}, {"bpm": True}, {"bpm": "120"},
            {"numerator": 0}, {"numerator": 4.0}, {"numerator": True},
            {"denominator": 3}, {"denominator": 0}, {"denominator": -4},
            {"subdivision": F(0)}, {"subdivision": F(-1, 4)},
            {"subdivision": 0.25}, {"offset_seconds": float("inf")},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Grid(**kwargs)

    def test_bad_conversion_inputs(self):
        for bad in (float("nan"), float("inf"), "1", True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                Grid().seconds_to_beat(bad)
        with self.assertRaises(ValueError):
            Grid().beat_to_seconds(0.5)


class NoteTests(unittest.TestCase):
    def test_note_bounds_identity_and_frozen(self):
        value = note()
        with self.assertRaises(FrozenInstanceError):
            value.end = F(8)
        for kwargs in (
            {"id": ""}, {"start": F(-1)}, {"start": 0.0}, {"end": F(0)},
            {"pitch": -1}, {"pitch": 128}, {"pitch": True},
            {"velocity": 0}, {"velocity": 128}, {"velocity": 10.0}, {"source": ""},
        ):
            values = dict(id="x", start=F(0), end=F(1), pitch=60)
            values.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Note(**values)

    def test_author_preserves_other_voices_and_splits_generated(self):
        store = NoteStore(generated=[note(), note("voice", pitch=64)])
        authored = store.author(F(1), F(2), 60, 87)
        self.assertIn(authored, store.resolved())
        self.assertEqual(spans(store), [
            (F(0), F(1), 60), (F(0), F(4), 64),
            (F(1), F(2), 60), (F(2), F(4), 60),
        ])
        self.assertEqual([n.id for n in store.resolved() if n.source == "generated"],
                         ["g", "voice", "g"])

    def test_author_restrike_clips_only_same_pitch_authored(self):
        store = NoteStore()
        first = store.author(F(0), F(4), 60)
        held = store.author(F(0), F(5), 67)
        second = store.author(F(2), F(4), 60)
        self.assertEqual(spans(store), [
            (F(0), F(2), 60), (F(0), F(5), 67), (F(2), F(4), 60),
        ])
        self.assertEqual(store.resolved()[0].id, first.id)
        self.assertIn(held, store.resolved())
        self.assertIn(second, store.resolved())

    def test_rejection_and_rest_preserve_unrelated_portions(self):
        store = NoteStore(generated=[note(), note("other", pitch=64)])
        store.reject(F(1), F(2), 60)
        self.assertEqual(spans(store), [
            (F(0), F(1), 60), (F(0), F(4), 64), (F(2), F(4), 60),
        ])
        store.reject(F(2), F(3), None)
        self.assertEqual(spans(store), [
            (F(0), F(1), 60), (F(0), F(2), 64),
            (F(3), F(4), 60), (F(3), F(4), 64),
        ])

    def test_rest_removes_authored_interval_but_later_author_wins(self):
        store = NoteStore()
        store.author(F(0), F(4), 60)
        store.reject(F(1), F(3), None)
        self.assertEqual(spans(store), [(F(0), F(1), 60), (F(3), F(4), 60)])
        manual = store.author(F(1), F(2), 60)
        self.assertIn(manual, store.resolved())
        store.replace_generated(F(0), F(4), [note(pitch=64)])
        self.assertNotIn((F(1), F(3), 64), spans(store))

    def test_publication_is_interval_bounded_and_not_undoable(self):
        store = NoteStore(generated=[note()])
        manual = store.author(F(1), F(2), 64)
        store.reject(F(3), F(4), 60)
        before = (list(store.authored), list(store.rejections), list(store.onsets))
        store.replace_generated(F(1), F(3), [note("fresh", -0, 8, 65)])
        self.assertEqual((store.authored, store.rejections, store.onsets), before)
        self.assertIn(manual, store.resolved())
        self.assertEqual([(n.id, n.start, n.end) for n in store.generated],
                         [("g", F(0), F(1)), ("fresh", F(1), F(3)), ("g", F(3), F(4))])
        self.assertTrue(store.undo())
        self.assertFalse(store.rejections)
        self.assertEqual(store.generated[1].id, "fresh")
        self.assertTrue(store.undo())
        self.assertFalse(store.authored)
        self.assertFalse(store.undo())

    def test_empty_publication_preserves_outside_interval(self):
        store = NoteStore(generated=[note()])
        store.replace_generated(F(1), F(3), [])
        self.assertEqual(spans(store), [(F(0), F(1), 60), (F(3), F(4), 60)])
        self.assertFalse(store.undo())

    def test_generated_restrike_splits_same_pitch_without_cutting_held_voice(self):
        store = NoteStore(generated=[
            note("first", 0, 4), note("next", 2, 5), note("held", 0, 6, 67),
        ])
        self.assertEqual(spans(store), [
            (F(0), F(2), 60), (F(0), F(6), 67), (F(2), F(5), 60),
        ])

    def test_same_proposal_republication_does_not_create_fake_onsets(self):
        store = NoteStore(generated=[note()])
        store.replace_generated(F(1), F(3), [note()])
        self.assertEqual(store.generated, [note()])
        self.assertEqual(store.resolved(), [note()])

    def test_release_generated_survives_inference(self):
        store = NoteStore(generated=[note()])
        store.release("g", F(2))
        self.assertEqual(spans(store), [(F(0), F(2), 60)])
        store.replace_generated(F(0), F(4), [note("new")])
        self.assertEqual(spans(store), [(F(0), F(2), 60)])
        self.assertTrue(store.undo())
        self.assertEqual(spans(store), [(F(0), F(4), 60)])
        self.assertTrue(store.redo())
        self.assertEqual(spans(store), [(F(0), F(2), 60)])

    def test_release_authored_does_not_leave_tail(self):
        store = NoteStore()
        value = store.author(F(0), F(4), 60)
        store.release(value.id, F(2))
        self.assertEqual(spans(store), [(F(0), F(2), 60)])
        self.assertEqual(store.resolved()[0].id, value.id)

    def test_release_caps_at_same_pitch_restrike_not_other_voice(self):
        store = NoteStore(generated=[
            note("first", 0, 1), note("other", 2, 8, 64), note("next", 3, 5),
        ])
        store.release("first", F(7))
        self.assertEqual(spans(store), [
            (F(0), F(3), 60), (F(2), F(8), 64), (F(3), F(5), 60),
        ])

    def test_invalid_edits_do_not_mutate(self):
        store = NoteStore(generated=[note()])
        for operation in (
            lambda: store.author(F(2), F(1), 60),
            lambda: store.reject(F(0), F(1), 200),
            lambda: store.release("g", F(0)),
            lambda: store.delete_onset(F(1), F(1)),
            lambda: store.replace_generated(F(0), F(1), [None]),
            lambda: store.add_onset(F(-1)),
        ):
            with self.assertRaises(ValueError):
                operation()
            self.assertEqual(store.revision, 0)
            self.assertEqual(store.resolved(), [note()])
            self.assertFalse(store.undo())
        with self.assertRaises(KeyError):
            store.release("missing", F(1))

    def test_delete_onset_merges_bounded_previous_and_preserves_voice(self):
        store = NoteStore(generated=[
            note("first", 0, 1), note("delete", 1, 2),
            note("next", 3, 4), note("held", 0, 5, 67),
        ], onsets=[F(0), F(1), F(3)])
        store.delete_onset(F(1), F(5))
        self.assertEqual(spans(store), [
            (F(0), F(3), 60), (F(0), F(5), 67), (F(3), F(4), 60),
        ])
        self.assertEqual(store.onsets, [F(0), F(3)])
        edited = store.resolved()
        store.undo()
        self.assertEqual(store.onsets, [F(0), F(1), F(3)])
        self.assertFalse(store.rejections)
        store.redo()
        self.assertEqual(store.resolved(), edited)

    def test_delete_last_onset_uses_input_window_forever(self):
        store = NoteStore(generated=[note("a", 0, 1), note("b", 1, 2)])
        store.delete_onset(F(1), F(7, 2))
        self.assertEqual(spans(store), [(F(0), F(7, 2), 60)])
        store.replace_generated(F(0), F(4), [note("a", 0, 1), note("b", 1, 2)])
        self.assertEqual(spans(store), [(F(0), F(7, 2), 60)])

    def test_delete_without_previous_does_not_invent_note(self):
        store = NoteStore(generated=[note("a", 1, 2)])
        store.delete_onset(F(1), F(4))
        self.assertEqual(store.resolved(), [])

    def test_onset_feedback_history_and_monotonic_generation(self):
        store = NoteStore()
        generations = [store.generation]
        store.add_onset(F(1, 3))
        generations.append(store.generation)
        store.add_onset(F(1, 3))
        self.assertEqual(store.onsets, [F(1, 3)])
        self.assertEqual(store.generation, generations[-1])
        store.undo()
        generations.append(store.generation)
        self.assertFalse(store.onsets)
        store.redo()
        generations.append(store.generation)
        store.replace_generated(F(0), F(1), [])
        generations.append(store.generation)
        self.assertEqual(generations, sorted(set(generations)))
        store.undo()
        store.add_onset(F(2))
        self.assertFalse(store.redo())

    def test_defaults_are_not_shared(self):
        left, right = NoteStore(), NoteStore()
        left.add_onset(F(1))
        self.assertFalse(right.onsets)
        a = Track("a", "bass", "", 0, 44100, "mono")
        b = Track("b", "bass", "", 0, 44100, "mono")
        a.settings["x"] = 1
        a.training.append({"label": "onset"})
        self.assertFalse(b.settings)
        self.assertFalse(b.training)


class ProjectTests(unittest.TestCase):
    def test_create_lookup_and_validation(self):
        project = Project.create("Example", Grid())
        other = Project.create("Example", Grid())
        self.assertNotEqual(project.id, other.id)
        track = Track("bass", "bass", "", 2.0, 44100, "bass__crepe_tiny+onset+key")
        project.tracks.append(track)
        self.assertIs(project.track("bass"), track)
        with self.assertRaises(KeyError):
            project.track("absent")
        for kwargs in (
            {"role": "violin"}, {"model": "imaginary"}, {"duration": -1},
            {"duration": float("inf")}, {"duration": float("nan")},
            {"sample_rate": 0}, {"sample_rate": 44100.0}, {"revision": -1},
            {"training": [{"beat": F(1, 3)}]}, {"training": [1]},
            {"training": [{"bad": float("nan")}]}, {"settings": {1: "bad"}},
        ):
            values = dict(id="t", role="bass", audio_path="", duration=2, sample_rate=44100, model="mono")
            values.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Track(**values)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.fixture = Path(__file__).resolve().parent / f".v2-domain-fixture-{uuid4().hex}"
        self.folder = self.fixture / "project"
        self.folder.mkdir(parents=True)
        self.media = self.folder / "stem.wav"
        self.media.write_bytes(b"synthetic fixture")
        self.external = self.fixture / "outside.wav"
        self.external.write_bytes(b"synthetic external fixture")
        self.project = Project.create("Música", Grid(123.5, 12, 8, F(1, 3), -0.125))
        store = NoteStore(generated=[note("generated", 0, 8), note("voice", 0, 8, 64)])
        store.author(F(1, 3), F(4, 3), 60, 73)
        store.reject(F(2), F(3), None)
        store.add_onset(F(7, 3))
        self.project.tracks.append(Track(
            "bass", "bass", str(self.media), 4.5, 48000, "mono", notes=store,
            settings={"threshold": 0.25, "enabled": True},
            training=[
                {"kind": "onset", "time": 0.5, "label": True},
                {"kind": "drum", "pitch": 36, "features": [0.1, 0.2], "label": False},
            ], revision=7,
        ))
        self.project.mix_path = str(self.external)
        self.project.key = "D minor"
        from audio_to_midi.domain import ChordSegment
        self.project.chords = [ChordSegment(F(0), F(4), "Dm/F", "authored", 0.5, ("F",))]
        self.project.stages = {"inference": {"state": "processed-empty", "intervals": [[0, 1]]}}
        self.project.revision = 9

    def tearDown(self):
        shutil.rmtree(self.fixture)

    def manifest(self):
        return json.loads((self.folder / "project.json").read_text(encoding="utf-8"))

    def write_manifest(self, data):
        (self.folder / "project.json").write_text(json.dumps(data), encoding="utf-8")

    def test_exact_round_trip_including_training_feedback_and_revisions(self):
        save_project(self.project, self.folder)
        loaded = load_project(self.folder)
        self.assertEqual(loaded, self.project)
        self.assertEqual(loaded.track("bass").notes.resolved(),
                         self.project.track("bass").notes.resolved())
        self.assertFalse(loaded.track("bass").notes.undo())
        self.assertEqual(loaded.track("bass").notes.generation,
                         self.project.track("bass").notes.generation)

    def test_schema_exact_fractions_and_relative_managed_paths(self):
        save_project(self.project, self.folder)
        data = self.manifest()
        self.assertEqual(data["schema_version"], SCHEMA_VERSION)
        self.assertEqual(data["grid"]["subdivision"], {"numerator": 1, "denominator": 3})
        self.assertEqual(data["tracks"][0]["notes"]["authored"][0]["start"],
                         {"numerator": 1, "denominator": 3})
        self.assertEqual(data["tracks"][0]["audio_path"], "stem.wav")
        self.assertEqual(data["mix_path"], str(self.external))

    def test_relative_input_path_is_project_relative(self):
        self.project.tracks[0].audio_path = "stem.wav"
        save_project(self.project, self.folder)
        self.assertEqual(load_project(self.folder).tracks[0].audio_path, str(self.media))

    def test_empty_audio_and_missing_mix_are_allowed(self):
        self.project.tracks[0].audio_path = ""
        self.project.mix_path = None
        save_project(self.project, self.folder)
        loaded = load_project(self.folder)
        self.assertEqual(loaded.tracks[0].audio_path, "")
        self.assertIsNone(loaded.mix_path)

    def test_missing_stem_and_mix_are_explicit(self):
        save_project(self.project, self.folder)
        self.media.unlink()
        with self.assertRaises(MissingMediaError) as caught:
            load_project(self.folder)
        self.assertEqual(caught.exception.path, self.media)
        self.media.write_bytes(b"restored")
        self.external.unlink()
        with self.assertRaises(MissingMediaError):
            load_project(self.folder)

    def test_missing_manifest_is_explicit(self):
        with self.assertRaises(MissingProjectError):
            load_project(self.folder)

    def test_unsupported_or_missing_schema(self):
        save_project(self.project, self.folder)
        original = self.manifest()
        for version in (999, "2", True, None):
            data = {**original, "schema_version": version}
            self.write_manifest(data)
            with self.subTest(version=version), self.assertRaises(SchemaVersionError):
                load_project(self.folder)

    def test_corrupt_manifest_and_duplicate_keys(self):
        for text in ("{broken", "[]", '{"schema_version":2,"schema_version":2}',
                     '{"schema_version":2,"bpm":NaN}'):
            (self.folder / "project.json").write_text(text, encoding="utf-8")
            with self.subTest(text=text), self.assertRaises(InvalidProjectError):
                load_project(self.folder)

    def test_invalid_fields_never_silently_normalize(self):
        save_project(self.project, self.folder)
        original = self.manifest()
        mutations = [
            lambda d: d["grid"].update(bpm=0),
            lambda d: d["grid"].update(subdivision={"numerator": 1, "denominator": 0}),
            lambda d: d["grid"].update(subdivision={"numerator": 1.0, "denominator": 3}),
            lambda d: d["tracks"][0].update(role="alien"),
            lambda d: d["tracks"][0].update(model="unknown"),
            lambda d: d["tracks"][0].update(duration=-1),
            lambda d: d["tracks"][0].update(duration=float("inf")),
            lambda d: d["tracks"][0].update(training=[1]),
            lambda d: d["tracks"][0]["notes"]["generated"][0].update(pitch=128),
            lambda d: d["tracks"][0]["notes"].update(revision=True),
            lambda d: d["tracks"][0]["notes"].update(onsets=[0.5]),
            lambda d: d["tracks"].append(d["tracks"][0]),
            lambda d: d.pop("chords"),
            lambda d: d.update(unrecognized="would be lost"),
        ]
        for index, mutate in enumerate(mutations):
            data = json.loads(json.dumps(original))
            mutate(data)
            self.write_manifest(data)
            with self.subTest(index=index), self.assertRaises(InvalidProjectError):
                load_project(self.folder)

    def test_failed_serialization_leaves_previous_manifest(self):
        save_project(self.project, self.folder)
        previous = (self.folder / "project.json").read_bytes()
        self.project.tracks[0].training.append({"invalid": object()})
        with self.assertRaises(ValueError):
            save_project(self.project, self.folder)
        self.assertEqual((self.folder / "project.json").read_bytes(), previous)
        self.assertEqual(load_project(self.folder).name, "Música")

    def test_failed_fsync_or_replace_preserves_valid_manifest_and_cleans_staging(self):
        save_project(self.project, self.folder)
        previous = (self.folder / "project.json").read_bytes()
        self.project.name = "Unsaved"
        for target in ("os.fsync", "os.replace"):
            with self.subTest(target=target):
                with patch(f"audio_to_midi.storage.{target}", side_effect=OSError("interrupted")):
                    with self.assertRaises(OSError):
                        save_project(self.project, self.folder)
                self.assertEqual((self.folder / "project.json").read_bytes(), previous)
                self.assertEqual(load_project(self.folder).name, "Música")
                self.assertEqual(sorted(p.name for p in self.folder.iterdir()),
                                 ["project.json", "stem.wav"])

    def test_fsync_occurs_before_replace(self):
        events = []
        from audio_to_midi import storage
        real_fsync, real_replace = storage.os.fsync, storage.os.replace

        def fsync(fd):
            events.append("fsync")
            real_fsync(fd)

        def replace(source, destination):
            events.append("replace")
            real_replace(source, destination)

        with patch.object(storage.os, "fsync", side_effect=fsync), \
                patch.object(storage.os, "replace", side_effect=replace):
            save_project(self.project, self.folder)
        self.assertEqual(events, ["fsync", "replace"])
        self.assertEqual(load_project(self.folder), self.project)


if __name__ == "__main__":
    unittest.main()
