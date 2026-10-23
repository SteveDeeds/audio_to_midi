from fractions import Fraction as F
import os
from pathlib import Path
import tempfile
import time
import unittest
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import soundfile as sf
from PySide6.QtWidgets import QApplication, QMessageBox

from audio_to_midi.adapters import Prediction
from audio_to_midi.application import add_stem, publish, token
from audio_to_midi.domain import Note
from audio_to_midi.ui import ProjectWindow


class IsolatedEditorSuite(unittest.TestCase):
    def test_editor_in_separate_qapplication(self):
        # Legacy coordinator tests own a QCoreApplication in the parent process.
        subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "unittest", "discover",
             "-s", "tests", "-p", "test_v2_ui.py"],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"}, check=True,
        )


class EditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.wav = self.root / "bass.wav"
        sf.write(self.wav, np.zeros(16000), 8000)
        self.window = ProjectWindow()
        self.window.auto.setChecked(False)
        self.track = add_stem(self.window.project, self.wav, "bass")
        self.window.project.key = "C"
        self.window._sync_tracks()
        self.addCleanup(self.dispose)

    def dispose(self):
        self.window._cancel()
        if self.window.job:
            self.window.job.wait(10000)
            self.app.processEvents()
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def wait_idle(self):
        timeout = time.monotonic() + 10
        while self.window.job and time.monotonic() < timeout:
            self.app.processEvents()
            time.sleep(.005)
        self.assertIsNone(self.window.job)

    def test_edit_undo_redo_rejections_and_training(self):
        self.window._add_note(F(0), 60)
        self.assertEqual(len(self.window._rows), 1)
        self.assertTrue(self.track.training)
        self.window._undo()
        self.assertEqual(len(self.window._rows), 0)
        self.assertEqual(self.track.training, [])
        self.window._redo()
        self.assertEqual(len(self.window._rows), 1)
        self.window._select_note(self.window._rows[0])
        self.window._rest()
        self.assertEqual(len(self.window._rows), 0)
        self.window._undo()
        self.assertEqual(len(self.window._rows), 1)

    def test_undo_does_not_restore_obsolete_generated_publication(self):
        self.window._add_note(F(0), 60)
        result = Prediction(F(0), F(4), (Note("new", F(1), F(2), 67),), ())
        publish(self.window.project, token(self.window.project, self.track), result)
        self.window._undo()
        self.assertEqual([n.id for n in self.track.notes.resolved()], ["new"])

    def test_async_publication_and_stale_edit_rejection(self):
        result = Prediction(F(0), F(4), (Note("g", F(0), F(1), 60),), (F(0),))
        with patch.object(self.window.service, "predict", return_value=result):
            self.window._analyze(False)
            self.wait_idle()
        self.assertEqual(len(self.window._rows), 1)
        old = token(self.window.project, self.track)
        self.window._add_note(F(1), 64)
        self.window._publish((old, result))
        self.assertIn("stale", self.window.status.text())
        self.assertEqual(len(self.track.notes.resolved()), 2)

    def test_worker_error_is_visible_and_window_remains_editable(self):
        with patch.object(self.window.service, "predict", side_effect=RuntimeError("missing model")):
            self.window._analyze(False)
            self.wait_idle()
        self.assertIn("missing model", self.window.status.text())
        self.window._add_note(F(0), 60)
        self.assertEqual(len(self.window._rows), 1)

    def test_window_renders_and_track_switch_does_not_move_shared_window(self):
        second = add_stem(self.window.project, self.wav, "vocals")
        self.window._sync_tracks()
        self.window._navigate(1)
        self.window.tracks.setCurrentIndex(1)
        self.assertEqual(self.window.active.id, second.id)
        self.assertEqual(self.window.timeline.start, F(8))
        self.window.show()
        self.app.processEvents()
        image = self.window.grab().toImage()
        self.assertGreater(image.width(), 500)
        self.assertFalse(image.isNull())

    def test_save_reopen_preserves_edit_state(self):
        self.window.folder = self.root / "project"
        self.window._add_note(F(0), 60)
        self.window._save()
        from audio_to_midi.storage import load_project
        loaded = load_project(self.window.folder)
        self.assertEqual(loaded.tracks[0].notes.resolved(), self.track.notes.resolved())

    def test_whole_track_publishes_multiple_batches_and_marks_complete(self):
        from audio_to_midi.application import complete

        sf.write(self.wav, np.zeros(48000), 8000)
        self.track.duration = 6
        calls = []

        def predict(project, track, a, b, cancelled):
            calls.append((a, b))
            return Prediction(a, b, (), ())

        with patch.object(self.window.service, "predict", side_effect=predict):
            self.window._analyze(True)
            timeout = time.monotonic() + 10
            while not complete(self.window.project, self.track) and time.monotonic() < timeout:
                self.app.processEvents()
                time.sleep(.005)
            self.wait_idle()
        self.assertEqual(calls, [(F(0), F(8)), (F(8), F(12))])
        self.assertTrue(complete(self.window.project, self.track))

    def _mix_only(self, seconds=20.0, rate=8000):
        """Turn the fixture into a mix-only project (detection is stubbed)."""
        mix_path = self.root / "mix.wav"
        sf.write(mix_path, np.zeros(int(seconds * rate)), rate)
        self.window.project.tracks.clear()
        self.window.folder = self.root / "project"
        self.window.folder.mkdir()
        with patch.object(self.window, "_detect"):
            self.window._set_mix(mix_path)
        return mix_path

    def test_menu_bar_and_gating(self):
        window = self.window
        self.assertEqual([a.text().replace("&", "") for a in window.menuBar().actions()],
                         ["File", "Edit", "View", "Model", "Help"])
        self.assertEqual([window.tracks.itemText(i) for i in range(window.tracks.count())], ["bass"])
        for name in ("detect", "warp", "separate", "import_stems", "chords", "confirm", "unwarp"):
            self.assertFalse(window.actions[name].isEnabled(), name)
        self.assertTrue(window.actions["analyze"].isEnabled())
        self.assertIn("mix", window.actions["detect"].toolTip())
        self._mix_only()
        self.assertEqual(window.view, "__mix__")
        self.assertEqual([window.tracks.itemText(i) for i in range(window.tracks.count())], ["Mix", "Chords"])
        self.assertTrue(window.actions["detect"].isEnabled())
        self.assertTrue(window.actions["warp"].isEnabled())
        self.assertFalse(window.actions["separate"].isEnabled())
        self.assertIn("Mix \u2713", window.stages.text())

    def test_mix_view_pins_count_bars_and_feed_warp_anchors(self):
        from PySide6.QtWidgets import QDialog
        from audio_to_midi.beats import Beat
        from audio_to_midi.dialogs import WarpDialog
        from audio_to_midi.timing import BeatSession

        window = self.window
        self._mix_only()
        # 120 BPM quarters plus weaker off-beat eighths (the usual tracker octave error).
        beats = [Beat(1.0 + 0.25 * i, 0.9 if i % 2 == 0 else 0.4) for i in range(72)]
        window.session = BeatSession(beats, 240.0, 20.0, 4, 4)
        window._session_changed()
        self.assertEqual({kind for _, kind, _ in window.timeline.beats}, {"uncounted"})
        self.assertAlmostEqual(window.project.grid.offset_seconds, 1.0)
        session = window.session
        at = lambda seconds: next(i for i, b in enumerate(session.beats) if abs(b.time - seconds) < 1e-9)
        window._pin(at(1.0), 1)
        window._pin(at(5.0), 3)
        self.assertAlmostEqual(window.project.grid.bpm, 120, places=3)
        self.assertAlmostEqual(window.project.grid.offset_seconds, 1.0)
        lane = {round(s, 3): (kind, label) for s, kind, label in window.timeline.beats}
        self.assertEqual(lane[1.0], ("pin", "1"))
        self.assertEqual(lane[3.0], ("downbeat", "2"))
        self.assertEqual(lane[1.5][0], "counted")
        self.assertEqual(lane[1.25][0], "uncounted")
        self.assertIn("Count bars \u2713", window.stages.text())
        with self.assertRaisesRegex(ValueError, "already pinned"):
            window._pin(at(3.0), 1)
        window._add_beat(10.1)
        self.assertEqual(len(session.beats), 73)
        self.assertAlmostEqual(window.timeline.selected_beat, 10.1)
        window._remove_selected_beat()
        self.assertEqual(len(session.beats), 72)
        window._select_beat(at(9.0))
        window._pin_selected()
        self.assertEqual(session.beats[at(9.0)].mark, (5, 1))
        window._unpin_selected()
        self.assertIsNone(session.beats[at(9.0)].mark)
        window.timeline.resize(900, 400)
        window.timeline.grab()

        captured = {}

        def fake_prepare(*args, **kwargs):
            captured.update(kwargs)
            return {"store": None, "warp_revision": {"id": "w1"}}

        def accept_stretched(dialog):
            dialog.as_is.setChecked(False)
            return QDialog.DialogCode.Accepted

        adopted = []
        with patch.object(WarpDialog, "exec", accept_stretched), \
                patch("audio_to_midi.pipeline.prepare_mix", fake_prepare), \
                patch.object(window, "_adopt", lambda store, warp_id: adopted.append(warp_id)):
            window._warp()
            self.wait_idle()
        anchors = captured["beat_anchors"]
        self.assertEqual(anchors[0], (1.0, 0.0))
        self.assertTrue(all(a[0] < b[0] and a[1] < b[1] for a, b in zip(anchors, anchors[1:])))
        self.assertAlmostEqual(captured["bpm"], 120, places=3)
        self.assertEqual(adopted, ["w1"])
        saved = window.project.stages["timing"]["beats"]
        self.assertEqual(len(saved["beats"]), 72)

    def test_warp_as_is_check_confirm_and_return_to_original(self):
        from PySide6.QtWidgets import QDialog
        from audio_to_midi.application import stale
        from audio_to_midi.dialogs import WarpDialog

        window = self.window
        mix_path = self._mix_only(seconds=4.0)
        yes = patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes)
        window._apply_timing((120.0, 4, 4, 0.0, "C"))
        with patch.object(WarpDialog, "exec", lambda dialog: QDialog.DialogCode.Accepted), \
                patch.object(window, "_detect") as detect:
            window._warp()
            self.wait_idle()
        self.assertTrue(window.warped)
        detect.assert_called_once()
        self.assertTrue(Path(window.project.mix_path).is_file())
        self.assertNotEqual(Path(window.project.mix_path), mix_path.resolve())
        self.assertFalse(window.confirmed)
        self.assertFalse(window.actions["separate"].isEnabled())
        self.assertTrue(window.actions["confirm"].isEnabled())
        window.check = [0.5, 1.0, 1.52, 1.6]
        window._refresh()
        self.assertEqual([kind for _, kind, _ in window.timeline.beats], ["check_ok", "check_ok", "check_ok", "check_bad"])
        self.assertIn("max 100 ms", window._check_text())
        window._set_confirmed(True)
        self.assertTrue(window.confirmed)
        self.assertTrue(window.actions["separate"].isEnabled())
        self.assertTrue(window.actions["import_stems"].isEnabled())
        track = add_stem(window.project, mix_path, "bass")
        window._sync_tracks()
        self.assertFalse(stale(window.project, track))
        with yes:
            window._unwarp()
        self.assertFalse(window.warped)
        self.assertEqual(Path(window.project.mix_path), mix_path.resolve())
        self.assertTrue(stale(window.project, track))
        self.assertIn("(stale)", window.tracks.itemText(window.tracks.findData(track.id)))

    def test_separation_asks_once_and_runs_in_background(self):
        window = self.window
        runs = []
        window._run = lambda *args, **kwargs: runs.append((args, kwargs))
        self.enterContext(patch.object(ProjectWindow, "confirmed", property(lambda _: True)))
        window.project.stages["preparation"] = {"directory": str(self.root / "prep"), "warp_revision": "w1"}
        with patch("audio_to_midi.provision.missing", return_value=["demucs>=4.0.1"]), \
                patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No) as ask:
            window._separate()
        self.assertIn("First use", ask.call_args.args[2])
        self.assertEqual(runs, [])
        with patch("audio_to_midi.provision.missing", return_value=[]), \
                patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes) as ask:
            window._separate()
        self.assertNotIn("First use", ask.call_args.args[2])
        self.assertEqual(len(runs), 1)
        self.assertTrue(runs[0][1]["reports"])
        self.assertEqual(runs[0][1]["kind"], "separate")

    def test_missing_model_installs_before_analysis(self):
        window = self.window
        runs = []
        window._run = lambda *args, **kwargs: runs.append((args, kwargs))
        track = SimpleNamespace(role="bass", model="crepe_tiny", settings={})
        self.enterContext(patch.object(ProjectWindow, "active", property(lambda _: track)))
        with patch("audio_to_midi.ui.stale", return_value=False), \
                patch("audio_to_midi.provision.missing", return_value=["torchcrepe>=0.0.23"]), \
                patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No) as ask:
            window._analyze(False)
            self.assertEqual(ask.call_count, 1)
            window._analyze(False, auto=True)
            self.assertEqual(ask.call_count, 1, "declined installs are not re-offered automatically")
        self.assertEqual(runs, [])
        with patch("audio_to_midi.ui.stale", return_value=False), \
                patch("audio_to_midi.provision.missing", return_value=["torchcrepe>=0.0.23"]), \
                patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            window._analyze(False)
            window._analyze(False, auto=True)
        self.assertEqual(len(runs), 1, "one install job, not one per navigation")
        self.assertEqual(runs[0][1]["kind"], "install")

    def test_timing_dialog_counts_the_meter_beat(self):
        from audio_to_midi.dialogs import TimingDialog

        dialog = TimingDialog(self.window, self.window.project.grid, self.window.project.key)
        dialog.bpm.setValue(120)
        dialog.denominator.setCurrentText("8")
        self.assertAlmostEqual(dialog.bpm.value(), 240)
        dialog.numerator.setValue(6)
        yes = patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes)
        with yes:
            self.window._apply_timing(dialog.values())
        grid = self.window.project.grid
        self.assertAlmostEqual(grid.bpm, 120)
        self.assertEqual((grid.numerator, grid.denominator), (6, 8))
        self.assertIn("240 BPM (1/8 = beat)", self.window.summary.text())
        self.assertIn("6/8", self.window.summary.text())
        dialog.bpm.setValue(90)
        with yes:
            self.window._apply_timing(dialog.values())
        self.assertAlmostEqual(self.window.project.grid.bpm, 45)
        self.assertEqual(self.window.timeline.end - self.window.timeline.start, 2 * F(3))
        dialog.close()

    def test_tap_downbeats_in_mix_view_pins_bars_and_sets_meter(self):
        from PySide6.QtWidgets import QInputDialog
        from audio_to_midi.beats import Beat
        from audio_to_midi.timing import BeatSession

        played = []

        class FakePlayer:
            def play_audio(self, stereo, rate):
                played.append(len(stereo))

            def stop(self):
                played.append("stop")

            def close_port(self):
                pass

        window = self.window
        window.preview = FakePlayer()
        self._mix_only(seconds=40.0)
        window._apply_timing((120.0, 3, 4, 0.0, ""))
        # 4/4 at 120 BPM tracked in eighths; beat 1 of every bar is strongest.
        beats = [Beat(0.1 + 0.25 * i, 0.9 if i % 8 == 0 else 0.6) for i in range(160)]
        window.session = BeatSession(beats, 240.0, 40.0, 3, 4)
        window._session_changed()
        window._tap(1.0)  # remembered only while tapping is on: cleared when it starts
        window.actions["tap"].setChecked(True)
        window._toggle_tap(True)
        self.assertTrue(window.tapping)
        self.assertEqual(window.taps, [])
        self.assertGreater(next(n for n in played if n != "stop"), 300000)
        self.assertIsNotNone(window._now())
        for bar in range(8):
            window._tap(0.1 + 2.0 * bar + 0.08)
        self.assertEqual(len(window.timeline.taps), 8)
        window.actions["tap"].setChecked(False)
        with patch.object(QInputDialog, "getInt", return_value=(1, True)):
            window._toggle_tap(False)
        self.assertFalse(window.tapping)
        self.assertIn("stop", played)
        self.assertIsNone(window.timeline.playhead)
        session = window.session
        self.assertEqual((session.numerator, session.denominator), (4, 4))
        pinned = {b.time: b.mark for b in session.beats if b.pin is not None}
        self.assertEqual(pinned[0.1], (1, 1))
        self.assertEqual(pinned[14.1], (8, 1))
        grid = window.project.grid
        self.assertAlmostEqual(grid.bpm, 120, places=3)
        self.assertAlmostEqual(grid.offset_seconds, 0.1, places=6)
        self.assertEqual((grid.numerator, grid.denominator), (4, 4))
        self.assertIn("8 bars counted", window.status.text())
        self.assertRegex(window.status.text(), r"\b(7[5-9]|8[0-5]) ms late")

    def test_chords_view_edits_with_undo(self):
        from audio_to_midi.domain import ChordSegment

        window = self.window
        self._mix_only(seconds=2.0)
        window.project.chords = [ChordSegment(F(0), F(4), "C"), ChordSegment(F(4), F(8), "G")]
        window.tracks.setCurrentIndex(window.tracks.findData("__chords__"))
        self.assertIs(window.pages.currentWidget(), window.chord_view)
        self.wait_idle()
        self.assertIsNotNone(window.chord_view.cqt)
        window.chord_view.set_symbol(0, "Am")
        self.assertEqual([c.symbol for c in window.project.chords], ["Am", "G"])
        window._undo()
        self.assertEqual([c.symbol for c in window.project.chords], ["C", "G"])
        window._redo()
        self.assertEqual(window.project.chords[0].symbol, "Am")
        self.assertTrue(window.actions["export_xml"].isEnabled())
        window.resize(1200, 800)
        window.show()
        self.app.processEvents()
        self.assertFalse(window.grab().toImage().isNull())

    def test_chord_detection_rejects_results_after_newer_edits(self):
        from audio_to_midi.domain import ChordSegment

        window = self.window
        window.project.stages["preparation"] = {"warp_revision": "warp"}
        window.timing["confirmed"] = True
        window.project.chords = [ChordSegment(F(0), F(4), "C")]
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            with patch.object(window, "_run") as run:
                window._detect_chords()
        callback = run.call_args.args[1]
        with patch("audio_to_midi.ui.mark_project_edited"):
            window._set_chords([ChordSegment(F(0), F(4), "Am", source="authored")])
        callback(("G", [ChordSegment(F(0), F(4), "G")]))
        self.assertEqual([c.symbol for c in window.project.chords], ["Am"])
        self.assertIn("changed", window.status.text())

    def test_clear_track_is_undoable(self):
        self.window._add_note(F(0), 60)
        self.window._onset(F(1), True)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            self.window._clear_track()
        self.assertEqual(self.track.notes.resolved(), [])
        self.assertEqual(self.track.notes.onsets, [])
        self.window._undo()
        self.assertEqual(len(self.track.notes.resolved()), 1)

    def test_right_click_menu_rests_a_beat(self):
        from PySide6.QtWidgets import QMenu

        self.window._add_note(F(0), 60, F(4))
        chosen = []

        def pick_first(menu, at):
            self.assertIsInstance(menu, QMenu)
            action = menu.actions()[0]
            chosen.append(action.text())
            action.trigger()

        with patch.object(self.window, "_popup", pick_first):
            self.window._roll_menu(F(5, 4), None)
        self.assertEqual(chosen, ["Rest for this beat"])
        self.assertEqual([(n.start, n.end) for n in self.track.notes.resolved()], [(F(0), F(1)), (F(2), F(4))])

    def test_release_edit_preserves_later_same_pitch_note(self):
        self.window._add_note(F(0), 60)
        self.window._add_note(F(2), 60)
        first = self.window._rows[0]
        self.window._edit_note(first, 1, "1", "4", 60, first.velocity)
        self.assertEqual([(n.start, n.end) for n in self.track.notes.resolved()], [
            (F(0), F(2)), (F(2), F(3)),
        ])

    def test_piano_roll_mouse_editing_and_cqt_background(self):
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest

        window, timeline = self.window, self.window.timeline
        window.resize(1280, 900)
        window.show()
        self.app.processEvents()
        self.assertEqual((timeline.low, timeline.high), (28, 67))
        self.assertIsNotNone(timeline.cqt)
        self.assertEqual(timeline.cqt.height(), 88)

        def point(beat, pitch):
            return QPoint(round(timeline._x(F(beat))), round(timeline._y(pitch) + timeline._row_height() / 2))

        left, right = Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton
        none = Qt.KeyboardModifier.NoModifier
        QTest.mouseClick(timeline, left, none, point(F(9, 8), 48))
        self.assertEqual([(n.start, n.end, n.pitch) for n in self.track.notes.resolved()], [(F(1), F(2), 48)])
        self.assertEqual(timeline.selected.pitch, 48)

        QTest.mousePress(timeline, left, none, point(F(1, 8), 50))
        QTest.mouseMove(timeline, point(F(3), 50))
        QTest.mouseRelease(timeline, left, none, point(F(3), 50))
        self.assertIn((F(0), F(3), 50), [(n.start, n.end, n.pitch) for n in self.track.notes.resolved()])

        edge = point(2, 48) - QPoint(2, 0)
        QTest.mousePress(timeline, left, none, edge)
        QTest.mouseMove(timeline, point(F(7, 2), 48))
        QTest.mouseRelease(timeline, left, none, point(F(7, 2), 48))
        self.assertIn((F(1), F(7, 2), 48), [(n.start, n.end, n.pitch) for n in self.track.notes.resolved()])

        QTest.mouseClick(timeline, right, none, point(F(3, 2), 48))
        self.assertEqual([(n.start, n.end, n.pitch) for n in self.track.notes.resolved()], [(F(0), F(3), 50)])
        window._undo()
        self.assertIn((F(1), F(7, 2), 48), [(n.start, n.end, n.pitch) for n in self.track.notes.resolved()])

        QTest.mouseClick(timeline, left, none, QPoint(round(timeline._x(F(2))), 50))
        self.assertIn(F(2), self.track.notes.onsets)

    def test_no_note_table_hover_shows_note_stats(self):
        from audio_to_midi.timeline import note_summary

        self.assertFalse(hasattr(self.window, "table"))
        self.assertIs(self.window.pages.widget(0), self.window.timeline)
        self.window._add_note(F(5, 2), 63)
        note = self.window._rows[0]
        self.assertEqual(note_summary(self.window.project, note),
                         f"D#4 (63) \u00b7 bar 1, beat 3 1/2 \u00b7 1 beat (quarter) \u00b7 velocity {note.velocity}")

    def test_info_tool_edits_and_deletes_a_clicked_note(self):
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        from audio_to_midi.dialogs import NoteInfoDialog

        window, timeline = self.window, self.window.timeline
        window.resize(1280, 900)
        window.show()
        self.app.processEvents()
        window._add_note(F(1), 48)
        window.actions["info"].trigger()
        self.assertEqual(timeline.tool, "info")
        self.assertTrue(window.actions["info"].isChecked() and not window.actions["brush"].isChecked())
        at = QPoint(round(timeline._x(F(3, 2))), round(timeline._y(48) + timeline._row_height() / 2))
        left, none = Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier

        def edit(dialog):
            self.assertEqual((dialog.bar.value(), dialog.beat.text(), dialog.length.text(), dialog.pitch.text()),
                             (1, "2", "1", "C3 (48)"))
            dialog.beat.setText("1 1/4")
            dialog.length.setText("1/2")
            dialog.pitch.setText("Eb3")
            return NoteInfoDialog.DialogCode.Accepted

        with patch.object(NoteInfoDialog, "exec", edit):
            QTest.mouseClick(timeline, left, none, at)
        self.assertEqual([(n.start, n.end, n.pitch) for n in self.track.notes.resolved()], [(F(1, 4), F(3, 4), 51)])
        QTest.mouseClick(timeline, left, none, QPoint(round(timeline._x(F(3))), at.y()))
        self.assertEqual(len(self.track.notes.resolved()), 1, "the info tool never adds notes")

        def delete(dialog):
            dialog._delete()
            return NoteInfoDialog.DialogCode.Accepted

        with patch.object(NoteInfoDialog, "exec", delete):
            window._note_info(window._rows[0])
        self.assertEqual(self.track.notes.resolved(), [])
        window.actions["brush"].trigger()
        self.assertEqual(timeline.tool, "brush")

    def test_drum_track_shows_adtof_lanes_under_half_spectrum(self):
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        from audio_to_midi.dialogs import NoteInfoDialog
        from audio_to_midi.timeline import DRUM_LANES, note_summary

        window, timeline = self.window, self.window.timeline
        drums = add_stem(window.project, self.wav, "drums")
        window._sync_tracks()
        window.tracks.setCurrentIndex(window.tracks.findData(drums.id))
        window.resize(1280, 900)
        window.show()
        self.app.processEvents()
        self.assertTrue(timeline.drums)
        self.assertEqual([name for name, _ in DRUM_LANES], ["Cymbal", "Hi-hat", "Tom", "Snare", "Kick"])
        middle = (timeline.ROLL_TOP + timeline._roll_bottom()) / 2
        self.assertAlmostEqual(timeline._rows_top(), middle)
        left, none = Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier

        def lane_y(lane):
            return round(middle + (lane + 0.5) * timeline._row_height())

        QTest.mouseClick(timeline, left, none, QPoint(round(timeline._x(F(1, 8))), lane_y(4)))
        QTest.mouseClick(timeline, left, none, QPoint(round(timeline._x(F(9, 8))), lane_y(1)))
        QTest.mouseClick(timeline, left, none, QPoint(round(timeline._x(F(9, 8))), round(middle - 20)))
        self.assertEqual(sorted(n.pitch for n in drums.notes.resolved()), [36, 42])
        drums.notes.author(F(2), F(3), 46)
        window._refresh()
        open_hat = next(n for n in window._rows if n.pitch == 46)
        self.assertTrue(note_summary(window.project, open_hat, drums=True).startswith("Hi-hat (46)"))
        self.assertIs(timeline._hit(timeline._x(F(5, 2)), lane_y(1))[0], open_hat)
        dialog = NoteInfoDialog(window, window.project, open_hat, drums=True)
        self.assertEqual(dialog.pitch_value(), 46, "an open hi-hat keeps its GM note in the Hi-hat lane")
        dialog.pitch.setCurrentIndex(0)
        self.assertEqual(dialog.pitch_value(), 49)
        dialog.close()
        self.assertFalse(window.grab().toImage().isNull())

    def test_builtin_preview_renders_notes_at_their_grid_time(self):
        from PySide6.QtCore import QSettings

        window = self.window
        window.settings = QSettings(str(self.root / "settings.ini"), QSettings.Format.IniFormat)
        window.sound_device = None
        window._add_note(F(1), 60, F(2))
        played = []
        window.preview = type("FakePreview", (), {
            "play_audio": lambda _, stereo, rate: played.append((stereo, rate)),
            "stop": lambda _: None, "close_port": lambda _: None,
        })()
        window._play()
        stereo, rate = played[0]
        self.assertEqual((rate, stereo.shape), (8000, (16000, 2)))
        left = np.abs(stereo[:, 0].astype(float))
        self.assertEqual(left[:4000].max(), 0)  # beat 1 at 120 BPM = 0.5 s
        self.assertGreater(left[4000:4100].max(), 0)
        self.assertGreater(left[4100:7900].max(), 1000)
        self.assertEqual(left[8000:].max(), 0)


def load_tests(loader, tests, pattern):
    instance = QApplication.instance()
    cls = IsolatedEditorSuite if instance is not None and not isinstance(instance, QApplication) else EditorTests
    return loader.loadTestsFromTestCase(cls)


if __name__ == "__main__":
    unittest.main()
