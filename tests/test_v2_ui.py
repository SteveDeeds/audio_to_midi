from fractions import Fraction as F
import os
from pathlib import Path
import tempfile
import time
import unittest
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
        from PySide6.QtCore import QSettings

        settings = patch("audio_to_midi.ui.QSettings", lambda *_: QSettings(
            str(self.root / "settings.ini"), QSettings.Format.IniFormat))
        settings.start()
        self.addCleanup(settings.stop)
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

    def wait_cqt(self):
        """The piano-roll spectrum appears once the whole-song level is measured in the background."""
        timeout = time.monotonic() + 10
        while self.window.timeline.cqt is None and time.monotonic() < timeout:
            self.app.processEvents()
            time.sleep(.01)
        self.assertIsNotNone(self.window.timeline.cqt)

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
        with patch.object(self.window.service, "predict", return_value=result), \
                patch("audio_to_midi.provision.missing", return_value=[]):
            self.window._analyze(False)
            self.wait_idle()
        self.assertEqual(len(self.window._rows), 1)
        old = token(self.window.project, self.track)
        self.window._add_note(F(1), 64)
        self.window._publish((old, result))
        self.assertIn("stale", self.window.status.text())
        self.assertEqual(len(self.track.notes.resolved()), 2)

    def test_worker_error_is_visible_and_window_remains_editable(self):
        with patch.object(self.window.service, "predict", side_effect=RuntimeError("missing model")), \
                patch("audio_to_midi.provision.missing", return_value=[]):
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

    def test_save_existing_project_has_no_dialog_and_exit_has_no_warning(self):
        from PySide6.QtGui import QCloseEvent
        from audio_to_midi.ui import QFileDialog

        self.wait_idle()
        self.window.folder = self.root / "project"
        self.window._add_note(F(0), 60)
        with patch.object(QFileDialog, "getExistingDirectory") as folder_dialog:
            self.window.actions["save"].trigger()
        folder_dialog.assert_not_called()
        event = QCloseEvent()
        with patch.object(QMessageBox, "question") as question:
            self.window.closeEvent(event)
        question.assert_not_called()
        self.assertTrue(event.isAccepted())

    def test_saved_project_warns_only_after_new_changes_and_failed_save_stays_dirty(self):
        self.window.folder = self.root / "project"
        self.window._save()
        self.track.settings["test_setting"] = True
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No) as question:
            self.assertFalse(self.window._confirm_replace())
        question.assert_called_once()
        with patch("audio_to_midi.ui.save_project", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.window._save()
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No):
            self.assertFalse(self.window._confirm_replace())
        self.window._save()
        with patch.object(QMessageBox, "question") as question:
            self.assertTrue(self.window._confirm_replace())
        question.assert_not_called()

    def test_opened_project_is_clean_and_nested_training_changes_are_unsaved(self):
        from audio_to_midi.ui import QFileDialog

        self.window.folder = self.root / "project"
        self.window._save()
        with patch.object(QFileDialog, "getExistingDirectory", return_value=str(self.window.folder)), patch.object(
            QMessageBox, "question",
        ) as question:
            self.window._open()
            self.assertTrue(self.window._confirm_replace())
        question.assert_not_called()
        self.window.project.tracks[0].training.append({"label": 1})
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No):
            self.assertFalse(self.window._confirm_replace())

    def test_new_project_and_pristine_window_are_clean(self):
        from audio_to_midi.ui import QFileDialog, QInputDialog

        window = ProjectWindow()
        try:
            with patch.object(QMessageBox, "question") as question:
                self.assertTrue(window._confirm_replace())
            question.assert_not_called()
            with patch.object(QInputDialog, "getText", return_value=("New song", True)), patch.object(
                QFileDialog, "getExistingDirectory", return_value=str(self.root),
            ):
                window._new()
            with patch.object(QMessageBox, "question") as question:
                self.assertTrue(window._confirm_replace())
            question.assert_not_called()
            window.project.key = "B"
            with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No):
                self.assertFalse(window._confirm_replace())
        finally:
            with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
                window.close()
            window.deleteLater()

    def test_cancelled_first_save_leaves_unsaved_changes(self):
        from audio_to_midi.ui import QFileDialog

        with patch.object(QFileDialog, "getExistingDirectory", return_value=""):
            self.window._save()
        self.assertIsNone(self.window.folder)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No):
            self.assertFalse(self.window._confirm_replace())

    def test_unsaved_timing_session_changes_are_detected(self):
        from audio_to_midi.beats import Beat
        from audio_to_midi.timing import BeatSession

        self.window.folder = self.root / "project"
        self.window.session = BeatSession([Beat(i * 0.5, 0.9) for i in range(16)], 120, 8)
        self.window._save()
        self.window.session.pin(0, 1)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No):
            self.assertFalse(self.window._confirm_replace())

    def test_analysis_after_save_is_unsaved_and_declining_exit_keeps_window_open(self):
        from PySide6.QtGui import QCloseEvent

        self.wait_idle()
        self.window.folder = self.root / "project"
        self.window._save()
        result = Prediction(F(0), F(4), (Note("generated", F(0), F(1), 60),), ())
        self.window._publish((token(self.window.project, self.track), result))
        event = QCloseEvent()
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No) as question:
            self.window.closeEvent(event)
        question.assert_called_once()
        self.assertFalse(event.isAccepted())

    def test_whole_track_publishes_multiple_batches_and_marks_complete(self):
        from audio_to_midi.application import complete

        sf.write(self.wav, np.zeros(48000), 8000)
        self.track.duration = 6
        calls = []

        def predict(project, track, a, b, cancelled):
            calls.append((a, b))
            return Prediction(a, b, (), ())

        with patch.object(self.window.service, "predict", side_effect=predict), \
                patch("audio_to_midi.provision.missing", return_value=[]):
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
        copy = Path(self.window.project.mix_path)
        self.assertEqual(copy, (self.window.folder / "audio" / "mix.wav").resolve())
        return copy

    def test_external_mix_is_copied_into_the_project(self):
        from audio_to_midi.storage import load_project, save_project
        from audio_to_midi.ui import QFileDialog

        window = self.window
        copy = self._mix_only(seconds=1.0)
        self.assertEqual(window.timing["source"], str(copy))
        window._save()
        # A project saved by an older version still points at the original in Downloads.
        outside = self.root / "Downloads" / "song.wav"
        outside.parent.mkdir()
        sf.write(outside, np.zeros(800), 8000)
        old = load_project(window.folder)
        old.mix_path = str(outside)
        old.stages["timing"]["source"] = str(outside)
        save_project(old, window.folder)
        with patch.object(QFileDialog, "getExistingDirectory", return_value=str(window.folder)), \
                patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            window._open()
        inside = str((window.folder / "audio" / "song.wav").resolve())
        self.assertEqual((window.project.mix_path, window.timing["source"]), (inside, inside))
        self.assertIn("Copied the original mix", window.status.text())
        self.assertEqual(load_project(window.folder).mix_path, inside)
        outside.unlink()
        self.assertEqual(load_project(window.folder).stages["timing"]["source"], inside)

    def test_menu_bar_and_gating(self):
        window = self.window
        self.assertEqual([a.text().replace("&", "") for a in window.menuBar().actions()],
                         ["File", "Edit", "View", "Model", "Help"])
        self.assertEqual([window.tracks.itemText(i) for i in range(window.tracks.count())], ["bass"])
        for name in ("detect", "warp", "separate", "import_stems", "chords", "unwarp"):
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
        self.assertNotIn("confirm", window.actions)
        self.assertNotIn("Confirm", window.stages.text())
        self.assertTrue(window.actions["separate"].isEnabled())  # warping alone unlocks stems
        self.assertTrue(window.actions["import_stems"].isEnabled())
        self.assertTrue(window.actions["chords"].isEnabled())
        window.check = [0.5, 1.0, 1.52, 1.6]
        window._refresh()
        self.assertEqual([kind for _, kind, _ in window.timeline.beats], ["check_ok", "check_ok", "check_ok", "check_bad"])
        self.assertIn("max 100 ms", window._check_text())
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
        self.enterContext(patch.object(ProjectWindow, "warped", property(lambda _: True)))
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

    def test_chord_detection_offers_separation_without_current_stems(self):
        window = self.window
        mix_path = self._mix_only(seconds=4.0)
        self.enterContext(patch.object(ProjectWindow, "warped", property(lambda _: True)))
        runs = []
        window._run = lambda *args, **kwargs: runs.append((args, kwargs))
        buttons = QMessageBox.StandardButton
        for answer, separates, detects in ((buttons.Cancel, 0, 0), (buttons.No, 0, 1), (buttons.Yes, 1, 0)):
            with self.subTest(answer=answer), patch.object(window, "_separate") as separate, \
                    patch.object(QMessageBox, "question", return_value=answer) as ask:
                runs.clear()
                window._detect_chords()
                self.assertIn("Separate stems", ask.call_args.args[1])
                self.assertEqual(separate.call_count, separates)
                if separates:
                    separate.assert_called_with(then_detect_chords=True)
                self.assertEqual(len(runs), detects)
        add_stem(window.project, mix_path, "piano")
        with patch.object(QMessageBox, "question") as ask:
            runs.clear()
            window._detect_chords()
        ask.assert_not_called()
        self.assertEqual(len(runs), 1)
        # A key detected from the mix is re-judged from the stems; a key the user typed is kept.
        for source, expected in (("detected", "B"), ("user", "E")):
            with self.subTest(key_source=source):
                window.project.key, window.timing["key_source"] = "E", source
                runs.clear()
                window._detect_chords()
                runs[0][0][1](("B", []))
                self.assertEqual(window.project.key, expected)
                self.assertEqual("from E to B" in window.status.text(), expected == "B")

    def test_key_cqt_window_shows_band_traces_per_source(self):
        from audio_to_midi.key_view import KeyChromaDialog

        window = self.window
        mix_path = self._mix_only(seconds=4.0)
        add_stem(window.project, mix_path, "bass")
        window._update_actions()
        action = window.actions["key_chroma"]
        self.assertTrue(action.isEnabled())
        from PySide6.QtWidgets import QMenu

        menus = [m for m in action.associatedObjects() if isinstance(m, QMenu)]
        self.assertEqual([m.title() for m in menus], ["&Advanced"])
        self.assertEqual(menus[0].menuAction().associatedObjects()[0].title(), "&Model")
        runs = []
        window._run = lambda *args, **kwargs: runs.append(args)
        cqt = np.zeros((88, 20), dtype=np.float32)
        for pitch in (40, 59, 64, 68, 83):
            cqt[pitch - 21] = 1.0
        action.trigger()
        dialog = window._key_dialog
        self.assertIsInstance(dialog, KeyChromaDialog)
        self.assertEqual([dialog.source.itemText(i) for i in range(dialog.source.count())],
                         ["Key detection input", "Full mix", "Bass stem"])
        with patch("audio_to_midi.ui.harmony_cqt", return_value=(cqt, None, cqt, 0.01, 1.0)):
            work, accept = runs[-1][:2]
            accept(work(lambda: False))
        self.assertEqual(set(dialog.chart.profiles), {"Low", "Mid", "High", "All"})
        self.assertEqual(dialog.chart.keys["All"], "E")
        self.assertIn("E", dialog.message.text())
        self.assertFalse(dialog.chart.grab().isNull())
        dialog.shapes.setChecked(True)
        for name, vector in dialog.chart._shown().items():
            self.assertAlmostEqual(float(vector.sum()), 1.0, msg=name)
        self.assertFalse(dialog.chart.grab().isNull())
        dialog.source.setCurrentIndex(2)
        self.assertEqual(dialog.chart.profiles, {})
        with patch("audio_to_midi.ui.mix_cqt", return_value=(cqt, 0.01, 1.0)) as load:
            work, accept = runs[-1][:2]
            accept(work(lambda: False))
        self.assertEqual(load.call_args.args[0], window.project.tracks[0].audio_path)
        runs.clear()
        dialog.source.setCurrentIndex(0)
        self.assertEqual(runs, [], "results are cached per source")
        self.assertEqual(dialog.chart.keys["All"], "E")
        dialog.close()

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

    def test_chords_model_box_lists_chord_recognizers_in_sync_with_menu(self):
        from PySide6.QtCore import QSettings

        window = self.window
        window.settings = QSettings(str(self.root / "settings.ini"), QSettings.Format.IniFormat)
        self._mix_only(seconds=2.0)
        window.tracks.setCurrentIndex(window.tracks.findData("__chords__"))
        self.assertTrue(window.models.isEnabled())
        self.assertEqual([window.models.itemText(i) for i in range(window.models.count())],
                         ["Built-in templates", "BTC transformer (deep learning)"])
        self.assertEqual(window.models.currentData(), "template")
        window.project.stages["preparation"] = {"warp_revision": "warp"}
        with patch.object(window, "_detect_chords") as detect:
            window.models.setCurrentIndex(window.models.findData("btc"))
            self.app.processEvents()
            self.assertEqual(detect.call_count, 1)
            self.assertIn("BTC", window.status.text())
            self.assertTrue(window.actions["chord_btc"].isChecked())
            self.assertEqual(window.chord_model, "btc")
            self.assertEqual(window.settings.value("chords/model"), "btc")
            window.actions["chord_template"].trigger()
            self.app.processEvents()
            self.assertEqual(detect.call_count, 2)
        self.assertEqual(window.models.currentData(), "template")
        self.assertEqual(window.settings.value("chords/model"), "template")
        window.project.stages.pop("preparation")
        with patch.object(window, "_detect_chords") as detect:
            window.actions["chord_btc"].trigger()
            self.app.processEvents()
        detect.assert_not_called()  # unwarped: nothing to detect yet
        self.wait_idle()

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

    def test_chroma_display_suppresses_adjacent_bin_leakage(self):
        view = self.window.chord_view
        view.project = self.window.project
        view.cqt = np.zeros((88, 100), dtype=np.float32)
        view.cqt[66 - 21] = 1.0
        view.cqt[65 - 21] = 0.9
        view.cqt[67 - 21] = 0.9
        view.frame_seconds = 0.01
        self.assertIsNotNone(view._chroma_image())
        self.assertEqual(int(view._image_buffer[11 - 6, 0, 0]), 250)
        self.assertEqual(int(view._image_buffer[11 - 5, 0, 0]), 20)
        self.assertEqual(int(view._image_buffer[11 - 7, 0, 0]), 20)

    def test_chroma_display_is_scaled_to_the_whole_song(self):
        view = self.window.chord_view
        view.project = self.window.project
        view.cqt = np.zeros((88, 200), dtype=np.float32)
        view.cqt[60 - 21, :100] = 1.0  # loud C for the first half
        view.cqt[60 - 21, 100:] = 1 - 40 / 60  # same C, 40 dB quieter
        view.frame_seconds = 0.01
        view.start, view.end = F(0), F(4)  # 0-2 s at 120 BPM
        view._chroma_image()
        loud = int(view._image_buffer[11, :, 0].max())
        quiet = int(view._image_buffer[11, 110:, 0].max())
        self.assertEqual(loud, 250)
        self.assertLess(quiet, 30)

    def test_zoom_buttons_and_song_scaled_piano_roll_spectrum(self):
        window = self.window
        rate = 8000
        seconds = np.arange(rate * 4) / rate
        loud = np.sin(2 * np.pi * 220 * seconds) * .5
        mix_path = self.root / "mix.wav"
        sf.write(mix_path, np.concatenate([loud, loud * 10 ** (-40 / 20)]), rate)
        window.project.tracks.clear()
        window.folder = self.root / "project"
        window.folder.mkdir()
        with patch.object(window, "_detect"):
            window._set_mix(mix_path)
        self.wait_cqt()
        loud_red = int(window.timeline._cqt_buffer[..., 0].max())
        window._navigate(1)  # bars 3-4: the quiet half
        quiet_red = int(window.timeline._cqt_buffer[..., 0].max())
        self.assertGreater(loud_red, 200)
        self.assertLess(quiet_red, 60)

        self.assertEqual(window.zoom_label.text(), "2 bars/screen")
        window.actions["zoom_out"].trigger()
        self.assertEqual(window.bars, 4)
        self.assertEqual(window.timeline.end - window.timeline.start, 4 * window.project.grid.bar_beats)
        self.assertEqual(window.zoom_label.text(), "4 bars/screen")
        for _ in range(5):
            window.actions["zoom_in"].trigger()
        self.assertEqual(window.bars, 1)
        self.assertFalse(window.actions["zoom_in"].isEnabled())
        self.assertTrue(window.actions["zoom_out"].isEnabled())

    def test_chord_recognizer_option_downloads_btc_and_routes_detection(self):
        from PySide6.QtCore import QSettings
        from audio_to_midi import provision

        window = self.window
        window.settings = QSettings(str(self.root / "settings.ini"), QSettings.Format.IniFormat)
        window.project.stages["preparation"] = {"warp_revision": "warp"}
        self.assertEqual(window.chord_model, "template")
        with patch.object(window, "_detect_chords") as detect:
            window.actions["chord_btc"].trigger()
            self.app.processEvents()
        detect.assert_called_once()
        self.assertEqual(window.chord_model, "btc")
        self.assertEqual(window.settings.value("chords/model"), "btc")
        self.assertFalse(window.actions["chord_template"].isChecked())

        with patch.object(provision, "missing", return_value=["BTC chord weights (~12 MB)"]), \
                patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No) as asked, \
                patch.object(window, "_run") as run:
            window._detect_chords()
        self.assertIn("12 MB", asked.call_args.args[2])
        run.assert_not_called()

        with patch.object(provision, "missing", return_value=["BTC chord weights (~12 MB)"]), \
                patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes), \
                patch.object(window, "_run") as run:
            window._detect_chords()
        self.assertEqual(run.call_args.kwargs["kind"], "install")
        with patch.object(provision, "ensure") as ensure:
            run.call_args.args[0](lambda: False, lambda text: None)
        self.assertEqual(ensure.call_args.args[0], "btc")
        self.assertNotIn("btc", window._installing)

        with patch.object(provision, "missing", return_value=[]), patch.object(window, "_run") as run, \
                patch("audio_to_midi.ui.analyze_harmony", return_value=("C", [])) as analyze:
            window._detect_chords()
            run.call_args.args[0](lambda: False)
        self.assertEqual(analyze.call_args.args[2], "btc")
        self.assertIn("BTC", run.call_args.kwargs["message"])

    def test_chord_detection_rejects_results_after_newer_edits(self):
        from audio_to_midi.domain import ChordSegment

        window = self.window
        window.project.stages["preparation"] = {"warp_revision": "warp"}
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
        self.wait_cqt()
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
                         f"Eb4 (63) \u00b7 bar 1, beat 3 1/2 \u00b7 1 beat (quarter) \u00b7 velocity {note.velocity}")

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


if __name__ == "__main__":
    unittest.main()
