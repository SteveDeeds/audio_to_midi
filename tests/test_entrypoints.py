import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audio_to_midi.__main__ import main, parser


class EntryPointTests(unittest.TestCase):
    def test_script_delegates_to_package_and_preserves_exit_status(self):
        script = Path(__file__).resolve().parents[1] / "app.py"
        with patch("audio_to_midi.__main__.main", return_value=17) as launch:
            with self.assertRaises(SystemExit) as result:
                runpy.run_path(str(script), run_name="__main__")
        launch.assert_called_once_with()
        self.assertEqual(result.exception.code, 17)

    def test_help_describes_supported_app_not_preview_or_legacy(self):
        help_text = parser().format_help()
        self.assertIn("Audio to MIDI", help_text)
        self.assertNotIn("preview", help_text)
        self.assertNotIn("legacy", help_text)

    def test_no_command_opens_project_editor(self):
        with (
            patch.object(sys, "argv", ["app.py"]),
            patch("PySide6.QtWidgets.QApplication") as application,
            patch("audio_to_midi.ui.ProjectWindow") as window,
            patch("audio_to_midi.spectrogram.warm_up") as warm_up,
        ):
            application.return_value.exec.return_value = 23
            self.assertEqual(main([]), 23)
        warm_up.assert_called_once_with()
        application.return_value.setApplicationName.assert_called_once_with("Audio to MIDI")
        window.assert_called_once_with()
        window.return_value.show.assert_called_once_with()

    def test_both_launchers_show_real_project_window(self):
        code = """
import runpy
import sys
from unittest.mock import patch
from PySide6.QtCore import QSettings, QTimer
from PySide6.QtWidgets import QApplication
from audio_to_midi.ui import ProjectWindow

launcher, settings_path = sys.argv[1:]
sys.argv = [launcher]
original_exec = QApplication.exec

def inspect_window(app):
    windows = [w for w in app.topLevelWidgets() if isinstance(w, ProjectWindow)]
    assert len(windows) == 1
    assert windows[0].isVisible()
    assert app.applicationName() == "Audio to MIDI"
    assert windows[0].windowTitle().endswith("Audio to MIDI")
    QTimer.singleShot(0, app.quit)
    return original_exec()

with patch.object(QApplication, "exec", inspect_window), patch(
    "audio_to_midi.ui.QSettings",
    lambda *_: QSettings(settings_path, QSettings.Format.IniFormat),
):
    if launcher == "app.py":
        runpy.run_path(launcher, run_name="__main__")
    else:
        runpy.run_module("audio_to_midi", run_name="__main__")
"""
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            for launcher in ("app.py", "audio_to_midi"):
                with self.subTest(launcher=launcher):
                    result = subprocess.run(
                        [sys.executable, "-X", "utf8", "-c", code, launcher,
                         str(Path(directory) / "settings.ini")],
                        cwd=root, env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
                        capture_output=True, text=True, timeout=45,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
