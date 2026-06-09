#!/usr/bin/env python
"""
app.py — Entry point for the unified Audio → MIDI application.
"""

import sys
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Audio to MIDI")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
