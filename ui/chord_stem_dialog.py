"""
ui/chord_stem_dialog.py
=======================
Dialog for selecting which stems to combine for chord analysis.
"""

import os
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QDialogButtonBox, QCheckBox, QLabel,
)


class ChordStemDialog(QDialog):
    """Checkbox dialog to pick stems for chord analysis."""

    def __init__(self, stem_paths: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Stems for Chord Analysis")
        self.setMinimumWidth(350)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Combine these stems for chord detection:"))

        self._checks: list[tuple[QCheckBox, str]] = []
        for path in stem_paths:
            cb = QCheckBox(os.path.basename(path))
            cb.setChecked(True)
            layout.addWidget(cb)
            self._checks.append((cb, path))

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_paths(self) -> list[str]:
        return [path for cb, path in self._checks if cb.isChecked()]
