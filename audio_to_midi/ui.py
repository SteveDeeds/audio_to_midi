"""Main window: menu bar, Mix / track / Chords views and background jobs.

Stages: import mix -> detect beats -> count bars (tap or pin)
-> separate or import stems -> transcribe notes -> chords -> export.
Source audio is never changed; counted beats and pins form the grid's tempo map.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import threading
import time

import numpy as np
import soundfile as sf
from PySide6.QtCore import QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QCursor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QMainWindow, QMenu, QMessageBox,
    QPushButton, QSlider, QStackedWidget, QToolButton, QVBoxLayout, QWidget,
)

from audio_to_midi import beats as beat_logic
from audio_to_midi.adapters import Cancelled, MODELS
from audio_to_midi.pipeline import Cancelled as PreparationCancelled
from audio_to_midi.application import (
    CHORD_MODELS, HARMONY_ROLES, TranscriptionService, add_stem, analyze_harmony, complete, covered,
    detect_key, harmony_cqt, harmony_sources, import_profile, key_from_chroma, mark_project_edited, media_identity,
    mix_cqt, publish, record_drum, record_onset, replace_stem, stems_for_chords, token, track_end,
)
from audio_to_midi.chord_view import ChordView
from audio_to_midi.dialogs import NoteInfoDialog, PlaybackSettingsDialog, TappingCalibrationDialog, TimingDialog
from audio_to_midi.domain import Note, Project, Track
from audio_to_midi.midi import export_midi, export_musicxml, import_notes
from audio_to_midi.notation import (
    bar_and_beat, beat_length, format_number, parse_number, position,
)
from audio_to_midi.preview import (
    PLAYBACK_SPEEDS, SYNC_TEST_PITCH, balance_gains, click_track, midi_outputs, midi_volume, mix,
    render_notes, stretch_audio,
)
from audio_to_midi.spectrogram import colorize, cqt_image, song_reference
from audio_to_midi.storage import import_media, load_project, save_project
from audio_to_midi.timebase import Grid, beat_bpm
from audio_to_midi.timeline import PITCH_RANGES, Timeline
from audio_to_midi.timing import BeatSession

MIX, CHORDS = "__mix__", "__chords__"
GRIDS = (
    ("1/4", Fraction(1)), ("1/8", Fraction(1, 2)), ("1/8 triplet", Fraction(1, 3)),
    ("1/16", Fraction(1, 4)), ("1/16 triplet", Fraction(1, 6)), ("1/32", Fraction(1, 8)),
)
ZOOMS = (1, 2, 4, 8, 16)
ROLE_WORDS = {
    "drums": ("drum", "kit", "perc"), "bass": ("bass",), "vocals": ("vocal", "vox", "voice", "sing"),
    "piano": ("piano", "keys", "key"), "guitar": ("guitar", "gtr"), "other": ("other", "rest", "inst"),
}
BEAT_TOLERANCE = 0.15


def _harmony_identity(project: Project):
    """Which audio files (and versions) chord detection would read; None without any."""
    try:
        return harmony_sources(project).identity()
    except ValueError:
        return None


def guess_role(path: Path) -> str | None:
    name = path.stem.lower()
    for role, words in ROLE_WORDS.items():
        if any(word in name for word in words):
            return role
    return None


class Job(QThread):
    result = Signal(object)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, operation, parent=None, reports: bool = False):
        super().__init__(parent)
        self.operation = operation
        self.reports = reports
        self.cancelled = threading.Event()

    def run(self) -> None:
        try:
            if self.reports:
                value = self.operation(self.cancelled.is_set, self.progress.emit)
            else:
                value = self.operation(self.cancelled.is_set)
            if not self.cancelled.is_set():
                self.result.emit(value)
        except (Cancelled, PreparationCancelled):
            pass
        except Exception as exc:
            # Thread boundary: every failure is presented by the shell.
            self.error.emit(f"{type(exc).__name__}: {exc}")


class ProjectWindow(QMainWindow):
    reference_ready = Signal(object, float)
    reference_failed = Signal(object, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio to MIDI")
        self.resize(1280, 820)
        self.project = Project.create("Untitled", Grid())
        self.folder: Path | None = None
        self.service = TranscriptionService()
        self.job: Job | None = None
        self._job_status = ""
        self.pending = None
        self._installing: set[str] = set()
        self._declined_installs: set[str] = set()
        self._analysis_run = 0
        self.preview = None
        self.settings = QSettings("audio_to_midi", "v2")
        self.sound_device = self.settings.value("preview/output", "") or None
        self.latency_ms = int(self.settings.value("preview/midi_latency_ms", 0))
        self.playback_speed = 1.0
        try:
            self.balance = max(-100, min(100, int(self.settings.value("preview/balance", 0))))
        except (TypeError, ValueError):
            self.balance = 0
        self.tap_offset_seconds = (
            float(self.settings.value("timing/tap_offset_ms")) / 1000
            if self.settings.contains("timing/tap_offset_ms") else None
        )
        self._history: dict[str, list[tuple]] = {}
        self._future: dict[str, list[tuple]] = {}
        self._rows = []
        self._building = False
        self._cqt_cache: dict[tuple, tuple] = {}
        self._references: dict[tuple, float] = {}  # (path, mtime) -> whole-song CQT peak
        self._reference_jobs: set[tuple] = set()
        self.reference_ready.connect(self._reference_ready)
        self.reference_failed.connect(self._reference_failed)
        self._mix_data: dict[tuple, tuple] = {}
        self._key_dialog = None
        self._key_chroma: dict[tuple, dict] = {}
        self.session: BeatSession | None = None
        self.taps: list[float] = []
        self.bars = 2
        self._clock: tuple[float, float, float, float] | None = None  # (monotonic start, from, until, speed)
        self.actions: dict[str, QAction] = {}
        self.tool_buttons: dict[str, QToolButton] = {}
        self._build()
        self._sync_tracks()
        self._saved_project = deepcopy(self.project)

    # -- state ------------------------------------------------------------
    @property
    def view(self) -> str:
        data = self.tracks.currentData()
        return data if data in (MIX, CHORDS) else ("track" if data else "none")

    @property
    def active(self) -> Track | None:
        data = self.tracks.currentData()
        return self.project.track(data) if data and data not in (MIX, CHORDS) else None

    @property
    def timing(self) -> dict:
        return self.project.stages.setdefault("timing", {"source": self.project.mix_path, "beats": None})

    @property
    def tapping(self) -> bool:
        return self.actions["tap"].isChecked()

    # -- construction -----------------------------------------------------
    def _act(self, menu: QMenu, name: str, text: str, slot, shortcut=None, checkable=False) -> QAction:
        action = QAction(text, self)
        action.setCheckable(checkable)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        if checkable:
            action.triggered.connect(lambda checked: self._guard(lambda: slot(checked)))
        else:
            action.triggered.connect(lambda checked=False: self._guard(slot))
        menu.addAction(action)
        menu.setToolTipsVisible(True)
        self.actions[name] = action
        return action

    def _build(self) -> None:
        bar = self.menuBar()
        file = bar.addMenu("&File")
        self._act(file, "new", "&New project...", self._new, "Ctrl+N")
        self._act(file, "open", "&Open project...", self._open, "Ctrl+O")
        self._act(file, "save", "&Save", self._save, "Ctrl+S")
        file.addSeparator()
        self._act(file, "import_mix", "Import &mix...", self._import_mix)
        self._act(file, "import_stems", "Import &stems...", self._import_stems)
        self._act(file, "import_midi", "Import MIDI into track...", self._import_midi)
        self._act(file, "import_chords", "Import chord MIDI...", self._import_chords)
        file.addSeparator()
        self._act(file, "export", "&Export MIDI...", self._export, "Ctrl+E")
        self._act(file, "export_xml", "Export score (MusicXML)...", self._export_score)
        file.addSeparator()
        self._act(file, "playback", "Playback settings...", self._playback_settings)
        file.addSeparator()
        self._act(file, "exit", "E&xit", self.close)

        edit = bar.addMenu("&Edit")
        self._act(edit, "undo", "&Undo", self._undo, "Ctrl+Z")
        self._act(edit, "redo", "&Redo", self._redo, "Ctrl+Y")
        edit.addSeparator()
        self._act(edit, "detect", "&Detect tempo, meter, and key", self._detect)
        self._act(edit, "tap", "&Tap downbeats (Space)", self._toggle_tap, checkable=True)
        self._act(edit, "clear_pins", "Clear all pins", self._clear_pins)
        self._act(edit, "calibrate_tapping", "Calibrate tapping...", self._calibrate_tapping)
        self._act(edit, "set_timing", "Set tempo, meter, and &key...", self._edit_timing)
        edit.addSeparator()
        self._act(edit, "separate", "&Separate stems...", self._separate)
        self._act(edit, "chords", "Detect &chords", self._detect_chords)
        edit.addSeparator()
        self._act(edit, "clear_track", "Clear track", self._clear_track)
        self._act(edit, "clear_all", "Clear all", self._clear_all)

        view = bar.addMenu("&View")
        self._act(view, "previous", "&Previous bars", lambda: self._navigate(-1), "Left").setShortcuts(
            [QKeySequence("Left"), QKeySequence("PgUp")])
        self._act(view, "next", "&Next bars", lambda: self._navigate(1), "Right").setShortcuts(
            [QKeySequence("Right"), QKeySequence("PgDown")])
        self._act(view, "zoom_in", "Zoom &in", lambda: self._zoom(-1), "Ctrl+=")
        self._act(view, "zoom_out", "Zoom &out", lambda: self._zoom(1), "Ctrl+-")
        view.addSeparator()
        self._act(view, "play", "&Play / stop (Space)", self._play_toggle)
        self._act(view, "metronome", "&Metronome click", lambda checked: None, checkable=True)
        view.addSeparator()
        tools = QActionGroup(self)
        for name, text, key in (("brush", "&Brush tool (add, resize, delete notes)", "B"),
                                ("info", "&Info tool (click a note to edit it)", "I")):
            action = self._act(view, name, text, lambda checked, n=name: self._set_tool(n) if checked else None,
                               key, checkable=True)
            tools.addAction(action)
        self.actions["brush"].setChecked(True)

        model = bar.addMenu("&Model")
        self._act(model, "analyze", "Analyze &visible bars", lambda: self._analyze(False))
        self._act(model, "analyze_all", "Analyze &whole track", lambda: self._analyze(True))
        self.auto = self._act(model, "auto", "&Reanalyze after edits", lambda checked: self._queue_visible(),
                              checkable=True)
        self.auto.setChecked(True)
        model.addSeparator()
        self._act(model, "profile", "Import onset profile...", self._profile)
        self._act(model, "cancel", "&Cancel analysis", self._cancel, "Esc")
        model.addSeparator()
        recognizers = model.addMenu("C&hord recognizer")
        group = QActionGroup(self)
        saved = self.settings.value("chords/model", "template")
        for name, text in CHORD_MODELS.items():
            action = self._act(recognizers, f"chord_{name}", text,
                               lambda checked, n=name: self._chord_model_chosen(n) if checked else None,
                               checkable=True)
            group.addAction(action)
            action.setChecked(name == saved)
        if group.checkedAction() is None:
            self.actions["chord_template"].setChecked(True)
        advanced = model.addMenu("&Advanced")
        self._act(advanced, "key_chroma", "Show &key CQT...", self._show_key_chroma)

        helper = bar.addMenu("&Help")
        self._act(helper, "help", "&Getting started", self._help, "F1")
        self._act(helper, "about", "&About", self._about)

        body = QWidget()
        self.setCentralWidget(body)
        root = QVBoxLayout(body)
        row = QHBoxLayout()
        row.addWidget(QLabel("Track:"))
        self.tracks = QComboBox()
        self.tracks.setMinimumWidth(150)
        self.tracks.currentIndexChanged.connect(self._track_changed)
        row.addWidget(self.tracks)
        row.addWidget(QLabel("Model:"))
        self.models = QComboBox()
        self.models.setMinimumWidth(150)
        self.models.currentTextChanged.connect(self._model_changed)
        row.addWidget(self.models)
        row.addStretch(1)
        self.stages = QLabel()
        self.stages.setToolTip("Workflow: import the mix, count bars, then stems.")
        row.addWidget(self.stages)
        root.addLayout(row)

        row = QHBoxLayout()
        self.summary = QLabel()
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(self.summary, 1)
        edit_timing = QPushButton("Edit...")
        edit_timing.setToolTip("Set tempo, meter, bar 1 and key (Edit > Set tempo, meter, and key)")
        edit_timing.clicked.connect(lambda: self._guard(self._edit_timing))
        row.addWidget(edit_timing)
        root.addLayout(row)

        row = QHBoxLayout()
        for text, step in (("Previous", -1), ("Next", 1)):
            button = QPushButton(text)
            button.setToolTip(f"Scroll half a screen {'back' if step < 0 else 'forward'} "
                              f"({'Left / PgUp' if step < 0 else 'Right / PgDown'})")
            button.clicked.connect(lambda checked=False, s=step: self._guard(lambda: self._navigate(s)))
            row.addWidget(button)
            if step < 0:
                self.window_label = QLabel()
                self.window_label.setMinimumWidth(90)
                self.window_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                row.addWidget(self.window_label)
        for name, text in (("zoom_out", "Zoom out (-)"), ("zoom_in", "Zoom in (+)")):
            button = QToolButton()
            button.setDefaultAction(self.actions[name])
            button.setText(text)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            row.addWidget(button)
        self.zoom_label = QLabel()
        self.zoom_label.setToolTip("Bars shown per screen (Ctrl+wheel also zooms)")
        row.addWidget(self.zoom_label)
        row.addSpacing(16)
        row.addWidget(QLabel("Grid:"))
        self.grid_choice = QComboBox()
        for text, value in GRIDS:
            self.grid_choice.addItem(text, value)
        self.grid_choice.currentIndexChanged.connect(lambda: self._guard(self._grid_changed))
        row.addWidget(self.grid_choice)
        row.addSpacing(16)
        for name, text in (("brush", "Brush (B)"), ("info", "Info (I)")):
            button = QToolButton()
            button.setDefaultAction(self.actions[name])
            button.setText(text)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            self.tool_buttons[name] = button
            row.addWidget(button)
        row.addStretch(1)
        row.addWidget(QLabel("Audio"))
        self.balance_slider = QSlider(Qt.Orientation.Horizontal)
        self.balance_slider.setRange(-100, 100)
        self.balance_slider.setPageStep(25)
        self.balance_slider.setFixedWidth(120)
        self.balance_slider.setValue(self.balance)
        self.balance_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # keep arrow keys for navigation
        self.balance_slider.setToolTip(
            "Playback mix: left plays only the audio, right only the synth (notes and clicks), "
            "center both at full level. Double-click to center.")
        self.balance_slider.valueChanged.connect(self._balance_changed)
        self.balance_slider.mouseDoubleClickEvent = lambda event: self.balance_slider.setValue(0)
        row.addWidget(self.balance_slider)
        row.addWidget(QLabel("Synth"))
        row.addSpacing(8)
        row.addWidget(QLabel("Speed"))
        self.speed_choice = QComboBox()
        for speed in PLAYBACK_SPEEDS:
            self.speed_choice.addItem(f"{speed:.0%}", speed)
        self.speed_choice.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.speed_choice.setToolTip(
            "Playback speed without changing pitch. Source audio and exports are unchanged. "
            "Downbeat tapping always plays at 100%.")
        self.speed_choice.currentIndexChanged.connect(lambda: self._guard(self._speed_changed))
        row.addWidget(self.speed_choice)
        for text, slot in (("Play", self._play), ("Stop", self._stop)):
            button = QPushButton(text)
            button.clicked.connect(lambda checked=False, f=slot: self._guard(f))
            row.addWidget(button)
        root.addLayout(row)

        self.timeline = Timeline()
        self.timeline.project = self.project
        self.timeline.onset_requested.connect(lambda b, keep: self._guard(lambda: self._onset(b, keep)))
        self.timeline.note_requested.connect(
            lambda b, pitch, end: self._guard(lambda: self._add_note(b, pitch, end))
        )
        self.timeline.note_removed.connect(lambda note: self._guard(lambda: self._remove_note(note)))
        self.timeline.note_resized.connect(lambda note, end: self._guard(lambda: self._resize_note(note, end)))
        self.timeline.note_selected.connect(self._select_note)
        self.timeline.note_info_requested.connect(lambda note, at: self._guard(lambda: self._note_info(note)))
        self.timeline.beat_clicked.connect(lambda s, right, at: self._guard(lambda: self._beat_clicked(s, right, at)))
        self.timeline.context_requested.connect(lambda b, p, at: self._guard(lambda: self._roll_menu(b, at)))
        self.timeline.zoom_requested.connect(lambda steps: self._guard(lambda: self._zoom(-steps)))
        self.chord_view = ChordView()
        self.chord_view.changed.connect(lambda segments: self._guard(lambda: self._set_chords(segments)))
        self.pages = QStackedWidget()
        self.pages.addWidget(self.timeline)
        self.pages.addWidget(self.chord_view)
        root.addWidget(self.pages, 1)

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.statusBar().addWidget(self.status, 1)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._cancel)
        self.cancel_button.hide()
        self.statusBar().addPermanentWidget(self.cancel_button)

        self.shortcuts: dict[str, QShortcut] = {}
        for key, slot in (
            ("Space", self._space), ("Delete", self._delete_key), ("D", self._pin_selected),
            ("U", self._unpin_selected),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            if key == "Space":
                shortcut.setAutoRepeat(False)
            shortcut.activated.connect(lambda f=slot: self._guard(f))
            self.shortcuts[key] = shortcut
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(150)
        self.debounce.timeout.connect(lambda: self._guard(lambda: self._analyze(False, auto=True)))
        self.clock = QTimer(self)
        self.clock.setInterval(30)
        self.clock.timeout.connect(self._tick)
        self.rebalance = QTimer(self)
        self.rebalance.setSingleShot(True)
        self.rebalance.setInterval(200)
        self.rebalance.timeout.connect(lambda: self._guard(self._replay_balanced))

    def _guard(self, action) -> None:
        try:
            action()
        except Exception as exc:
            # Qt callback boundary: include backend/device exceptions in the UI.
            self.status.setText(f"Error: {exc}")
            QMessageBox.warning(self, "Audio to MIDI", str(exc))

    # -- refresh ----------------------------------------------------------
    def _refresh(self) -> None:
        grid = self.project.grid
        view, track = self.view, self.active
        self.timeline.project = self.chord_view.project = self.project
        self.timeline.track = track
        self.timeline.mode = "mix" if view == MIX else "notes"
        self._rows = [
            n for n in track.notes.resolved()
            if n.end > self.timeline.start and n.start < self.timeline.end
        ] if track else []
        self.timeline.notes = self._rows
        if self.timeline.selected not in self._rows:
            self.timeline.selected = None
        self.timeline.beats = self._lane_beats() if view == MIX else []
        self.timeline.taps = list(self.taps) if view == MIX else []
        self.timeline.update()
        self.chord_view.start, self.chord_view.end = self.timeline.start, self.timeline.end
        self.chord_view.update()
        first = bar_and_beat(grid, self.timeline.start)[0]
        last = bar_and_beat(grid, max(self.timeline.start, self.timeline.end - grid.subdivision))[0]
        self.window_label.setText(f"Bar {first}" if first == last else f"Bars {first}\u2013{last}")
        self.zoom_label.setText(f"{self.bars} bar{'s' if self.bars != 1 else ''}/screen")
        self.actions["zoom_in"].setEnabled(self.bars > ZOOMS[0])
        self.actions["zoom_out"].setEnabled(self.bars < ZOOMS[-1])
        index = self.grid_choice.findData(grid.subdivision)
        self._building, building = True, self._building
        if index < 0:
            self.grid_choice.addItem(format_number(grid.subdivision), grid.subdivision)
            index = self.grid_choice.count() - 1
        self.grid_choice.setCurrentIndex(index)
        self._building = building
        self.summary.setText(self._summary_text())
        self.stages.setText(self._stage_text())
        self.setWindowTitle(f"{self.project.name} - Audio to MIDI")
        self._update_actions()

    def _summary_text(self) -> str:
        grid = self.project.grid
        if self.session is not None and self.session.fit is None:
            tempo = f"~{beat_bpm(self.session.candidate_bpm, grid.denominator):.1f} BPM (bars not counted yet)"
        else:
            tempo = f"{format_number(Fraction(beat_bpm(grid.bpm, grid.denominator)).limit_denominator(10000))} BPM"
            if len(grid.anchors) > 1:
                tempo = "avg " + tempo
            if grid.denominator != 4:
                tempo += f" (1/{grid.denominator} = beat)"
        parts = [
            tempo, f"{grid.numerator}/{grid.denominator}", f"Bar 1: {grid.offset_seconds:.3f} s",
            f"Key: {self.project.key or '-'}",
        ]
        track = self.active
        if track:
            parts.append("notes complete" if complete(self.project, track) else "notes partial / not analyzed")
        return " \u00b7 ".join(parts)

    def _stage_text(self) -> str:
        def mark(done: bool) -> str:
            return "\u2713" if done else "\u25cb"

        counted = bool(self.session and self.session.fit)
        stems = len(self.project.tracks)
        return (
            f"Mix {mark(bool(self.project.mix_path))} \u203a Count bars {mark(counted)} \u203a "
            f"Stems ({stems})"
        )

    def _update_actions(self) -> None:
        has_mix, track, busy = bool(self.project.mix_path), self.active, self.job is not None
        tapping = self.tapping
        self.speed_choice.setEnabled(not tapping)
        transcribable = track is not None and track.role != "other"
        rules = {
            "import_stems": (has_mix, "Import a mix first."),
            "import_midi": (track is not None, "Choose a track first."),
            "import_chords": (has_mix, "Import a mix first."),
            "export": (bool(self.project.tracks or self.project.chords), "Nothing to export yet."),
            "export_xml": (
                bool(self.project.chords) or any(
                    t.role != "other" and (t.notes.generated or t.notes.authored) for t in self.project.tracks
                ),
                "Transcribe a track or detect chords first.",
            ),
            "detect": (has_mix and not tapping, "Import a mix first (File > Import mix)."),
            "tap": (tapping or self.session is not None, "Detect tempo first."),
            "clear_pins": (bool(self.session and self.session.pins), "No downbeat pins to clear."),
            "calibrate_tapping": (not tapping and not busy, "Finish tapping or cancel the current task first."),
            "separate": (has_mix, "Import a mix first."),
            "chords": (has_mix, "Import a mix first."),
            "clear_track": (track is not None, "Choose a track first."),
            "clear_all": (bool(self.project.tracks or self.project.chords), "Nothing to clear."),
            "analyze": (transcribable, "Choose a transcribable stem."),
            "analyze_all": (transcribable, "Choose a transcribable stem."),
            "profile": (track is not None, "Choose a track first."),
            "cancel": (busy, "Nothing is running."),
            "key_chroma": (has_mix, "Import a mix first."),
        }
        for name, (ok, why) in rules.items():
            action = self.actions[name]
            action.setEnabled(bool(ok))
            action.setToolTip("" if ok else why)
        for key in ("D", "U"):
            self.shortcuts[key].setEnabled(self.view == MIX)
        self.cancel_button.setVisible(busy)

    # -- beat lane ----------------------------------------------------------
    def _lane_beats(self) -> list[tuple[float, str, str]]:
        session = self.session
        if session is None:
            return []
        lane = []
        for index, beat in enumerate(session.beats):
            mark = session.mark_of(index)
            if beat.mark:
                lane.append((beat.time, "pin", f"{beat.mark[0]}" if beat.mark[1] == 1 else f"{beat.mark[0]}.{beat.mark[1]}"))
            elif mark is None:
                lane.append((beat.time, "uncounted", ""))
            elif mark[1] == 1:
                lane.append((beat.time, "downbeat", str(mark[0])))
            else:
                lane.append((beat.time, "counted", ""))
        return lane

    # -- tracks and views ---------------------------------------------------
    def _sync_tracks(self) -> None:
        previous = self.tracks.currentData()
        self._building = True
        self.tracks.clear()
        if self.project.mix_path:
            self.tracks.addItem("Mix", MIX)
        for track in self.project.tracks:
            self.tracks.addItem(track.role, track.id)
        if self.project.mix_path:
            self.tracks.addItem("Chords", CHORDS)
        index = self.tracks.findData(previous) if previous else -1
        if index < 0 and self.project.tracks:
            index = self.tracks.findData(self.project.tracks[0].id)
        self.tracks.setCurrentIndex(max(0, index))
        self._building = False
        self._track_changed()

    def _track_changed(self) -> None:
        if self._building:
            return
        self._cancel(analysis_only=True)
        self._building = True
        self.models.clear()
        track = self.active
        view = self.view
        if track and MODELS[track.role]:
            self.models.addItems(MODELS[track.role])
            self.models.setCurrentText(track.model)
        elif view == CHORDS:
            for name, text in CHORD_MODELS.items():
                self.models.addItem(text, name)
            self.models.setCurrentIndex(self.models.findData(self.chord_model))
        self.models.setEnabled(bool(track and MODELS[track.role]) or view == CHORDS)
        self.models.setToolTip("Chord recognizer used by Edit > Detect chords" if view == CHORDS else "")
        self._building = False
        self.timeline.set_role(track.role if track else (MIX if view == MIX else None))
        self.timeline.selected = None
        self.timeline.selected_beat = None
        self.pages.setCurrentIndex(1 if view == CHORDS else 0)
        for button in self.tool_buttons.values():
            button.setVisible(view == "track")
        self._load_envelope()
        if view == CHORDS:
            self._load_mix_data()
        self._refresh()
        self._queue_visible()

    def _set_tool(self, name: str) -> None:
        self.timeline.tool = name
        self.timeline.unsetCursor()
        self.timeline.update()

    def _audio_path(self) -> str | None:
        track = self.active
        if track:
            return track.audio_path
        if self.view == MIX:
            return self.project.mix_path
        return None

    def _load_envelope(self) -> None:
        self.timeline.envelope = np.zeros(0)
        self.timeline.audio_span = (0.0, 0.0)
        self.timeline.set_cqt(None)
        path = self._audio_path()
        if not path or not Path(path).exists():
            return
        grid = self.project.grid
        ident = (path, Path(path).stat().st_mtime_ns)
        reference = self._references.get(ident)
        if reference is None:
            self._request_reference(ident)
        key = (*ident, grid, self.timeline.start, self.timeline.end, reference)
        if key not in self._cqt_cache:
            # Read only the visible window; never load a whole file on the UI thread.
            with sf.SoundFile(path) as audio:
                rate = audio.samplerate
                start = max(0, round(grid.beat_to_seconds(self.timeline.start) * rate))
                end = min(len(audio), round(grid.beat_to_seconds(self.timeline.end) * rate))
                audio.seek(min(start, len(audio)))
                segment = audio.read(max(0, end - start), dtype="float32", always_2d=True)
            envelope = np.zeros(0)
            rgb = None
            if len(segment):
                stride = max(1, len(segment) // 800)
                envelope = np.array([
                    np.max(np.abs(segment[i:i + stride])) for i in range(0, len(segment), stride)
                ])
                # Until the whole-song level is known, show no spectrum rather than a self-normalized one.
                if reference is not None:
                    rgb = colorize(cqt_image(segment, rate, reference))
            if len(self._cqt_cache) >= 24:
                self._cqt_cache.pop(next(iter(self._cqt_cache)))
            self._cqt_cache[key] = envelope, (start / rate, (start + len(segment)) / rate), rgb
        envelope, span, rgb = self._cqt_cache[key]
        self.timeline.envelope, self.timeline.audio_span = envelope, span
        self.timeline.set_cqt(rgb)

    def _request_reference(self, ident: tuple) -> None:
        """Measure a file's whole-song CQT peak once, off the UI thread."""
        if ident in self._reference_jobs:
            return
        self._reference_jobs.add(ident)

        def work():
            try:
                value = song_reference(ident[0])
            except Exception as error:
                try:
                    self.reference_failed.emit(ident, f"{type(error).__name__}: {error}")
                except RuntimeError:  # window already closed
                    pass
                return
            try:
                self.reference_ready.emit(ident, value)
            except RuntimeError:  # window already closed
                pass

        threading.Thread(target=work, daemon=True, name="cqt-reference").start()

    def _reference_failed(self, ident: tuple, message: str) -> None:
        self._reference_jobs.discard(ident)
        self.status.setText(f"Spectrogram reference failed for {ident[0]}: {message}")

    def _reference_ready(self, ident: tuple, value: float) -> None:
        self._reference_jobs.discard(ident)
        self._references[ident] = value
        path = self._audio_path()
        if path and Path(path).exists() and (path, Path(path).stat().st_mtime_ns) == ident:
            self._load_envelope()
            self.timeline.update()

    def _load_mix_data(self) -> None:
        """Chord-view spectrum (the audio chord detection uses) and mix amplitude, cached per source."""
        path = self.project.mix_path
        if not path or not Path(path).exists():
            return
        key = (path, Path(path).stat().st_mtime_ns, _harmony_identity(self.project))
        if key in self._mix_data:
            self._show_mix_data(self._mix_data[key])
            return
        project = deepcopy(self.project)

        def work(cancelled):
            cqt, _, _, frame_seconds, _ = harmony_cqt(project, cancelled)
            with sf.SoundFile(path) as audio:
                rate = audio.samplerate
                data = audio.read(dtype="float32", always_2d=True).mean(axis=1)
            step = max(1, rate // 100)
            usable = len(data) // step * step
            envelope = np.abs(data[:usable]).reshape(-1, step).max(axis=1) if usable else np.zeros(0)
            peak = float(envelope.max()) if len(envelope) else 0.0
            return cqt, frame_seconds, envelope / peak if peak > 0 else envelope, step / rate

        def accept(value):
            self._mix_data.clear()
            self._mix_data[key] = value
            if self.view == CHORDS:
                self._show_mix_data(value)

        self._run(work, accept, kind="mix", message="Computing the chord view spectrum...")

    def _show_mix_data(self, value) -> None:
        cqt, frame_seconds, envelope, step = value
        view = self.chord_view
        view.cqt, view.frame_seconds, view.envelope, view.envelope_step = cqt, frame_seconds, envelope, step
        view.update()

    def _show_key_chroma(self) -> None:
        """Model > Advanced > Show key CQT: folded pitch-class energy per register for a chosen audio."""
        from audio_to_midi.key_view import KeyChromaDialog

        if not self.project.mix_path:
            raise ValueError("Import a mix first.")
        sources = [("key", "Key detection input"), ("mix", "Full mix")] + [
            (track.id, f"{track.role.title()} stem")
            for track in self.project.tracks if track.audio_path and Path(track.audio_path).exists()
        ]
        if self._key_dialog is not None:
            self._key_dialog.close()
        dialog = self._key_dialog = KeyChromaDialog(self, sources)
        dialog.source_changed.connect(lambda source: self._guard(lambda: self._load_key_chroma(source)))
        dialog.show()
        self._load_key_chroma(dialog.current())

    def _load_key_chroma(self, source: str) -> None:
        from audio_to_midi.chords import band_chroma
        from audio_to_midi.notation import spelling
        from audio_to_midi.spectrogram import LOW_PITCH

        dialog, project = self._key_dialog, deepcopy(self.project)
        track = next((t for t in project.tracks if t.id == source), None)
        path = track.audio_path if track else project.mix_path
        identity = (source, _harmony_identity(project) if source == "key" else media_identity(path))

        def show(profiles):
            if dialog is not self._key_dialog or dialog.current() != source:
                return
            keys = {name: key_from_chroma(vector) if vector.any() else "" for name, vector in profiles.items()}
            dialog.show_result(profiles, spelling(self.project.key), keys)

        if identity in self._key_chroma:
            show(self._key_chroma[identity])
            return

        def work(cancelled):
            cqt = harmony_cqt(project, cancelled)[2] if source == "key" else mix_cqt(path, cancelled)[0]
            return band_chroma(cqt, LOW_PITCH)

        def accept(profiles):
            self._key_chroma[identity] = profiles
            show(profiles)

        dialog.set_message("Computing the CQT of the whole song...")
        self._run(work, accept, kind="mix", message="Computing the key CQT...")

    def _chord_model_chosen(self, name: str) -> None:
        self.settings.setValue("chords/model", name)
        if self.view == CHORDS and self.models.currentData() != name:
            self._building = True
            self.models.setCurrentIndex(self.models.findData(name))
            self._building = False
        self._redetect_with(name)

    def _redetect_with(self, name: str) -> None:
        """Switching recognizer re-runs detection so the change is visible right away."""
        if self.project.mix_path:
            self.status.setText(f"Chord recognizer: {CHORD_MODELS[name]}.")
            QTimer.singleShot(0, self._detect_chords_safely)

    def _detect_chords_safely(self) -> None:
        try:
            self._detect_chords()
        except Exception as error:
            QMessageBox.warning(self, "Detect chords", str(error))

    def _model_changed(self, model: str) -> None:
        if not self._building and self.view == CHORDS and not self.active:
            name = self.models.currentData()
            if name in CHORD_MODELS and not self.actions[f"chord_{name}"].isChecked():
                self.actions[f"chord_{name}"].setChecked(True)
                self.settings.setValue("chords/model", name)
                self._redetect_with(name)
            return
        if not self._building and self.active and model:
            self._cancel(analysis_only=True)
            self.active.model = model
            self.active.revision += 1
            self._refresh()
            self._queue_visible()

    # -- window, zoom and grid ------------------------------------------------
    def _set_window(self, start: Fraction) -> None:
        self.timeline.start = max(Fraction(0), start)
        self.timeline.end = self.timeline.start + self.bars * self.project.grid.bar_beats

    def _navigate(self, direction: int) -> None:
        """Scroll half a screen so the previous view's edge stays in sight."""
        self._cancel(analysis_only=True)
        self._set_window(self.timeline.start + (self.timeline.end - self.timeline.start) / 2 * direction)
        self._load_envelope()
        self._refresh()
        self._queue_visible()

    def _zoom(self, steps: int) -> None:
        """Positive steps show more bars."""
        index = min(range(len(ZOOMS)), key=lambda i: abs(ZOOMS[i] - self.bars))
        self.bars = ZOOMS[max(0, min(len(ZOOMS) - 1, index + steps))]
        bar_beats = self.project.grid.bar_beats
        bar = self.timeline.start / bar_beats
        self._set_window((bar.numerator // bar.denominator) * bar_beats)
        self._cancel(analysis_only=True)
        self._load_envelope()
        self._refresh()
        self._queue_visible()

    def _grid_changed(self) -> None:
        if self._building:
            return
        value = self.grid_choice.currentData()
        grid = self.project.grid
        if value is None or value == grid.subdivision:
            return
        self.project.grid = replace(grid, subdivision=value)
        self.project.revision += 1
        self._cancel(analysis_only=True)
        self._refresh()
        self._queue_visible()

    # -- background jobs ---------------------------------------------------
    def _show_job_status(self, text: str) -> None:
        self._job_status = text
        self.status.setText(text)

    def _run(self, operation, callback, kind: str = "analysis", message: str = "", reports: bool = False) -> None:
        """``reports`` operations take (cancelled, report) and stream status lines."""
        if self.job is not None:
            # Note analysis is cheap to restart; other jobs (detection, separation) run to completion.
            if getattr(self.job, "kind", "analysis") == "analysis":
                self.job.cancelled.set()
            self.pending = operation, callback, kind, message, reports
            return
        job = Job(operation, self, reports)
        job.kind = kind
        if reports:
            job.progress.connect(lambda line, m=message: self._show_job_status(f"{m}  {line}" if m else line))
        self.job = job
        job.result.connect(lambda value: self._guard(lambda: callback(value)))
        job.error.connect(lambda text, k=kind: self.status.setText(
            f"{'Analysis' if k == 'analysis' else 'Task'} failed: {text}"
        ))
        job.finished.connect(self._job_finished)
        job.start()
        self._show_job_status(message or "Working in background; edits remain available. Cancel stops between model batches.")
        self._update_actions()

    def _job_finished(self) -> None:
        old, self.job = self.job, None
        old.deleteLater()
        if self.status.text() == self._job_status:
            self.status.clear()
        self._job_status = ""
        if self.pending:
            pending, self.pending = self.pending, None
            self._run(*pending)
        else:
            self._update_actions()

    def _cancel(self, analysis_only: bool = False) -> None:
        self._analysis_run += 1
        self.debounce.stop()
        if analysis_only:
            if self.pending and self.pending[2] == "analysis":
                self.pending = None
            if self.job and getattr(self.job, "kind", "analysis") == "analysis":
                self.job.cancelled.set()
            return
        self.pending = None
        if self.job:
            self.job.cancelled.set()

    # -- note analysis ----------------------------------------------------
    def _queue_visible(self) -> None:
        track = self.active
        if not self.auto.isChecked() or track is None or track.role == "other":
            return
        end = min(self.timeline.end, track_end(self.project, track))
        if self.timeline.start < end and not covered(self.project, track, self.timeline.start, end):
            self.debounce.start()

    def _analyze(self, whole: bool, auto: bool = False) -> None:
        if not self.active or self.active.role == "other":
            raise ValueError("Choose a transcribable stem.")
        from audio_to_midi import provision

        model = self.active.model
        terms = (
            "ADTOF weights are licensed for non-commercial use only. "
            if model == "adtof" else
            "Review the Piano-HR checkpoint's terms before relying on its output. " if model == "piano_hr" else ""
        )
        needs = provision.missing(model)
        if needs:
            if model in self._installing or (auto and model in self._declined_installs):
                return
            if QMessageBox.question(
                self, "Install model",
                f"{provision.LABELS.get(model, model)} is not installed yet. Download and install it now "
                f"({', '.join(needs)})? This happens once, needs an internet connection and can take a few minutes. "
                + terms,
            ) != QMessageBox.StandardButton.Yes:
                self._declined_installs.add(model)
                return
            self.active.settings["license_acknowledged"] = True
            self._installing.add(model)

            def installed(_value, whole=whole):
                self._installing.discard(model)
                self.status.setText(f"{provision.LABELS.get(model, model)} is ready.")
                self._analyze(whole)

            def install(cancel, report):
                try:
                    provision.ensure(model, report, cancel)
                finally:
                    self._installing.discard(model)

            self._run(install, installed, kind="install",
                      message=f"Installing {provision.LABELS.get(model, model)}:", reports=True)
            return
        if terms and not self.active.settings.get("license_acknowledged"):
            if QMessageBox.question(self, "Model terms", terms + "Continue?") != QMessageBox.StandardButton.Yes:
                return
            self.active.settings["license_acknowledged"] = True
        self._cancel(analysis_only=True)
        run_id = self._analysis_run
        track_id = self.active.id
        start = Fraction(0) if whole else self.timeline.start
        end = track_end(self.project, self.active) if whole else min(self.timeline.end, track_end(self.project, self.active))
        if end <= start:
            raise ValueError("No audio in this window.")
        self._analyze_range(track_id, start, end, run_id, whole)

    def _analyze_range(self, track_id: str, start: Fraction, end: Fraction, run_id: int, whole: bool) -> None:
        if run_id != self._analysis_run:
            return
        project = deepcopy(self.project)
        track = project.track(track_id)
        result_token = token(self.project, self.project.track(track_id))
        chunk_end = min(end, start + project.grid.bar_beats * 2) if whole else end

        def accept(value):
            if run_id != self._analysis_run:
                return
            self._publish(value)
            if chunk_end < end:
                QTimer.singleShot(0, lambda: self._analyze_range(track_id, chunk_end, end, run_id, whole))

        self._run(
            lambda cancelled: (result_token, self.service.predict(project, track, start, chunk_end, cancelled)),
            accept,
        )

    def _publish(self, value) -> None:
        result_token, prediction = value
        if publish(self.project, result_token, prediction):
            self._refresh()
            self.status.setText("Notes updated.")
        else:
            self.status.setText("Discarded stale inference; edits/project changed.")

    # -- undo -------------------------------------------------------------
    def _history_key(self) -> str:
        if self.view == CHORDS:
            return CHORDS
        if self.active is None:
            raise ValueError("Choose a track first.")
        return self.active.id

    def _state(self, key: str):
        if key == CHORDS:
            return list(self.project.chords)
        return self._edit_state(self.project.track(key))

    @staticmethod
    def _edit_state(track: Track) -> tuple:
        return deepcopy((
            track.notes.authored, track.notes.rejections, track.notes.onsets,
            track.training, track.settings.get("imported_midi"),
        ))

    def _checkpoint(self, key: str | None = None) -> None:
        key = key or self._history_key()
        mark_project_edited(self.project)
        self._history.setdefault(key, []).append(self._state(key))
        self._history[key] = self._history[key][-100:]
        self._future[key] = []
        self._cancel(analysis_only=True)

    def _restore(self, source, target) -> None:
        key = self._history_key()
        if not source.get(key):
            return
        self._cancel(analysis_only=True)
        target.setdefault(key, []).append(self._state(key))
        state = source[key].pop()
        if key == CHORDS:
            self.project.chords = state
            self.project.revision += 1
            self._refresh()
            return
        track = self.project.track(key)
        authored, rejections, onsets, training, imported = state
        track.notes.authored = authored
        track.notes.rejections = rejections
        track.notes.onsets = onsets
        track.training = training
        if imported is None:
            track.settings.pop("imported_midi", None)
        else:
            track.settings["imported_midi"] = imported
        track.notes.revision += 1
        self._edited(track)

    def _undo(self) -> None:
        self._restore(self._history, self._future)

    def _redo(self) -> None:
        self._restore(self._future, self._history)

    def _edited(self, track: Track | None = None) -> None:
        (track or self.active).revision += 1
        self.project.revision += 1
        self._refresh()
        self._queue_visible()

    # -- note editing -----------------------------------------------------
    def _require_transcribable(self) -> Track:
        if not self.active or self.active.role == "other":
            raise ValueError("Choose a transcribable stem.")
        return self.active

    def _onset(self, beat: Fraction, keep: bool) -> None:
        track = self._require_transcribable()
        if beat < self.timeline.start or beat >= min(self.timeline.end, track_end(self.project, track)):
            raise ValueError("Onset is outside the active audio interval.")
        self._checkpoint()
        if keep:
            track.notes.add_onset(beat)
        else:
            track.notes.delete_onset(beat, self.timeline.end)
        record_onset(track, beat, keep)
        self._edited()

    def _clamp_release(self, start: Fraction, end: Fraction, pitch: int, ignore=None) -> Fraction:
        """Stop at the audio end and before the next same-pitch attack."""
        later = [
            note.start for note in self.active.notes.resolved()
            if note.pitch == pitch and note.start > start and note != ignore
        ]
        return min([end, track_end(self.project, self.active)] + later)

    def _add_note(self, beat: Fraction, pitch: int, end: Fraction | None = None) -> None:
        track = self._require_transcribable()
        grid = self.project.grid
        if end is None:
            end = max(beat + grid.subdivision, grid.snap(beat + beat_length(grid)))
        stop = self._clamp_release(beat, end, pitch)
        if beat < 0 or stop <= beat:
            raise ValueError("Note is outside the audio.")
        self._checkpoint()
        track.notes.author(beat, stop, pitch)
        record_onset(track, beat, True)
        if track.role == "drums":
            record_drum(track, self.project, beat, stop, pitch, True)
        self._edited()
        self._select_at(beat, pitch)

    def _selected(self):
        if self.timeline.selected is None or self.timeline.selected not in self._rows:
            raise ValueError("Select a note first.")
        return self.timeline.selected

    def _select_note(self, note) -> None:
        self.timeline.selected = note
        self.timeline.update()

    def _select_at(self, start: Fraction, pitch: int) -> None:
        match = next((n for n in self._rows if n.start == start and n.pitch == pitch), None)
        if match is not None:
            self._select_note(match)

    def _note_info(self, note) -> None:
        """Info tool: edit or delete one note in a small dialog."""
        if not self.active or note not in self._rows:
            raise ValueError("The note changed; try again.")
        dialog = NoteInfoDialog(self, self.project, note, drums=self.active.role == "drums")
        if dialog.exec() != NoteInfoDialog.DialogCode.Accepted:
            return
        if dialog.deleted:
            self._remove_note(note)
        else:
            self._edit_note(note, *dialog.values())

    def _replace_note(self, old, start: Fraction, end: Fraction, pitch: int, velocity: int) -> None:
        end = self._clamp_release(start, end, pitch, ignore=old)
        if end <= start:
            raise ValueError("The note would have no length.")
        self._checkpoint()
        self.active.notes.reject(old.start, old.end, old.pitch)
        self.active.notes.author(start, end, pitch, velocity)
        record_onset(self.active, start, True)
        if self.active.role == "drums":
            record_drum(self.active, self.project, old.start, old.end, old.pitch, False)
            record_drum(self.active, self.project, start, end, pitch, True)
        self._edited()
        self._select_at(start, pitch)

    def _edit_note(self, old, bar: int, beat_text: str, length_text: str, pitch: int, velocity: int) -> None:
        grid = self.project.grid
        start = position(grid, int(bar), parse_number(beat_text))
        end = start + parse_number(length_text) * beat_length(grid)
        Note("validate", start, end, pitch, velocity)
        if (start, end, pitch, velocity) == (old.start, old.end, old.pitch, old.velocity):
            return
        if start != grid.snap(start) or end != grid.snap_end(end):
            raise ValueError(
                f"Start must fit the {format_number(grid.subdivision / beat_length(grid))}-beat grid; "
                f"note end must fit the {format_number(grid.subdivision / beat_length(grid) / 2)}-beat half-grid."
            )
        if end > track_end(self.project, self.active):
            raise ValueError("Release exceeds audio duration.")
        self._replace_note(old, start, end, pitch, velocity)

    def _resize_note(self, note, end: Fraction) -> None:
        if not self.active or note not in self._rows:
            raise ValueError("The note changed before the resize finished; try again.")
        self._replace_note(note, note.start, end, note.pitch, note.velocity)

    def _remove_note(self, note) -> None:
        if not self.active or note not in self._rows:
            raise ValueError("The note changed before it could be removed; try again.")
        self._checkpoint()
        self.active.notes.reject(note.start, note.end, note.pitch)
        if self.active.role == "drums":
            record_drum(self.active, self.project, note.start, note.end, note.pitch, False)
        self._edited()

    def _rest(self) -> None:
        """Rest over the selected note's span (all pitches)."""
        note = self._selected()
        self._rest_range(note.start, note.end)

    def _rest_range(self, start: Fraction, end: Fraction) -> None:
        self._require_transcribable()
        self._checkpoint()
        self.active.notes.reject(start, end, None)
        self._edited()

    def _popup(self, menu: QMenu, at) -> None:
        menu.exec(at or QCursor.pos())

    def _roll_menu(self, beat: Fraction, at) -> None:
        """Right-click on an empty part of the roll."""
        if self.view == MIX:
            self._beat_clicked(self.project.grid.beat_to_seconds(beat), True, at)
            return
        if not self.active:
            return
        grid = self.project.grid
        unit, bar = beat_length(grid), grid.bar_beats
        cell = Fraction((beat / unit).numerator // (beat / unit).denominator) * unit
        bar_start = Fraction((beat / bar).numerator // (beat / bar).denominator) * bar
        menu = QMenu(self)
        menu.addAction("Rest for this beat", lambda: self._guard(lambda: self._rest_range(cell, cell + unit)))
        menu.addAction("Rest for this bar", lambda: self._guard(lambda: self._rest_range(bar_start, bar_start + bar)))
        self._popup(menu, at)

    def _delete_key(self) -> None:
        if self.view == MIX:
            self._remove_selected_beat()
        elif self.view == "track" and self.timeline.selected is not None:
            self._remove_note(self.timeline.selected)

    def _clear(self, track: Track) -> None:
        self._checkpoint(track.id)
        track.notes.generated = []
        track.notes.authored = []
        track.notes.rejections = []
        track.notes.onsets = []
        track.notes.revision += 1
        track.training = []
        track.settings.pop("imported_midi", None)
        track.revision += 1

    def _clear_track(self) -> None:
        track = self.active
        if track is None:
            raise ValueError("Choose a track first.")
        if QMessageBox.question(
            self, "Clear track", f"Remove every note, onset and edit from the {track.role} track? Undo restores edits.",
        ) != QMessageBox.StandardButton.Yes:
            return
        self._clear(track)
        self._edited(track)
        self.status.setText(f"Cleared the {track.role} track. Analysis reruns when you view it (if enabled).")

    def _clear_all(self) -> None:
        if QMessageBox.question(
            self, "Clear all", "Remove the notes and edits of every track, and all chords?",
        ) != QMessageBox.StandardButton.Yes:
            return
        for track in self.project.tracks:
            self._clear(track)
        if self.project.chords:
            self._checkpoint(CHORDS)
            self.project.chords = []
        self.project.revision += 1
        self._refresh()
        self.status.setText("Cleared every track and the chords.")

    def _set_chords(self, segments) -> None:
        self._checkpoint(CHORDS)
        self.project.chords = list(segments)
        self.project.revision += 1
        self._refresh()

    # -- project files ----------------------------------------------------
    def _confirm_replace(self) -> bool:
        if self.session is not None or "timing" in self.project.stages:
            self._store_timing()
        return self.project == self._saved_project or QMessageBox.question(
            self, "Unsaved changes", "Unsaved changes will be lost. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        ) == QMessageBox.StandardButton.Yes

    def _reset_state(self) -> None:
        self._history.clear()
        self._future.clear()
        self.taps = []
        self.session = None
        timing = self.project.stages.get("timing") or {}
        if timing.get("beats"):
            grid = self.project.grid
            self.session = BeatSession.from_json(timing["beats"], grid.numerator, grid.denominator)
        if self._key_dialog is not None:
            self._key_dialog.close()
            self._key_dialog = None
        self._key_chroma.clear()
        self._set_window(Fraction(0))
        self._sync_tracks()
        self._saved_project = deepcopy(self.project)

    def _new(self) -> None:
        if not self._confirm_replace():
            return
        name, ok = QInputDialog.getText(self, "New project", "Project name")
        if not ok:
            return
        name = name.strip() or "Untitled"
        parent = QFileDialog.getExistingDirectory(self, "Choose where to create the project folder")
        if not parent:
            return
        safe = "".join("_" if c in '<>:"/\\|?*' else c for c in name).strip(" .") or "Untitled"
        folder = Path(parent) / safe
        if (folder / "project.json").exists():
            raise ValueError(f"{folder} already contains a project; use File > Open project instead.")
        folder.mkdir(parents=True, exist_ok=True)
        self._cancel()
        self._stop()
        self.project = Project.create(name, Grid())
        self.folder = folder
        save_project(self.project, folder)
        self._reset_state()
        self.status.setText(f"Created {folder}. Next: File > Import mix.")

    def _open(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open project folder")
        if not folder:
            return
        loaded = load_project(Path(folder))
        if not self._confirm_replace():
            return
        self._cancel()
        self._stop()
        copied = self._bring_media_in(loaded, Path(folder))
        if copied:
            save_project(loaded, Path(folder))
        self.project, self.folder = loaded, Path(folder)
        self._reset_state()
        self.status.setText(f"Opened {folder}." + (f" Copied the original mix into the project ({copied})." if copied else ""))

    @staticmethod
    def _bring_media_in(project: Project, folder: Path) -> str:
        """Copy an original mix that lives outside the project folder into its audio/ subfolder."""
        timing = project.stages.get("timing")
        sources = {p for p in (project.mix_path, timing.get("source") if timing else None) if p}
        moved = {}
        for source in sources:
            path = Path(source)
            if path.is_file() and not path.resolve().is_relative_to(folder.resolve()):
                try:
                    moved[source] = str(import_media(path, folder))
                except OSError:
                    continue
        if project.mix_path in moved:
            project.mix_path = moved[project.mix_path]
        if timing and timing.get("source") in moved:
            timing["source"] = moved[timing["source"]]
        return ", ".join(sorted(Path(p).name for p in moved.values()))

    def _store_timing(self) -> None:
        timing = self.timing
        timing["beats"] = self.session.to_json() if self.session else None

    def _save(self) -> None:
        if self.folder is None:
            folder = QFileDialog.getExistingDirectory(self, "Choose project folder")
            if not folder:
                return
            if (Path(folder) / "project.json").exists() and load_project(Path(folder)).id != self.project.id:
                if QMessageBox.question(
                    self, "Replace a different project", "This folder contains another project. Replace its manifest?",
                ) != QMessageBox.StandardButton.Yes:
                    return
            self.folder = Path(folder)
        self._bring_media_in(self.project, self.folder)
        if self.session is not None or "timing" in self.project.stages:
            self._store_timing()
        save_project(self.project, self.folder)
        self._saved_project = deepcopy(self.project)
        self.status.setText(f"Saved {self.folder}")

    def _import_mix(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import mix", filter="Audio (*.wav *.flac *.aif *.aiff *.ogg *.mp3)",
        )
        if path:
            self._set_mix(Path(path))

    def _set_mix(self, path: Path) -> None:
        path = path.resolve(strict=True)
        sf.info(path)
        if self.project.tracks and QMessageBox.question(
            self, "Replace mix",
            "A new mix resets the tempo detection; existing stems may no longer line up with it. Continue?",
        ) != QMessageBox.StandardButton.Yes:
            return
        self._cancel()
        self._stop()
        if self.folder is not None:
            path = import_media(path, self.folder)
        self.project.stages.pop("preparation", None)
        self.project.stages["timing"] = {"source": str(path), "beats": None}
        self.project.mix_path = str(path)
        self.project.revision += 1
        for track in self.project.tracks:
            track.revision += 1
        self.session, self.taps = None, []
        self._set_window(Fraction(0))
        self._sync_tracks()
        self.tracks.setCurrentIndex(self.tracks.findData(MIX))
        self._detect()

    def _import_stems(self) -> None:
        if not self.project.mix_path:
            raise ValueError("Import a mix before importing stems.")
        paths, _ = QFileDialog.getOpenFileNames(self, "Import stems aligned to the mix", filter="Audio (*.wav *.flac)")
        last = None
        for name in paths:
            path = Path(name)
            role = guess_role(path)
            if role is None:
                role, ok = QInputDialog.getItem(self, "Stem role", f"Role of {path.name}", list(MODELS), editable=False)
                if not ok:
                    continue
            existing = next((t for t in self.project.tracks if t.role == role), None)
            if existing is not None:
                if QMessageBox.question(
                    self, "Replace stem",
                    f"Replace the {role} audio with {path.name}? Your note edits keep their bar/beat positions; "
                    "detected notes are cleared and will be re-detected.",
                ) != QMessageBox.StandardButton.Yes:
                    continue
                replace_stem(self.project, existing, path)
                last = existing
            else:
                last = add_stem(self.project, path, role)
        if last is not None:
            self._sync_tracks()
            self.tracks.setCurrentIndex(self.tracks.findData(last.id))

    def _import_midi(self) -> None:
        if not self.active:
            raise ValueError("Select a track first.")
        path, _ = QFileDialog.getOpenFileName(self, "Import MIDI into track", filter="MIDI (*.mid *.midi)")
        if path:
            self._checkpoint()
            count = import_notes(Path(path), self.project, self.active)
            self._edited()
            self.status.setText(f"Imported {count} notes. MIDI tempo map was converted into fixed-grid musical positions.")

    def _import_chords(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import chord MIDI", filter="MIDI (*.mid *.midi)")
        if not path:
            return
        from core.midi_export import load_midi_notes
        from audio_to_midi.chords import notes_to_segments

        if self.project.chords and QMessageBox.question(
            self, "Replace chords", "Replace the current chords? (Undo in the Chords view restores them.)",
        ) != QMessageBox.StandardButton.Yes:
            return
        bpm = self.project.grid.bpm
        notes = [
            Note(f"chord-import:{i}", Fraction(str(a * bpm / 60)).limit_denominator(1000000),
                 Fraction(str(b * bpm / 60)).limit_denominator(1000000), pitch, velocity, "imported")
            for i, (a, b, pitch, velocity, _) in enumerate(load_midi_notes(path))
        ]
        self._set_chords(notes_to_segments(notes, self.project.grid))
        self.status.setText(f"Imported {len(self.project.chords)} chords from {Path(path).name}.")

    def _export_scope(self, title: str, kind: str) -> tuple[list[Track], bool, bool] | None:
        """Ask which tracks to export: (tracks, include chords, partial), or None if cancelled."""
        choices = ["All tracks + chords"] + [t.role for t in self.project.tracks if t.role != "other"]
        if self.project.chords:
            choices.append("chords")
        scope, accepted = QInputDialog.getItem(self, title, "Track(s)", choices, editable=False)
        if not accepted:
            return None
        selected = self.project.tracks if scope == choices[0] else [
            t for t in self.project.tracks if t.role == scope
        ]
        partial = any(not complete(self.project, t) for t in selected)
        if partial and QMessageBox.question(
            self, "Partial export",
            "Some tracks are unprocessed or stale. Export only currently resolved notes "
            f"as an explicitly partial {kind}? Choose No to analyze whole tracks first.",
        ) != QMessageBox.StandardButton.Yes:
            return None
        return selected, scope in (choices[0], "chords"), partial

    def _export(self) -> None:
        chosen = self._export_scope("Export MIDI", "MIDI")
        if chosen is None:
            return
        selected, include_chords, partial = chosen
        path, _ = QFileDialog.getSaveFileName(
            self, "Export MIDI", "partial.mid" if partial else "song.mid", "MIDI (*.mid)",
        )
        if path:
            export_midi(self.project, Path(path), selected, include_chords=include_chords)
            self.status.setText(f"{'PARTIAL' if partial else 'Complete'} MIDI exported: {path}")

    def _export_score(self) -> None:
        chosen = self._export_scope("Export score", "score")
        if chosen is None:
            return
        selected, include_chords, partial = chosen
        path, _ = QFileDialog.getSaveFileName(
            self, "Export score", f"{self.project.name}.musicxml",
            "MusicXML (*.musicxml *.xml);;Compressed MusicXML (*.mxl)",
        )
        if path:
            export_musicxml(self.project, Path(path), selected, include_chords=include_chords)
            self.status.setText(f"{'PARTIAL' if partial else 'Complete'} score exported: {path}")

    def _profile(self) -> None:
        if not self.active:
            raise ValueError("Select a track first.")
        path, _ = QFileDialog.getOpenFileName(self, "Import onset profile", filter="Profile (*.json)")
        if path:
            ignored = import_profile(self.active, Path(path))
            self._refresh()
            self._queue_visible()
            self.status.setText("Onset profile imported" + (f"; ignored: {', '.join(ignored)}" if ignored else "."))

    # -- timing -----------------------------------------------------------
    def _edit_timing(self) -> None:
        dialog = TimingDialog(self, self.project.grid, self.project.key)
        if dialog.exec() == TimingDialog.DialogCode.Accepted:
            self._apply_timing(dialog.values())

    def _apply_timing(self, values) -> None:
        bpm, numerator, denominator, offset, key = values
        old = self.project.grid
        # A typed tempo or bar 1 replaces the counted tempo map with a constant tempo.
        retimed = abs(bpm - old.bpm) > 1e-3 or abs(offset - old.offset_seconds) > 1e-3
        grid = (Grid(bpm, numerator, denominator, old.subdivision, offset) if retimed
                else replace(old, numerator=numerator, denominator=denominator))
        if grid != old and self.project.tracks and QMessageBox.question(
            self, "Change shared grid",
            "Notes remain at their musical positions; analysis becomes stale. Continue?",
        ) != QMessageBox.StandardButton.Yes:
            return
        if self.session is not None and (numerator, denominator) != (
            self.session.numerator, self.session.denominator,
        ):
            self.session.set_meter(numerator, denominator)
            if not retimed:
                grid = self.session.grid(old.subdivision) or grid
        self._cancel(analysis_only=True)
        if key != self.project.key:
            self.timing["key_source"] = "user" if key else "detected"
        self.project.grid, self.project.key = grid, key
        self.project.revision += 1
        for track in self.project.tracks:
            track.revision += 1
        if self.session is not None:
            self._store_timing()
        self._set_window(self.timeline.start)
        self._load_envelope()
        self._refresh()
        self._queue_visible()

    def _detect(self) -> None:
        path = self.project.mix_path
        if not path:
            raise ValueError("Import a mix first (File > Import mix).")

        def work(cancel):
            audio, rate = sf.read(path, dtype="float32", always_2d=True)
            detection = beat_logic.detect(audio.mean(axis=1), rate, cancel)
            try:
                key = detect_key(detection.cqt)
            except ValueError:
                key = ""
            return detection, key

        def accept(value):
            detection, key = value
            if self.project.mix_path != path:
                return
            if key and (not self.project.key or self.timing.get("key_source") != "user"):
                self.project.key = key
                self.timing["key_source"] = "detected"
            if self.session is not None and self.session.pins and QMessageBox.question(
                self, "Replace beats", "Detecting again replaces the beats and your downbeat pins. Continue?",
            ) != QMessageBox.StandardButton.Yes:
                return
            grid = self.project.grid
            self.session = BeatSession(
                detection.beats, detection.candidate_bpm, detection.duration, grid.numerator, grid.denominator,
            )
            self._session_changed()
            self.status.setText(self.session.summary())

        self._run(work, accept, kind="detect", message="Detecting beats, tempo and key (about 10 s per 5 minutes)...")

    def _session_changed(self) -> None:
        """The counted beats and pins define the project grid's tempo map."""
        session = self.session
        if session is not None:
            subdivision = self.project.grid.subdivision
            grid = session.grid(subdivision)
            if grid is None and session.beats:
                first = next((b.time for b in session.beats if b.confidence >= 0.5), session.beats[0].time)
                grid = Grid(round(session.candidate_bpm, 4), session.numerator, session.denominator, subdivision, first)
            if grid is not None and grid != self.project.grid:
                self.project.grid = grid
                self.project.revision += 1
                self._set_window(self.timeline.start)
            self._store_timing()
        self._load_envelope()
        self._refresh()

    # -- beats in the Mix view -------------------------------------------------
    def _require_session(self) -> BeatSession:
        if self.session is None:
            raise ValueError("Detect beats first (Edit > Detect tempo, meter, and key).")
        return self.session

    def _selected_index(self) -> int:
        session = self._require_session()
        if self.timeline.selected_beat is None:
            raise ValueError("Click a beat in the beat lane first.")
        index = session.nearest(self.timeline.selected_beat, 1e-3)
        if index < 0:
            raise ValueError("The selected beat no longer exists.")
        return index

    def _select_beat(self, index: int) -> None:
        self.timeline.selected_beat = self.session.beats[index].time if index >= 0 else None
        self.timeline.update()

    def _beat_clicked(self, seconds: float, right: bool, at) -> None:
        session = self._require_session()
        index = session.nearest(seconds, BEAT_TOLERANCE)
        self._select_beat(index)
        if not right:
            if index >= 0:
                mark = session.mark_of(index)
                where = f"bar {mark[0]} beat {format_number(mark[1])}" if mark else "not counted"
                self.status.setText(f"Beat at {session.beats[index].time:.3f} s ({where}). "
                                    "D: pin as downbeat, U: unpin, Delete: remove.")
            return
        menu = QMenu(self)
        if index >= 0:
            guess = session.guess_bar(index)
            menu.addAction(f"Downbeat of bar {guess}", lambda: self._guard(lambda: self._pin(index, guess)))
            menu.addAction("Downbeat of bar...", lambda: self._guard(lambda: self._pin_ask(index)))
            if session.beats[index].mark:
                menu.addAction("Unpin", lambda: self._guard(lambda: self._unpin(index)))
            menu.addAction("Remove this beat", lambda: self._guard(lambda: self._remove_beat(index)))
        else:
            menu.addAction("Add a beat here", lambda: self._guard(lambda: self._add_beat(seconds)))
        if session.pins:
            menu.addSeparator()
            menu.addAction("Clear all pins", lambda: self._guard(self._clear_pins))
        self._popup(menu, at)

    def _pin(self, index: int, bar: int, beat: int = 1) -> None:
        self._require_session().pin(index, bar, beat)
        self._select_beat(index)
        self._session_changed()
        self.status.setText(self.session.summary())

    def _pin_ask(self, index: int) -> None:
        bar, ok = QInputDialog.getInt(self, "Pin downbeat", "Bar number", self.session.guess_bar(index), -999, 9999)
        if ok:
            self._pin(index, bar)

    def _unpin(self, index: int) -> None:
        self._require_session().unpin(index)
        self._session_changed()

    def _remove_beat(self, index: int) -> None:
        self._require_session().remove_beat(index)
        self.timeline.selected_beat = None
        self._session_changed()

    def _add_beat(self, seconds: float) -> None:
        self._select_beat(self._require_session().add_beat(seconds))
        self._session_changed()

    def _clear_pins(self) -> None:
        self._require_session().clear_pins()
        self._session_changed()

    def _pin_selected(self) -> None:
        index = self._selected_index()
        self._pin(index, self.session.guess_bar(index))

    def _unpin_selected(self) -> None:
        self._unpin(self._selected_index())

    def _remove_selected_beat(self) -> None:
        self._remove_beat(self._selected_index())

    # -- tapping ------------------------------------------------------------
    def _calibrate_tapping(self) -> None:
        self._stop()
        dialog = TappingCalibrationDialog(self)
        if dialog.exec() == TappingCalibrationDialog.DialogCode.Accepted:
            self.tap_offset_seconds = dialog.offset_seconds
            if self.tap_offset_seconds is None:
                self.settings.remove("timing/tap_offset_ms")
                self.status.setText("Tapping uses automatic lag correction.")
            else:
                milliseconds = self.tap_offset_seconds * 1000
                self.settings.setValue("timing/tap_offset_ms", milliseconds)
                self.status.setText(f"Tapping calibrated: {milliseconds:+.0f} ms. Applies to new taps only.")

    def _toggle_tap(self, checked: bool) -> None:
        if not checked:
            self._stop()
            self._finish_taps()
            return
        try:
            self._require_session()
            if self.view != MIX:
                self.tracks.setCurrentIndex(self.tracks.findData(MIX))
            self.taps = []
            self._play()
        except Exception:
            self.actions["tap"].setChecked(False)
            raise
        self.status.setText(
            "Tapping: press Space on the first beat (the one) of every bar while the mix plays. "
            "Untick Edit > Tap downbeats when done."
        )
        self._update_actions()

    def _tap(self, seconds: float | None = None) -> None:
        if seconds is None:
            seconds = self._now()
            if seconds is None:
                return
        self.taps.append(float(seconds) - (self.tap_offset_seconds or 0.0))
        self.timeline.taps = list(self.taps)
        self.timeline.update()

    def _finish_taps(self) -> None:
        taps = sorted(self.taps)
        self._update_actions()
        if len(taps) < 2:
            self.taps = []
            self._refresh()
            self.status.setText("Tap at least two downbeats to count bars.")
            return
        session = self._require_session()
        lag = None if self.tap_offset_seconds is None else 0.0
        first_guess = session.tap_first_bar(
            taps[0], lag_seconds=beat_logic.TYPICAL_TAP_LAG if lag is None else lag,
        )
        first, ok = QInputDialog.getInt(
            self, "Tapped downbeats", "Bar number of your first tap", first_guess, -999, 9999,
        )
        if not ok:
            return
        session.apply_taps(taps, first, lag_seconds=lag)
        self.taps = []
        self._session_changed()
        self.status.setText(session.tap_note + (
            f" Calibrated correction: {self.tap_offset_seconds * 1000:+.0f} ms."
            if self.tap_offset_seconds is not None else ""
        ))

    def _space(self) -> None:
        if self.tapping:
            self._tap()
        else:
            self._play_toggle()

    # -- stems and chords ---------------------------------------------------
    def _separate(self, then_detect_chords: bool = False) -> None:
        if not self.project.mix_path:
            raise ValueError("Import a mix first.")
        from audio_to_midi import provision

        needs = provision.missing("demucs")
        first = (
            f"\n\nFirst use: the app installs Demucs ({', '.join(needs)}) and downloads its model (about 80 MB). "
            "This needs an internet connection." if needs else ""
        )
        if QMessageBox.question(
            self, "Separate stems",
            "Split the mix into drums, bass, vocals, piano, guitar and other with Demucs? "
            "A full song takes a few minutes on a CPU. Existing tracks keep their edits and get the new audio." + first,
        ) != QMessageBox.StandardButton.Yes:
            return
        if self.folder is None:
            QMessageBox.information(self, "Separate stems", "Choose a project folder first; the stems are stored in it.")
            self._save()
            if self.folder is None:
                return
        from audio_to_midi.pipeline import ArtifactStore, separate_mix

        store = ArtifactStore(self.folder / "preparation")
        new = bool(self.project.stages.get("preparation", {}).get("separation_revision"))
        project_id, mix_path = self.project.id, self.project.mix_path

        def accept(revision):
            if self.project.id != project_id or self.project.mix_path != mix_path:
                raise ValueError("The project or mix changed during separation; the stems stay in the preparation folder.")
            for role in ("drums", "bass", "vocals", "piano", "guitar", "other"):
                path = store.output(revision, role)
                existing = next((t for t in self.project.tracks if t.role == role), None)
                if existing is None:
                    add_stem(self.project, path, role)
                else:
                    replace_stem(self.project, existing, path)
            self.project.stages["preparation"] = {"directory": str(store.root), "separation_revision": revision["id"]}
            self._sync_tracks()
            if self.folder is not None:
                save_project(self.project, self.folder)
            self.status.setText("Six aligned stems imported. Choose a track to see its notes.")
            if self.view == CHORDS:
                self._load_mix_data()
            if then_detect_chords:
                self._detect_chords()

        self._run(
            lambda cancel, report: separate_mix(
                store, mix_path, force_new_revision=new, cancel=cancel, progress=report,
            ), accept, kind="separate", message="Separating stems with Demucs:", reports=True,
        )

    @property
    def chord_model(self) -> str:
        return "btc" if self.actions["chord_btc"].isChecked() else "template"

    def _detect_chords(self) -> None:
        if not self.project.mix_path:
            raise ValueError("Import a mix first.")
        model = self.chord_model
        if model == "btc":
            from audio_to_midi import provision

            needs = provision.missing("btc")
            if needs:
                if "btc" in self._installing:
                    return
                if QMessageBox.question(
                    self, "Install chord model",
                    f"The BTC chord recognizer is not installed yet. Download it now ({', '.join(needs)})? "
                    "This happens once and needs an internet connection. BTC is MIT-licensed "
                    "(github.com/jayg996/BTC-ISMIR19).",
                ) != QMessageBox.StandardButton.Yes:
                    return
                self._installing.add("btc")

                def install(cancel, report):
                    try:
                        provision.ensure("btc", report, cancel)
                    finally:
                        self._installing.discard("btc")

                self._run(install, lambda _value: self._detect_chords(), kind="install",
                          message="Installing the BTC chord model:", reports=True)
                return
        # BTC was trained on full mixes, so only the template recognizer needs stems to be reliable.
        if model == "template" and self.project.mix_path and not stems_for_chords(self.project):
            outdated = any(t.role in HARMONY_ROLES for t in self.project.tracks)
            answer = QMessageBox.question(
                self, "Separate stems first?",
                ("The separated stems no longer match the current timing. " if outdated else "")
                + "Chords are more reliable from separated stems: drums are ignored, the bass note comes "
                "from the bass stem, and vocal slides and melody notes do not count.\n\n"
                "Yes: separate stems now (a few minutes), then detect chords.\n"
                "No: detect chords from the full mix.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes,
            )
            if answer == QMessageBox.StandardButton.Yes:
                self._separate(then_detect_chords=True)
                return
            if answer != QMessageBox.StandardButton.No:
                return
        if any(c.source != "detected" for c in self.project.chords) and QMessageBox.question(
            self, "Replace chords", "Replace the chords, including your edits? (Undo in the Chords view restores them.)",
        ) != QMessageBox.StandardButton.Yes:
            return
        project = deepcopy(self.project)
        sources = _harmony_identity(project)
        work = project
        # A detected key came from the full mix; the stems (without bass and drums) judge it better.
        if project.key and self.timing.get("key_source") != "user" and stems_for_chords(project):
            work = deepcopy(project)
            work.key = ""

        def accept(value):
            if (
                self.project.id != project.id or self.project.chords != project.chords
                or self.project.grid != project.grid or self.project.mix_path != project.mix_path
                or self.project.key != project.key
                or _harmony_identity(self.project) != sources
            ):
                self.status.setText("Chords or timing changed during detection; discarded stale results.")
                return
            key, segments = value
            changed = ""
            if not self.project.key or work is not project:
                if self.project.key and key != self.project.key:
                    changed = f" Key changed from {self.project.key} to {key} (judged from the stems)."
                self.project.key = key
                self.timing["key_source"] = "detected"
            self._set_chords(segments)
            if self.view != CHORDS:
                self.tracks.setCurrentIndex(self.tracks.findData(CHORDS))
            self.status.setText(
                f"Detected {len(segments)} chords with {CHORD_MODELS[model]}.{changed} Click a chord to select "
                "it, type to rename, right-click for alternatives, drag a boundary to move it."
            )

        self._run(lambda cancel: analyze_harmony(work, cancel, model), accept, kind="chords",
                  message=f"Detecting chords ({CHORD_MODELS[model]})...")

    # -- playback -----------------------------------------------------------
    def _player(self):
        if self.preview is None:
            from audio_to_midi.preview import Preview

            self.preview = Preview()
        return self.preview

    def _clicks(self, a: float, b: float, length: int, rate: int, speed: float = 1.0) -> np.ndarray:
        grid = self.project.grid
        unit = beat_length(grid)
        steps = grid.seconds_to_beat(a) / unit
        beat = -((-steps.numerator) // steps.denominator) * unit
        strong, weak = [], []
        while (seconds_at := grid.beat_to_seconds(beat)) < b:
            if seconds_at >= a:
                (strong if beat % grid.bar_beats == 0 else weak).append((seconds_at - a) / speed)
            beat += unit
        return click_track(strong, length, rate, 1600.0, 0.7) + click_track(weak, length, rate, 1000.0, 0.45)

    def _balance_changed(self, value: int) -> None:
        self.balance = int(value)
        self.settings.setValue("preview/balance", self.balance)
        if self._now() is not None:
            self.rebalance.start()  # re-render the playing buffer once the slider settles

    def _replay_balanced(self) -> None:
        now = self._now()
        if now is not None:
            self._play(now)

    def _speed_changed(self) -> None:
        now = self._now()
        self.playback_speed = self.speed_choice.currentData()
        if now is not None:
            self._play(now)

    def _play(self, start_seconds: float | None = None) -> None:
        view, track = self.view, self.active
        path = self.project.mix_path if view in (MIX, CHORDS) else (track.audio_path if track else None)
        if not path:
            raise ValueError("Import a mix or choose a track.")
        grid = self.project.grid
        a = max(0.0, grid.beat_to_seconds(self.timeline.start) if start_seconds is None else start_seconds)
        with sf.SoundFile(path) as audio:
            rate = audio.samplerate
            audio.seek(min(len(audio), round(a * rate)))
            segment = audio.read(dtype="float32", always_2d=True).mean(axis=1)
        if not len(segment):
            raise ValueError("No audio in this window.")
        b = a + len(segment) / rate
        speed = 1.0 if self.tapping else self.playback_speed
        self._stop()
        segment = stretch_audio(segment, speed)
        clicks = self.actions["metronome"].isChecked()
        extra = self._clicks(a, b, len(segment), rate, speed) if clicks else np.zeros_like(segment)
        rows, drums = [], False
        if track:
            # Same beat-to-seconds mapping as the piano roll, so heard timing matches drawn timing.
            rows = [
                (max(a, grid.beat_to_seconds(n.start)), min(b, grid.beat_to_seconds(n.end)), n.pitch, n.velocity)
                for n in track.notes.resolved()
                if grid.beat_to_seconds(n.end) > a and grid.beat_to_seconds(n.start) < b
            ]
            drums = track.role == "drums"
        elif view == CHORDS:
            from audio_to_midi.chords import voicing

            for chord in self.project.chords:
                start, end = grid.beat_to_seconds(chord.start), grid.beat_to_seconds(chord.end)
                if end > a and start < b:
                    rows.extend((max(a, start), min(b, end), pitch, 70) for pitch in voicing(chord.symbol))
        device = self.sound_device if track else None
        player = self._player()
        audio_gain, synth_gain = balance_gains(self.balance)
        relative = [((start - a) / speed, (end - a) / speed, pitch, velocity)
                    for start, end, pitch, velocity in rows]
        if device is None:
            if rows:
                extra = extra + render_notes(relative, 0.0, len(segment), rate, drums)
            player.play_audio(mix(segment, extra if rows or clicks else None, audio_gain, synth_gain), rate)
            self.status.setText(
                "Playing" + (": audio mostly right, notes and clicks mostly left (sample-aligned)." if rows or clicks else ".")
                + " Space stops."
            )
        else:
            player.play_with_midi(mix(segment, extra, audio_gain, synth_gain), rate, relative, device,
                                  9 if drums else 0, self.latency_ms / 1000, midi_volume(synth_gain))
            self.status.setText(
                f"Playing with MIDI out '{device}' at {self.latency_ms} ms latency compensation "
                "(File > Playback settings)."
            )
        self.status.setText(self.status.text() + f" Speed: {speed:.0%}.")
        lead = max(0.0, self.latency_ms / 1000) if device is not None else 0.0
        self._clock = (time.monotonic() + lead, a, b, speed)
        self.clock.start()

    def _now(self) -> float | None:
        if self._clock is None:
            return None
        started, origin, until, speed = self._clock
        now = origin + max(0.0, time.monotonic() - started) * speed
        return now if now <= until else None

    def _tick(self) -> None:
        now = self._now()
        if now is None:
            self._clear_playhead()
            return
        self.timeline.playhead = self.chord_view.playhead = now
        if self.project.grid.seconds_to_beat(now) >= self.timeline.end:
            self._set_window(self.timeline.end)
            self._load_envelope()
            self._refresh()
        self.timeline.update()
        self.chord_view.update()

    def _clear_playhead(self) -> None:
        self._clock = None
        self.clock.stop()
        self.rebalance.stop()
        self.timeline.playhead = self.chord_view.playhead = None
        self.timeline.update()
        self.chord_view.update()

    def _play_toggle(self) -> None:
        if self._clock is not None:
            self._stop()
        else:
            self._play()

    def _stop(self) -> None:
        if self.preview:
            self.preview.stop()
        self._clear_playhead()

    def _playback_settings(self) -> None:
        try:
            outputs = midi_outputs()
        except Exception:
            outputs = []
        dialog = PlaybackSettingsDialog(self, outputs)
        if dialog.exec() == PlaybackSettingsDialog.DialogCode.Accepted:
            self.sound_device = dialog.sound.currentData()
            self.latency_ms = dialog.latency.value()
            self.settings.setValue("preview/output", self.sound_device or "")
            self.settings.setValue("preview/midi_latency_ms", self.latency_ms)

    def _sync_test(self, device: str | None, latency_ms: int) -> None:
        if device is None:
            raise ValueError("Built-in tones are already in sync; choose a MIDI output to test.")
        rate = 44100
        beat = 60 / self.project.grid.bpm
        times = [i * beat for i in range(16) if i * beat < 8]
        clicks = click_track(times, round((times[-1] + 0.5) * rate), rate)
        notes = [(t, t + 0.05, SYNC_TEST_PITCH, 110) for t in times]
        self._player().play_with_midi(mix(clicks, np.zeros_like(clicks)), rate, notes, device, 9, latency_ms / 1000)
        self.status.setText(
            "Sync test: audio clicks (right) and MIDI wood block (left) on each quarter note. "
            "If the wood block is late, raise the latency; if early, lower it. Repeat until they sound as one."
        )

    # -- help -----------------------------------------------------------------
    def _help(self) -> None:
        guide = Path(__file__).resolve().parents[1] / "docs" / "getting-started.md"
        if guide.exists():
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices

            QDesktopServices.openUrl(QUrl.fromLocalFile(str(guide)))
            return
        QMessageBox.information(self, "Getting started", (
            "1. File > New project, then File > Import mix.\n"
            "2. Count bars: Edit > Tap downbeats (Space on each bar's one) or right-click beats to pin downbeats.\n"
            "3. Listen with View > Metronome click; pin more downbeats where the clicks drift.\n"
            "4. Edit > Separate stems (or File > Import stems), pick a track and analyze notes.\n"
            "5. Edit > Detect chords, then File > Export MIDI or lead sheet."
        ))

    def _about(self) -> None:
        QMessageBox.about(self, "About Audio to MIDI", (
            "Audio to MIDI\n\nTranscribes a song into editable MIDI: tempo and bars from the mix, notes from "
            "separated stems, and a lead-sheet chord track.\nThird-party models (CREPE, basic-pitch, ADTOF, Demucs, "
            "...) keep their own licenses; see the README."
        ))

    def closeEvent(self, event) -> None:
        self._cancel()
        self._stop()
        if self.job and self.job.isRunning():
            self.status.setText("Waiting for the current task to stop; close again when idle.")
            event.ignore()
            return
        if not self._confirm_replace():
            event.ignore()
            return
        if self.preview:
            self.preview.close_port()
        event.accept()
