"""
ui/envelope_view.py
===================
QGraphicsView showing the transient envelope waveform with onset markers.
Left-click to add (restore) an onset, right-click to remove one.
Stays time-synced with the CQT view above.
"""

import numpy as np

from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsLineItem,
    QGraphicsSimpleTextItem, QGraphicsPathItem,
)
from PySide6.QtCore import Qt, QRectF, QLineF, Signal
from PySide6.QtGui import (
    QPen, QColor, QBrush, QPainter, QPainterPath, QFont,
    QTransform, QWheelEvent, QKeyEvent, QMouseEvent,
)

try:
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    _USE_OPENGL = True
except ImportError:
    _USE_OPENGL = False

from core.audio_engine import AudioMarkers, TimeGrid
from ui.view_state import compute_time_window

# ── Colors ────────────────────────────────────────────────────────────────────

BG_COLOR = QColor("#0f0f1a")
ENVELOPE_COLOR = QColor("#44aacc")
ENVELOPE_FILL = QColor(68, 170, 204, 40)
BAR_LINE_COLOR = QColor("#808080")
BEAT_LINE_COLOR = QColor("#555555")
SIXTEENTH_LINE_COLOR = QColor("#2f2f2f")
ONSET_COLOR = QColor("#1a8fff")
ONSET_USER_COLOR = QColor("#ffffff")

# Label colors / fonts
BAR_LABEL_COLOR = QColor("#ddddee")
BEAT_LABEL_COLOR = QColor("#8888aa")
SIXTEENTH_LABEL_COLOR = QColor("#444466")


class EnvelopeView(QGraphicsView):
    """Transient envelope with onset markers.

    Scene coordinates:
        X = time in seconds  (same as CQT view)
        Y = amplitude 0..1   (0 at bottom, 1 at top via Y-flip)

    Signals:
        onset_added(float)   — left-click adds/restores onset at time t
        onset_removed(float) — right-click removes onset at time t
        bar_changed(int)     — emitted when bar navigation happens here
    """

    onset_added = Signal(float)
    onset_removed = Signal(float)
    bar_changed = Signal(int)
    zoom_changed = Signal(int)  # new zoom_bars value

    def __init__(self, parent=None):
        super().__init__(parent)
        self._markers: AudioMarkers | None = None
        self._grid: TimeGrid | None = None

        self._current_bar = 1
        self._zoom_bars = 3

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._scene.setBackgroundBrush(QBrush(BG_COLOR))

        if _USE_OPENGL:
            try:
                self.setViewport(QOpenGLWidget())
                self.setViewportUpdateMode(
                    QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
            except Exception:
                pass

        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMouseTracking(True)

        self._envelope_path: QGraphicsPathItem | None = None
        self._grid_lines: list[QGraphicsLineItem] = []
        self._onset_lines: list[QGraphicsLineItem] = []
        self._tick_labels: list[QGraphicsSimpleTextItem] = []

    # ── Public API ────────────────────────────────────────────────────────

    def set_markers(self, markers: AudioMarkers) -> None:
        self._markers = markers
        self._grid = markers.grid
        self._current_bar = 1
        self._rebuild_scene()

    def refresh(self) -> None:
        if self._markers:
            self._rebuild_scene()

    def sync_view(self, bar: int, zoom_bars: int) -> None:
        """Sync time window from the CQT view."""
        self._current_bar = bar
        self._zoom_bars = zoom_bars
        self._rebuild_scene()

    def time_window(self) -> tuple[float, float]:
        """Public wrapper for the current visible time window."""
        return self._time_window()

    @property
    def current_bar(self) -> int:
        return self._current_bar

    @property
    def zoom_bars(self) -> int:
        return self._zoom_bars

    # ── Time window (mirrors CQT view logic) ─────────────────────────────

    def _time_window(self) -> tuple[float, float]:
        if not self._grid or not self._markers:
            return 0.0, 1.0
        return compute_time_window(
            current_bar=self._current_bar,
            zoom_bars=self._zoom_bars,
            bar_duration=self._grid.bar_dur,
            total_duration=self._markers.duration,
        )

    # ── Scene rebuild ─────────────────────────────────────────────────────

    def _rebuild_scene(self) -> None:
        if not self._markers or not self._grid:
            return

        t_start, t_end = self._time_window()
        m = self._markers
        g = self._grid

        self._scene.clear()
        self._grid_lines.clear()
        self._onset_lines.clear()
        self._tick_labels.clear()
        self._envelope_path = None

        # Reserve top strip for labels (scene Y 1.0..1.15)
        label_y = 1.08   # scene Y for label text placement

        # ── Note-duration grid: ticks + labels ─────────────────────────────
        step_dur = float(m.current_grid_duration()) if m else 0.0
        if step_dur <= 0.0:
            step_dur = g.beat_dur / 4.0
        steps_per_beat = max(1, int(round(g.beat_dur / step_dur)))
        steps_per_bar = max(1, int(g.beats_per_bar) * steps_per_beat)

        first_step = int(t_start / step_dur)
        last_step = int(t_end / step_dur) + 1

        for idx in range(first_step, last_step + 1):
            t = idx * step_dur
            if t < t_start or t > t_end:
                continue

            # Determine whether this tick lands on a beat or bar.
            is_bar = (idx % steps_per_bar == 0)
            is_beat = (idx % steps_per_beat == 0)

            # Grid line
            if is_bar:
                pen = QPen(BAR_LINE_COLOR)
                pen.setWidthF(2.0)
            elif is_beat:
                pen = QPen(BEAT_LINE_COLOR)
                pen.setWidthF(1.0)
            else:
                pen = QPen(SIXTEENTH_LINE_COLOR)
                pen.setWidthF(0.5)
            pen.setCosmetic(True)
            line = self._scene.addLine(QLineF(t, 0.0, t, 1.0), pen)
            line.setZValue(2)
            self._grid_lines.append(line)

            # Labels
            bar_num, beat_num = g.seconds_to_bar_beat(t)

            if is_bar:
                # Bold bar label: "3.1"
                lbl = self._scene.addSimpleText(f"{bar_num}.{beat_num}")
                lbl.setBrush(QBrush(BAR_LABEL_COLOR))
                lbl.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                lbl.setZValue(20)
                lbl.setFlag(
                    QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations,
                    True)
                lbl.setPos(t, label_y)
                self._tick_labels.append(lbl)
            elif is_beat:
                # Beat label: "3.2", "3.3", etc.
                lbl = self._scene.addSimpleText(f"{bar_num}.{beat_num}")
                lbl.setBrush(QBrush(BEAT_LABEL_COLOR))
                lbl.setFont(QFont("Segoe UI", 8))
                lbl.setZValue(20)
                lbl.setFlag(
                    QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations,
                    True)
                lbl.setPos(t, label_y)
                self._tick_labels.append(lbl)
            else:
                # Subdivision tick label: thin tick mark only.
                tick_line = self._scene.addLine(
                    QLineF(t, 1.0, t, 1.04),
                    QPen(SIXTEENTH_LABEL_COLOR))
                pen2 = tick_line.pen()
                pen2.setWidthF(1.0)
                pen2.setCosmetic(True)
                tick_line.setPen(pen2)
                tick_line.setZValue(19)

        # ── Envelope path ─────────────────────────────────────────────────
        env_times, env_vals = m.transient_envelope_slice(t_start, t_end)
        if len(env_times) > 1:
            env_max = float(np.max(env_vals))
            if env_max < 1e-8:
                env_max = 1.0
            env_norm = env_vals / env_max

            path = QPainterPath()
            path.moveTo(float(env_times[0]), 0.0)
            for i in range(len(env_times)):
                path.lineTo(float(env_times[i]), float(env_norm[i]))
            path.lineTo(float(env_times[-1]), 0.0)
            path.closeSubpath()

            fill_item = self._scene.addPath(
                path, QPen(Qt.PenStyle.NoPen), QBrush(ENVELOPE_FILL))
            fill_item.setZValue(3)

            stroke_path = QPainterPath()
            stroke_path.moveTo(float(env_times[0]), float(env_norm[0]))
            for i in range(1, len(env_times)):
                stroke_path.lineTo(float(env_times[i]), float(env_norm[i]))

            stroke_pen = QPen(ENVELOPE_COLOR)
            stroke_pen.setWidthF(1.5)
            stroke_pen.setCosmetic(True)
            self._envelope_path = self._scene.addPath(
                stroke_path, stroke_pen, QBrush(Qt.BrushStyle.NoBrush))
            self._envelope_path.setZValue(4)

        # ── Onset markers ─────────────────────────────────────────────────
        visible = m.visible_onset_markers(t_start, t_end)
        for t, source in visible:
            color = QColor(ONSET_USER_COLOR if source == "manual" else ONSET_COLOR)
            color.setAlphaF(1.0 if source == "manual" else 0.5)
            onset_pen = QPen(color)
            onset_pen.setWidthF(2.0)
            onset_pen.setCosmetic(True)
            line = self._scene.addLine(QLineF(t, 0.0, t, 1.0), onset_pen)
            line.setZValue(6)
            self._onset_lines.append(line)

        # ── Fit view ──────────────────────────────────────────────────────
        self._fit_view_to_window(t_start, t_end)

    def _fit_view_to_window(self, t_start: float, t_end: float) -> None:
        """Set view transform: X = seconds, Y includes label strip at top."""
        vw = self.viewport().width()
        vh = self.viewport().height()
        if vw <= 0 or vh <= 0:
            return

        # Scene Y range: 0 (bottom) to ~1.15 (top, includes label strip)
        y_lo = 0.0
        y_hi = 1.15
        y_range = y_hi - y_lo

        sx = vw / max(t_end - t_start, 0.001)
        sy = vh / y_range

        # Y-flip: scene Y_hi maps to pixel 0 (top of widget)
        transform = QTransform(sx, 0, 0, -sy, -t_start * sx, y_hi * sy)
        self.setTransform(transform)
        self.setSceneRect(QRectF(t_start, y_lo, t_end - t_start, y_range))

    # ── Navigation ────────────────────────────────────────────────────────

    def _navigate_to_bar(self, bar: int) -> None:
        if not self._grid or not self._markers:
            return
        total_bars = max(1, int(self._markers.duration / self._grid.bar_dur))
        self._current_bar = max(1, min(bar, total_bars))
        self._rebuild_scene()
        self.bar_changed.emit(self._current_bar)

    # ── Qt events ─────────────────────────────────────────────────────────

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key.Key_Left:
            self._navigate_to_bar(self._current_bar - 1)
        elif key == Qt.Key.Key_Right:
            self._navigate_to_bar(self._current_bar + 1)
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        old = self._zoom_bars
        if delta > 0 and self._zoom_bars > 1:
            self._zoom_bars -= 1
        elif delta < 0 and self._zoom_bars < 16:
            self._zoom_bars += 1
        if self._zoom_bars != old:
            self._rebuild_scene()
            self.zoom_changed.emit(self._zoom_bars)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        scene_pos = self.mapToScene(event.pos())
        if event.button() == Qt.MouseButton.LeftButton:
            self.onset_added.emit(scene_pos.x())
        elif event.button() == Qt.MouseButton.RightButton:
            self.onset_removed.emit(scene_pos.x())
        super().mousePressEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        t_start, t_end = self._time_window()
        self._fit_view_to_window(t_start, t_end)
