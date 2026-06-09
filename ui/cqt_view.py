"""
ui/cqt_view.py
==============
QGraphicsView-based CQT spectrogram viewer.
Hardware-accelerated via QOpenGLWidget.

Scene coordinates: X = seconds, Y = MIDI note.
View transform flips Y so high notes appear at top.
"""

import time
import numpy as np

from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsLineItem, QGraphicsRectItem, QGraphicsSimpleTextItem,
)
from PySide6.QtCore import Qt, QRectF, QLineF, Signal
from PySide6.QtGui import (
    QPixmap, QImage, QPen, QColor, QBrush, QPainter, QFont,
    QTransform, QWheelEvent, QKeyEvent, QMouseEvent,
)

try:
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    _USE_OPENGL = True
except ImportError:
    _USE_OPENGL = False

from core.audio_engine import AudioMarkers, TimeGrid
from core.chord_analysis import chord_to_piano_notes

# ── Constants ─────────────────────────────────────────────────────────────────

NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

BAR_LINE_COLOR = QColor("#808080")
BEAT_LINE_COLOR = QColor("#555555")
MIN_GRID_LINE_COLOR = QColor("#2f2f2f")
ONSET_COLOR = QColor("#1a8fff")
ONSET_USER_COLOR = QColor("#ffffff")
CQT_BG = QColor("#0f0f1a")

# Inferno LUT (256×4 uint8)
def _build_inferno_lut() -> np.ndarray:
    lut = np.zeros((256, 4), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        r = max(0, min(255, int(255 * (
            -0.0155 + 1.175 * t**0.42 + 0.38 * t**3 - 0.62 * t**6))))
        g = max(0, min(255, int(255 * (
            -0.01 + 0.11 * t + 1.85 * t**2.5 - 1.95 * t**4))))
        b = max(0, min(255, int(255 * (
            0.015 + 1.75 * t**0.7 - 2.8 * t**1.5 + 1.75 * t**3))))
        lut[i] = [r, g, b, 255]
    return lut

INFERNO_LUT = _build_inferno_lut()

# Pitch box styles
PITCH_STYLES = {
    "manual": (QColor("#ffffff"), Qt.PenStyle.SolidLine, 1.0),
    "ml":     (QColor("#1a8fff"), Qt.PenStyle.DashLine, 0.50),
    "pyin":   (QColor("#1a8fff"), Qt.PenStyle.DashLine, 0.50),
}


# ── CQT → QPixmap ────────────────────────────────────────────────────────────

def cqt_to_qpixmap(cqt_slice: np.ndarray, midi_lo: int, midi_hi: int,
                    colored_pcs: set[int] | None = None,
                    frame_colored_pcs: np.ndarray | None = None) -> QPixmap:
    """CQT (n_bins, n_frames) → QPixmap with inferno colormap.
    Rows whose pitch class is NOT in colored_pcs are desaturated.
    If colored_pcs is None, all rows keep full color."""
    vmin, vmax = -50.0, 0.0
    norm = np.clip((cqt_slice - vmin) / (vmax - vmin), 0.0, 1.0)
    indices = (norm * 255).astype(np.uint8)
    rgba = INFERNO_LUT[indices]

    if frame_colored_pcs is not None:
        # Per-frame mask: rows are desaturated only when out-of-chord for that frame.
        row_pcs = (np.arange(midi_lo, midi_hi + 1) % 12).reshape(-1, 1)
        frame_idx = np.arange(rgba.shape[1], dtype=int).reshape(1, -1)
        keep_mask = frame_colored_pcs[row_pcs, frame_idx]
        gray_mask = ~keep_mask
        rgb = rgba[:, :, :3].astype(np.float32)
        gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.uint8)
        rgba[:, :, 0][gray_mask] = gray[gray_mask]
        rgba[:, :, 1][gray_mask] = gray[gray_mask]
        rgba[:, :, 2][gray_mask] = gray[gray_mask]
    elif colored_pcs is not None:
        row_pcs = np.arange(midi_lo, midi_hi + 1) % 12
        for row_idx, pc in enumerate(row_pcs):
            if row_idx >= rgba.shape[0]:
                break
            if pc not in colored_pcs:
                rgb = rgba[row_idx, :, :3].astype(np.float32)
                gray = (0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2])
                rgba[row_idx, :, 0] = gray.astype(np.uint8)
                rgba[row_idx, :, 1] = gray.astype(np.uint8)
                rgba[row_idx, :, 2] = gray.astype(np.uint8)

    rgba = np.ascontiguousarray(rgba)
    h, w, _ = rgba.shape
    qimg = QImage(rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


# ── CQTGraphicsView ──────────────────────────────────────────────────────────

class CQTGraphicsView(QGraphicsView):
    """Hardware-accelerated CQT viewer.

    Signals:
        bar_changed(int) — emitted when current bar changes
        block_clicked(float) — emitted with scene X (seconds) on left click
        block_right_clicked(float) — emitted with scene X on right click
    """

    bar_changed = Signal(int)
    block_clicked = Signal(float, int)       # (time_sec, midi_note)
    block_right_clicked = Signal(float, int) # (time_sec, midi_note)
    zoom_changed = Signal(int)               # new zoom_bars value

    def __init__(self, parent=None):
        super().__init__(parent)
        self._markers: AudioMarkers | None = None
        self._grid: TimeGrid | None = None
        self._scale_pcs: set[int] = set()

        self._midi_lo = 12
        self._midi_hi = 94
        self._view_midi_lo = 12
        self._view_midi_window = 48
        self._current_bar = 1
        self._zoom_bars = 3

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._scene.setBackgroundBrush(QBrush(CQT_BG))

        if _USE_OPENGL:
            try:
                self.setViewport(QOpenGLWidget())
                self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
            except Exception:
                pass

        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMouseTracking(True)

        self._cqt_pixmap_item: QGraphicsPixmapItem | None = None
        self._grid_lines: list[QGraphicsLineItem] = []
        self._onset_lines: list[QGraphicsLineItem] = []
        self._pitch_rects: list[QGraphicsRectItem] = []
        self._y_labels: list[QGraphicsSimpleTextItem] = []
        self._chord_results: list = []
        self._view_mode: str = "Gray off-scale"
        self._cqt_source: str = "db"

    # ── Public API ────────────────────────────────────────────────────────

    def set_markers(self, markers: AudioMarkers, scale_pcs: set[int]) -> None:
        """Attach an AudioMarkers object and initial scale."""
        self._markers = markers
        self._grid = markers.grid
        self._scale_pcs = scale_pcs
        self._midi_lo = markers.cqt_fmin_midi
        self._midi_hi = markers.cqt_fmin_midi + markers.cqt_n_bins - 1
        self._view_midi_lo = markers.cqt_fmin_midi
        self._current_bar = 1
        self._navigate_to_bar(1)

    def refresh(self) -> None:
        """Re-render the current view (e.g. after pitch detection finishes)."""
        if self._markers:
            self._rebuild_scene()

    def set_scale_pcs(self, pcs: set[int]) -> None:
        self._scale_pcs = pcs
        self.refresh()

    def set_chord_results(self, results: list) -> None:
        self._chord_results = results
        self.refresh()

    def set_view_mode(self, mode: str) -> None:
        self._view_mode = mode
        self.refresh()

    def set_cqt_source(self, source: str) -> None:
        self._cqt_source = source
        self.refresh()

    @property
    def current_bar(self) -> int:
        return self._current_bar

    @property
    def zoom_bars(self) -> int:
        return self._zoom_bars

    # ── Navigation ────────────────────────────────────────────────────────

    def _time_window(self) -> tuple[float, float]:
        if not self._grid or not self._markers:
            return 0.0, 1.0
        bar_start = (self._current_bar - 1) * self._grid.bar_dur
        t_start = bar_start - 0.5 * self._grid.bar_dur
        t_end = t_start + self._zoom_bars * self._grid.bar_dur
        t_start = max(0.0, t_start)
        t_end = min(self._markers.duration, t_end)
        return t_start, t_end

    def _navigate_to_bar(self, bar: int) -> None:
        if not self._grid or not self._markers:
            return
        total_bars = max(1, int(self._markers.duration / self._grid.bar_dur))
        self._current_bar = max(1, min(bar, total_bars))
        self._rebuild_scene()
        self.bar_changed.emit(self._current_bar)

    def _shift_pitch_view(self, delta: int) -> None:
        max_lo = self._midi_hi + 1 - self._view_midi_window
        self._view_midi_lo = max(self._midi_lo, min(self._view_midi_lo + delta, max_lo))
        t_start, t_end = self._time_window()
        self._fit_view_to_window(t_start, t_end)
        self._update_label_positions()
        self.viewport().update()

    # ── Scene rebuild ─────────────────────────────────────────────────────

    def _get_colored_pcs(self, t_start: float, t_end: float) -> set[int] | None:
        """Return the set of pitch classes to keep colored, or None for all."""
        if self._view_mode == "Show all":
            return None
        if self._view_mode == "Gray off-chord" and self._chord_results:
            # Union of all chord PCs active in this time window
            pcs: set[int] = set()
            for row in self._chord_results:
                _, _, _, chord_name, _, ct0, ct1 = row
                if ct1 < t_start or ct0 > t_end:
                    continue
                notes = chord_to_piano_notes(chord_name)
                pcs.update(n % 12 for n in notes)
            return pcs if pcs else None
        # Default: gray off-scale
        return self._scale_pcs if self._scale_pcs else None

    def _build_frame_chord_mask(self, frame_times: np.ndarray) -> np.ndarray | None:
        """Build a (12, n_frames) boolean mask for Gray off-chord mode."""
        if self._view_mode != "Gray off-chord" or not self._chord_results:
            return None
        n_frames = len(frame_times)
        if n_frames == 0:
            return None
        mask = np.zeros((12, n_frames), dtype=bool)
        for _, _, _, chord_name, _, ct0, ct1 in self._chord_results:
            notes = chord_to_piano_notes(chord_name)
            if not notes:
                continue
            pcs = sorted({int(n) % 12 for n in notes})
            col_mask = (frame_times >= ct0) & (frame_times < ct1)
            if not np.any(col_mask):
                continue
            for pc in pcs:
                mask[pc, col_mask] = True
        # If no chord coverage for visible frames, fall back to no masking.
        if not np.any(mask):
            return None
        return mask

    def _cqt_matrix_for_source(self) -> np.ndarray:
        if not self._markers:
            return np.zeros((1, 1), dtype=np.float32)
        m = self._markers
        if self._cqt_source in ("db", "cqt_db", "cqt_mag"):
            return m.cqt_db
        if self._cqt_source in ("spectral peel", "spectral_peel", "cqt_sal"):
            return m.cqt_peel_db
        return m.cqt_db

    def _rebuild_scene(self) -> None:
        if not self._markers or not self._grid:
            return
        t_start, t_end = self._time_window()
        m = self._markers

        self._scene.clear()
        self._grid_lines.clear()
        self._onset_lines.clear()
        self._pitch_rects.clear()
        self._y_labels.clear()
        self._cqt_pixmap_item = None

        # CQT pixmap
        frame_lo = int(np.searchsorted(m.cqt_times, t_start))
        frame_hi = int(np.searchsorted(m.cqt_times, t_end))
        frame_hi = min(frame_hi, m.cqt_db.shape[1] - 1)

        if frame_hi > frame_lo:
            cqt_source = self._cqt_matrix_for_source()
            cqt_slice = cqt_source[:, frame_lo:frame_hi]
            t_lo = float(m.cqt_times[frame_lo])
            t_hi = float(m.cqt_times[frame_hi])
            frame_times = m.cqt_times[frame_lo:frame_hi]
            frame_chord_mask = self._build_frame_chord_mask(frame_times)
            colored_pcs = self._get_colored_pcs(t_start, t_end)
            pixmap = cqt_to_qpixmap(
                cqt_slice,
                self._midi_lo,
                self._midi_hi,
                colored_pcs,
                frame_chord_mask,
            )
            self._cqt_pixmap_item = self._scene.addPixmap(pixmap)
            n_bins, n_frames = cqt_slice.shape
            sx = (t_hi - t_lo) / max(n_frames, 1)
            sy = (self._midi_hi - self._midi_lo + 1) / max(n_bins, 1)
            transform = QTransform()
            transform.translate(t_lo, self._midi_lo - 0.5)
            transform.scale(sx, sy)
            self._cqt_pixmap_item.setTransform(transform)
            self._cqt_pixmap_item.setZValue(1)

        # Minimum/beat/bar grid (16th-note minimum grid)
        sixteenth_dur = self._grid.beat_dur / 4.0
        first_16th = int(t_start / sixteenth_dur)
        last_16th = int(t_end / sixteenth_dur) + 1
        for idx in range(first_16th, last_16th + 1):
            t = idx * sixteenth_dur
            if t < t_start or t > t_end:
                continue

            beat_idx_in_bar = (idx % (self._grid.beats_per_bar * 4)) // 4
            sub = idx % 4
            is_bar = (beat_idx_in_bar == 0 and sub == 0)
            is_beat = (sub == 0)

            if is_bar:
                pen = QPen(BAR_LINE_COLOR)
                pen.setWidthF(1.5)
            elif is_beat:
                pen = QPen(BEAT_LINE_COLOR)
                pen.setWidthF(0.8)
            else:
                pen = QPen(MIN_GRID_LINE_COLOR)
                pen.setWidthF(0.4)
            pen.setCosmetic(True)
            line = self._scene.addLine(
                QLineF(t, self._midi_lo - 0.5, t, self._midi_hi + 0.5), pen)
            line.setZValue(3)
            self._grid_lines.append(line)

        # Onset markers
        visible = m.visible_onset_markers(t_start, t_end)
        for t, source in visible:
            color = QColor(ONSET_USER_COLOR if source == "manual" else ONSET_COLOR)
            color.setAlphaF(1.0 if source == "manual" else 0.5)
            onset_pen = QPen(color)
            onset_pen.setWidthF(1.5)
            onset_pen.setCosmetic(True)
            line = self._scene.addLine(
                QLineF(t, self._midi_lo - 0.5, t, self._midi_hi + 0.5), onset_pen)
            line.setZValue(5)
            self._onset_lines.append(line)

        # Pitch boxes
        pitches = m.get_cached_block_pitches(t_start, t_end)
        for entry in pitches:
            t0, t1, midi, source = entry if len(entry) == 4 else (*entry, "pyin")
            if midi is None:
                continue
            color, style, alpha = PITCH_STYLES.get(source, PITCH_STYLES["pyin"])
            c = QColor(color)
            c.setAlphaF(alpha)
            pen = QPen(c)
            pen.setWidthF(2.0)
            pen.setCosmetic(True)
            pen.setStyle(style)
            rect = self._scene.addRect(
                QRectF(max(t0, t_start), midi - 0.45,
                       min(t1, t_end) - max(t0, t_start), 0.9),
                pen, QBrush(Qt.BrushStyle.NoBrush))
            rect.setZValue(8)
            self._pitch_rects.append(rect)

        # Y-axis note labels
        C_MAJOR = {0, 2, 4, 5, 7, 9, 11}
        for midi in range(self._midi_lo, self._midi_hi + 1):
            if (midi % 12) not in C_MAJOR:
                continue
            name = f"{NOTE_NAMES[midi % 12]}{(midi // 12) - 1}"
            text = self._scene.addSimpleText(name)
            text.setBrush(QBrush(QColor("#cccccc")))
            text.setFont(QFont("Segoe UI", 8))
            text.setZValue(20)
            text.setFlag(
                QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            text.setData(0, midi)
            self._y_labels.append(text)

        self._fit_view_to_window(t_start, t_end)

    def _fit_view_to_window(self, t_start: float, t_end: float) -> None:
        midi_lo_view = self._view_midi_lo - 0.5
        midi_hi_view = self._view_midi_lo + self._view_midi_window - 0.5
        vw = self.viewport().width()
        vh = self.viewport().height()
        if vw <= 0 or vh <= 0:
            return
        sx = vw / max(t_end - t_start, 0.001)
        sy = vh / max(midi_hi_view - midi_lo_view, 0.001)
        transform = QTransform(sx, 0, 0, -sy, -t_start * sx, midi_hi_view * sy)
        self.setTransform(transform)
        self.setSceneRect(QRectF(
            t_start, midi_lo_view, t_end - t_start, midi_hi_view - midi_lo_view))
        self._update_label_positions()

    def _update_label_positions(self) -> None:
        if not self._y_labels:
            return
        left_scene = self.mapToScene(5, 0).x()
        for text in self._y_labels:
            midi = text.data(0)
            text.setPos(left_scene, midi)

    # ── Qt events ─────────────────────────────────────────────────────────

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key.Key_Left:
            self._navigate_to_bar(self._current_bar - 1)
        elif key == Qt.Key.Key_Right:
            self._navigate_to_bar(self._current_bar + 1)
        elif key == Qt.Key.Key_Up:
            self._shift_pitch_view(12)
        elif key == Qt.Key.Key_Down:
            self._shift_pitch_view(-12)
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
        midi_note = max(self._midi_lo, min(int(round(scene_pos.y())), self._midi_hi))
        if event.button() == Qt.MouseButton.LeftButton:
            self.block_clicked.emit(scene_pos.x(), midi_note)
        elif event.button() == Qt.MouseButton.RightButton:
            self.block_right_clicked.emit(scene_pos.x(), midi_note)
        super().mousePressEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        t_start, t_end = self._time_window()
        self._fit_view_to_window(t_start, t_end)
