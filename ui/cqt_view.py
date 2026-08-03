"""
ui/cqt_view.py
==============
QGraphicsView-based CQT spectrogram viewer.
Hardware-accelerated via QOpenGLWidget.

Scene coordinates: X = seconds, Y = MIDI note.
View transform flips Y so high notes appear at top.
"""

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
from ui.view_state import build_onset_pen, compute_grid_steps, compute_time_window

# ── Constants ─────────────────────────────────────────────────────────────────

NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

# Drum MIDI note names (GM percussion standard)
DRUM_MIDI_NAMES = {
    36: "Kick Drum",
    37: "Rim Shot",
    38: "Acoustic Snare",
    39: "Hand Clap",
    40: "Electric Snare",
    41: "Low Floor Tom",     # Primary Floor Tom
    42: "Hi-Hat Closed",
    43: "High Floor Tom",    # Secondary Floor Tom
    44: "Pedal Hi-Hat",
    45: "Low-Mid Tom",       # Primary Mid/Rack Tom 2
    46: "Hi-Hat Open",
    47: "High-Mid Tom",      # Extra Rack Tom
    48: "Hi-Mid Tom 2",      # Extra Rack Tom
    49: "Crash Cymbal 1",
    50: "High Tom",          # Primary High/Rack Tom 1
    51: "Ride Cymbal 1",
    52: "Chinese Cymbal",
    53: "Ride Bell",
    54: "Tambourine",
    55: "Splash Cymbal",
    56: "Cowbell",
    57: "Crash Cymbal 2",
    58: "Vibraslap",
    59: "Ride Cymbal 2",
    60: "Bongo High",
    61: "Bongo Low",
    62: "Conga High",
    63: "Conga Low",
    64: "Conga Low Muted",
    65: "Timbale High",
    66: "Timbale Low",
    67: "Agogo High",
    68: "Agogo Low",
    69: "Guiro Long",
    70: "Guiro Short",
    71: "Claves",
    72: "Wood Block High",
    73: "Wood Block Low",
    74: "Cuica Muted",
    75: "Cuica Open",
    76: "Triangle Muted",
    77: "Triangle Open",
    78: "Shaker",
    79: "Jingle Bell",
    80: "Bell Tree",
    81: "Castanets"
}

BAR_LINE_COLOR = QColor("#808080")
BEAT_LINE_COLOR = QColor("#555555")
MIN_GRID_LINE_COLOR = QColor("#2f2f2f")
ONSET_COLOR = QColor("#1a8fff")
ONSET_USER_COLOR = QColor("#ffffff")
CQT_BG = QColor("#0f0f1a")

# Inferno LUT — exact values sampled from matplotlib's inferno colormap (256×4 uint8)
_INFERNO_RGB = [
    [0,0,3],[0,0,4],[0,0,6],[1,0,7],[1,1,9],[1,1,11],[2,1,14],[2,2,16],
    [3,2,18],[4,3,20],[4,3,22],[5,4,24],[6,4,27],[7,5,29],[8,6,31],[9,6,33],
    [10,7,35],[11,7,38],[13,8,40],[14,8,42],[15,9,45],[16,9,47],[18,10,50],[19,10,52],
    [20,11,54],[22,11,57],[23,11,59],[25,11,62],[26,11,64],[28,12,67],[29,12,69],[31,12,71],
    [32,12,74],[34,11,76],[36,11,78],[38,11,80],[39,11,82],[41,11,84],[43,10,86],[45,10,88],
    [46,10,90],[48,10,92],[50,9,93],[52,9,95],[53,9,96],[55,9,97],[57,9,98],[59,9,100],
    [60,9,101],[62,9,102],[64,9,102],[65,9,103],[67,10,104],[69,10,105],[70,10,105],[72,11,106],
    [74,11,106],[75,12,107],[77,12,107],[79,13,108],[80,13,108],[82,14,108],[83,14,109],[85,15,109],
    [87,15,109],[88,16,109],[90,17,109],[91,17,110],[93,18,110],[95,18,110],[96,19,110],[98,20,110],
    [99,20,110],[101,21,110],[102,21,110],[104,22,110],[106,23,110],[107,23,110],[109,24,110],[110,24,110],
    [112,25,110],[114,25,109],[115,26,109],[117,27,109],[118,27,109],[120,28,109],[122,28,109],[123,29,108],
    [125,29,108],[126,30,108],[128,31,107],[129,31,107],[131,32,107],[133,32,106],[134,33,106],[136,33,106],
    [137,34,105],[139,34,105],[141,35,105],[142,36,104],[144,36,104],[145,37,103],[147,37,103],[149,38,102],
    [150,38,102],[152,39,101],[153,40,100],[155,40,100],[156,41,99],[158,41,99],[160,42,98],[161,43,97],
    [163,43,97],[164,44,96],[166,44,95],[167,45,95],[169,46,94],[171,46,93],[172,47,92],[174,48,91],
    [175,49,91],[177,49,90],[178,50,89],[180,51,88],[181,51,87],[183,52,86],[184,53,86],[186,54,85],
    [187,55,84],[189,55,83],[190,56,82],[191,57,81],[193,58,80],[194,59,79],[196,60,78],[197,61,77],
    [199,62,76],[200,62,75],[201,63,74],[203,64,73],[204,65,72],[205,66,71],[207,68,70],[208,69,68],
    [209,70,67],[210,71,66],[212,72,65],[213,73,64],[214,74,63],[215,75,62],[217,77,61],[218,78,59],
    [219,79,58],[220,80,57],[221,82,56],[222,83,55],[223,84,54],[224,86,52],[226,87,51],[227,88,50],
    [228,90,49],[229,91,48],[230,92,46],[230,94,45],[231,95,44],[232,97,43],[233,98,42],[234,100,40],
    [235,101,39],[236,103,38],[237,104,37],[237,106,35],[238,108,34],[239,109,33],[240,111,31],[240,112,30],
    [241,114,29],[242,116,28],[242,117,26],[243,119,25],[243,121,24],[244,122,22],[245,124,21],[245,126,20],
    [246,128,18],[246,129,17],[247,131,16],[247,133,14],[248,135,13],[248,136,12],[248,138,11],[249,140,9],
    [249,142,8],[249,144,8],[250,145,7],[250,147,6],[250,149,6],[250,151,6],[251,153,6],[251,155,6],
    [251,157,6],[251,158,7],[251,160,7],[251,162,8],[251,164,10],[251,166,11],[251,168,13],[251,170,14],
    [251,172,16],[251,174,18],[251,176,20],[251,177,22],[251,179,24],[251,181,26],[251,183,28],[251,185,30],
    [250,187,33],[250,189,35],[250,191,37],[250,193,40],[249,195,42],[249,197,44],[249,199,47],[248,201,49],
    [248,203,52],[248,205,55],[247,207,58],[247,209,60],[246,211,63],[246,213,66],[245,215,69],[245,217,72],
    [244,219,75],[244,220,79],[243,222,82],[243,224,86],[243,226,89],[242,228,93],[242,230,96],[241,232,100],
    [241,233,104],[241,235,108],[241,237,112],[241,238,116],[241,240,121],[241,242,125],[242,243,129],[242,244,133],
    [243,246,137],[244,247,141],[245,248,145],[246,250,149],[247,251,153],[249,252,157],[250,253,160],[252,254,164],
]
_rgb = np.array(_INFERNO_RGB, dtype=np.uint8)
INFERNO_LUT = np.column_stack([_rgb, np.full(256, 255, dtype=np.uint8)])

# Pitch box styles
PITCH_STYLES = {
    "manual": (QColor("#ffffff"), Qt.PenStyle.SolidLine, 1.0),
    "ml":     (QColor("#1a8fff"), Qt.PenStyle.DashLine, 0.50),
    "pyin":   (QColor("#1a8fff"), Qt.PenStyle.DashLine, 0.50),
}

# Percussive layout constants
PERC_DRUM_MIN = 36
PERC_DRUM_MAX = 81
PERC_LANES_DEFAULT = [
    36,  # Kick Drum
    37,  # Rim Shot
    38,  # Acoustic Snare
    41,  # Low Floor Tom
    45,  # Low-Mid Tom
    50,  # High Tom
    42,  # Hi-Hat Closed
    44,  # Pedal Hi-Hat
    46,  # Hi-Hat Open
    49,  # Crash Cymbal 1
    51,  # Ride Cymbal 1
]
PERC_BASE_Y = 36


# ── CQT → QPixmap ────────────────────────────────────────────────────────────

def cqt_to_qpixmap(cqt_slice: np.ndarray, midi_lo: int, midi_hi: int,
                    colored_pcs: set[int] | None = None,
                    frame_colored_pcs: np.ndarray | None = None,
                    vmin: float = -50.0,
                    vmax: float = 0.0,
                    log_color: bool = False) -> QPixmap:
    """CQT (n_bins, n_frames) → QPixmap with inferno colormap.
    Rows whose pitch class is NOT in colored_pcs are desaturated.
    If colored_pcs is None, all rows keep full color."""
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = np.clip((cqt_slice - vmin) / (vmax - vmin), 0.0, 1.0)
    if log_color:
        # Log-shaped mapping boosts quieter high-frequency details (e.g. cymbals).
        norm = np.log1p(24.0 * norm) / np.log1p(24.0)
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
    block_hovered = Signal(float, int)       # (time_sec, midi_note)
    zoom_changed = Signal(int)               # new zoom_bars value
    note_release_requested = Signal(str, float)  # (note_id, release_sec)

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
        self._playhead_line: QGraphicsLineItem | None = None
        self._view_mode: str = "Gray off-scale"
        self._cqt_source: str = "db"
        self._current_mode: str = "Pitched"  # "Pitched" or "Percussive"
        self._perc_lane_midis: list[int] = []
        self._perc_lane_y: dict[int, float] = {}
        self._perc_view_lo: float = float(self._midi_lo) - 0.5
        self._perc_view_hi: float = float(self._midi_hi) + 0.5
        self._perc_lane_lo: float = 35.5
        self._perc_lane_hi: float = 50.5
        self._release_drag_rect: QGraphicsRectItem | None = None
        self._release_drag_note_id: str | None = None

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

    def apply_pitch_feedback(self, t: float, midi: int, keep: bool) -> None:
        """Apply immediate visual feedback while model work runs in the background."""
        if self._current_mode == "Percussive":
            return
        matching = [
            rect for rect in self._pitch_rects
            if rect.data(3) == int(midi)
            and float(rect.data(1) or rect.rect().left()) <= float(t)
            and float(t) < float(rect.data(2) or rect.rect().right())
        ]
        if keep:
            pen = QPen(QColor("#ffffff"))
            pen.setWidthF(4.0)
            pen.setCosmetic(True)
            pen.setStyle(Qt.PenStyle.SolidLine)
            if matching:
                for rect in matching:
                    rect.setPen(pen)
                    rect.setData(4, "manual")
                self.viewport().update()
                return
            if not self._markers:
                return
            key = self._markers.get_block_key_at(float(t))
            if key is None:
                return
            t0 = float(key[0]) / float(self._markers.sr)
            block_t1 = float(key[1]) / float(self._markers.sr)
            grid_duration = float(self._markers.current_grid_duration())
            t1 = min(block_t1, t0 + grid_duration) if grid_duration > 0.0 else block_t1
            rect = self._scene.addRect(
                QRectF(t0, float(midi) - 0.45, t1 - t0, 0.9),
                pen, QBrush(Qt.BrushStyle.NoBrush),
            )
            rect.setZValue(9)
            rect.setData(1, t0)
            rect.setData(2, t1)
            rect.setData(3, int(midi))
            rect.setData(4, "manual")
            self._pitch_rects.append(rect)
        else:
            for rect in matching:
                self._scene.removeItem(rect)
                self._pitch_rects.remove(rect)
        self.viewport().update()

    def rendered_suggestion_signature(self) -> set[tuple[float, float, int, str]]:
        """Return the generated pitch boxes currently shown in the viewport."""
        signature = set()
        for rect in self._pitch_rects:
            source = str(rect.data(4) or "")
            if source not in {"nnls", "ml", "pyin"}:
                continue
            signature.add((
                round(float(rect.data(1) or rect.rect().left()), 6),
                round(float(rect.data(2) or rect.rect().right()), 6),
                int(rect.data(3)),
                source,
            ))
        return signature

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

    def set_mode(self, mode: str) -> None:
        """Set mode to 'Pitched' or 'Percussive'."""
        self._current_mode = mode
        if mode != "Percussive":
            self._view_midi_window = 48
            self._view_midi_lo = self._midi_lo
        self.refresh()

    def sync_view(self, bar: int, zoom_bars: int) -> None:
        """Sync current bar and zoom from another timeline view."""
        self._current_bar = max(1, int(bar))
        self._zoom_bars = max(1, int(zoom_bars))
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

    # ── Navigation ────────────────────────────────────────────────────────

    def _time_window(self) -> tuple[float, float]:
        if not self._grid or not self._markers:
            return 0.0, 1.0
        return compute_time_window(
            current_bar=self._current_bar,
            zoom_bars=self._zoom_bars,
            bar_duration=self._grid.bar_dur,
            total_duration=self._markers.duration,
        )

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
        
        if self._view_mode == "Gray off-chord":
            if self._chord_results:
                # Union of all chord PCs active in this time window
                pcs: set[int] = set()
                for row in self._chord_results:
                    _, _, _, chord_name, _, ct0, ct1 = row
                    if ct1 < t_start or ct0 > t_end:
                        continue
                    notes = chord_to_piano_notes(chord_name)
                    pcs.update(n % 12 for n in notes)
                if pcs:
                    return pcs
            # Fall back to scale if no chords in window or no chord results
            return self._scale_pcs if self._scale_pcs else None
        
        # Default: gray off-scale
        return self._scale_pcs if self._scale_pcs else None

    def _recommendation_visible(self, source: str, midi: int, t0: float, t1: float,
                                t_start: float, t_end: float) -> bool:
        """Return whether a pitch recommendation should remain visible.
        
        Manual pitches always show. NNLS/ML/pyin recommendations are filtered by gray mode.
        """
        if midi is None:
            return False

        # Percussive mode does not use pitch-class gray filtering.
        if self._current_mode == "Percussive":
            return True

        # User-trained predictions must remain visible so feedback can be evaluated.
        if source in {"manual", "ml"} or self._view_mode == "Show all":
            return True

        # For recommendations (nnls, ml, pyin), apply gray mode filtering
        if t1 <= t_start or t0 >= t_end:
            return False

        midi_pc = int(midi) % 12

        if self._view_mode == "Gray off-chord":
            if self._chord_results:
                # Use only chord tones active during this note block.
                chord_pcs: set[int] = set()
                for row in self._chord_results:
                    _, _, _, chord_name, _, ct0, ct1 = row
                    if ct1 <= t0 or ct0 >= t1:
                        continue
                    notes = chord_to_piano_notes(chord_name)
                    chord_pcs.update(int(n) % 12 for n in notes)
                if chord_pcs:
                    return midi_pc in chord_pcs
            # Fall back to scale when chord coverage is missing.
            if self._scale_pcs:
                return midi_pc in self._scale_pcs
            return True

        # Default: Gray off-scale
        if self._scale_pcs:
            return midi_pc in self._scale_pcs
        return True

    def visible_recommendation_rows(self, t_start: float, t_end: float) -> list[tuple]:
        """Return the visible recommendation rows using the current gray mode."""
        if not self._markers:
            return []
        if self._current_mode == "Percussive":
            return self._markers.get_cached_drum_hits(t_start, t_end)
        rows = []
        for entry in self._markers.get_pitched_note_rows(t_start, t_end):
            t0, t1, midi, source, vel, _note_id = entry
            if self._recommendation_visible(source, midi, t0, t1, t_start, t_end):
                rows.append(entry)
        return rows

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

    def _spectrogram_matrix_and_times(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._markers:
            return np.zeros((1, 1), dtype=np.float32), np.zeros(1, dtype=np.float32)
        m = self._markers
        if self._current_mode == "Percussive":
            src = self._cqt_source.strip().lower()
            if src in ("cqt (percussive)", "cqt", "db", "cqt_db", "cqt_mag"):
                return m.cqt_db, m.cqt_times
            if src in ("mel", "mel db", "melspectrogram", "mels"):
                return m.mel_db, m.mel_times
            return m.stft_db, m.stft_times
        if self._cqt_source in ("db", "cqt_db", "cqt_mag"):
            return m.cqt_db, m.cqt_times
        if self._cqt_source in ("spectral peel", "spectral_peel", "cqt_sal"):
            return m.cqt_peel_db, m.cqt_times
        if self._cqt_source in ("pitch salience", "hcqt", "cqt_hcqt_db"):
            return m.cqt_hcqt_db, m.cqt_times
        if self._cqt_source in ("flux novelty", "flux", "cqt_flux_db"):
            return m.cqt_flux_db, m.cqt_times
        return m.cqt_db, m.cqt_times

    def _rebuild_scene(self) -> None:
        if not self._markers or not self._grid:
            return
        t_start, t_end = self._time_window()
        m = self._markers

        self._cancel_release_drag()
        self._scene.clear()
        self._grid_lines.clear()
        self._onset_lines.clear()
        self._pitch_rects.clear()
        self._y_labels.clear()
        self._cqt_pixmap_item = None

        pitches = self.visible_recommendation_rows(t_start, t_end)
        if self._current_mode == "Percussive":
            self._prepare_percussive_layout(pitches)

        if self._current_mode != "Percussive":
            spec_matrix, spec_times = self._spectrogram_matrix_and_times()

            # Spectrogram pixmap
            frame_lo = int(np.searchsorted(spec_times, t_start))
            frame_hi = int(np.searchsorted(spec_times, t_end))
            frame_hi = min(frame_hi, spec_matrix.shape[1] - 1)

            if frame_hi <= frame_lo:
                frame_hi = frame_lo

        if self._current_mode != "Percussive" and frame_hi > frame_lo:
            cqt_slice = spec_matrix[:, frame_lo:frame_hi]
            t_lo = float(spec_times[frame_lo])
            t_hi = float(spec_times[frame_hi])
            frame_times = spec_times[frame_lo:frame_hi]
            frame_chord_mask = self._build_frame_chord_mask(frame_times)
            colored_pcs = self._get_colored_pcs(t_start, t_end)

            vmin = -50.0
            vmax = 0.0
            log_color = False
            pix_midi_lo = self._midi_lo
            pix_midi_hi = self._midi_hi

            pixmap = cqt_to_qpixmap(
                cqt_slice,
                pix_midi_lo,
                pix_midi_hi,
                colored_pcs,
                frame_chord_mask,
                vmin=vmin,
                vmax=vmax,
                log_color=log_color,
            )
            self._cqt_pixmap_item = self._scene.addPixmap(pixmap)
            n_bins, n_frames = cqt_slice.shape
            sx = (t_hi - t_lo) / max(n_frames, 1)
            sy = (self._midi_hi - self._midi_lo + 1) / max(n_bins, 1)
            y0 = self._midi_lo - 0.5
            transform = QTransform()
            transform.translate(t_lo, y0)
            transform.scale(sx, sy)
            self._cqt_pixmap_item.setTransform(transform)
            self._cqt_pixmap_item.setZValue(1)

        # Minimum/beat/bar grid using active note-duration spacing.
        req_step = float(self._markers.current_grid_duration()) if self._markers else 0.0
        step_dur, steps_per_beat, steps_per_bar = compute_grid_steps(
            beat_duration=self._grid.beat_dur,
            beats_per_bar=int(self._grid.beats_per_bar),
            requested_step_duration=req_step,
        )

        first_step = int(t_start / step_dur)
        last_step = int(t_end / step_dur) + 1
        for idx in range(first_step, last_step + 1):
            t = idx * step_dur
            if t < t_start or t > t_end:
                continue

            is_bar = (idx % steps_per_bar == 0)
            is_beat = (idx % steps_per_beat == 0)

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
            y_grid_lo = self._midi_lo - 0.5
            y_grid_hi = self._midi_hi + 0.5
            if self._current_mode == "Percussive":
                y_grid_lo = self._perc_lane_lo
                y_grid_hi = self._perc_lane_hi
            line = self._scene.addLine(
                QLineF(t, y_grid_lo, t, y_grid_hi), pen)
            line.setZValue(3)
            self._grid_lines.append(line)

        # Onset markers
        visible = m.visible_onset_markers(t_start, t_end)
        for t, source in visible:
            onset_pen = build_onset_pen(
                source=source,
                user_color=ONSET_USER_COLOR,
                guess_color=ONSET_COLOR,
                width=1.5,
            )
            y_onset_lo = self._midi_lo - 0.5
            y_onset_hi = self._midi_hi + 0.5
            if self._current_mode == "Percussive":
                y_onset_lo = self._perc_lane_lo
                y_onset_hi = self._perc_lane_hi
            line = self._scene.addLine(
                QLineF(t, y_onset_lo, t, y_onset_hi), onset_pen)
            line.setZValue(5)
            self._onset_lines.append(line)

        # Pitch boxes
        for entry in pitches:
            t0, t1, midi, source = entry[:4]
            if midi is None:
                continue
            y_note = float(midi)
            if self._current_mode == "Percussive":
                y_note = self._percussive_y_for_midi(int(midi))
            color, style, alpha = PITCH_STYLES.get(source, PITCH_STYLES["pyin"])
            c = QColor(color)
            c.setAlphaF(alpha)
            pen = QPen(c)
            pen.setWidthF(4.0)
            pen.setCosmetic(True)
            pen.setStyle(style)
            # In Percussive mode, use smaller height (0.4) to leave room for envelope view
            box_height = 0.4 if self._current_mode == "Percussive" else 0.9
            box_half = box_height / 2.0
            rect = self._scene.addRect(
                QRectF(max(t0, t_start), y_note - box_half,
                       min(t1, t_end) - max(t0, t_start), box_height),
                pen, QBrush(Qt.BrushStyle.NoBrush))
            rect.setZValue(8)
            if self._current_mode != "Percussive" and len(entry) >= 6:
                rect.setData(0, entry[5])
                rect.setData(1, float(t0))
                rect.setData(2, float(t1))
                rect.setData(3, int(midi))
                rect.setData(4, str(source))
            self._pitch_rects.append(rect)

        # Y-axis note labels
        if self._current_mode == "Percussive":
            lane_midis = self._perc_lane_midis
            for midi in lane_midis:
                y_lane = self._percussive_y_for_midi(midi)
                lane_pen = QPen(QColor("#3a3a44"))
                lane_pen.setCosmetic(True)
                lane_pen.setWidthF(0.8)
                lane = self._scene.addLine(
                    QLineF(t_start, y_lane, t_end, y_lane),
                    lane_pen,
                )
                lane.setZValue(2)

                name = DRUM_MIDI_NAMES.get(midi, f"D{midi}")
                text = self._scene.addSimpleText(name)
                text.setBrush(QBrush(QColor("#cccccc")))
                text.setFont(QFont("Segoe UI", 8))
                text.setZValue(20)
                text.setFlag(
                    QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
                text.setData(0, midi)
                text.setData(1, y_lane)
                self._y_labels.append(text)
        else:
            # Pitched mode: show pitch names on C major scale
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
        if self._current_mode == "Percussive":
            midi_lo_view = self._perc_view_lo
            midi_hi_view = self._perc_view_hi
        else:
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

    def set_playhead_position(self, time_sec: float) -> None:
        """Update playhead line position (vertical line at time_sec)."""
        if not self._markers:
            return
        
        # Remove old playhead if it exists
        if self._playhead_line is not None and self._playhead_line in self._scene.items():
            self._scene.removeItem(self._playhead_line)
        
        # Create new playhead line if time is within visible window
        if time_sec >= 0:
            if self._current_mode == "Percussive":
                midi_lo = self._perc_view_lo
                midi_hi = self._perc_view_hi
            else:
                midi_lo = self._view_midi_lo - 0.5
                midi_hi = self._view_midi_lo + self._view_midi_window - 0.5
            
            pen = QPen(QColor("#00ff00"), 1.0)
            pen.setStyle(Qt.PenStyle.SolidLine)
            self._playhead_line = self._scene.addLine(
                QLineF(time_sec, midi_lo, time_sec, midi_hi), pen)
            self._playhead_line.setZValue(100)

    def _prepare_percussive_layout(self, visible_rows: list[tuple]) -> None:
        lanes = self._percussive_lane_midis(visible_rows)
        if not lanes:
            lanes = [36]
        self._perc_lane_midis = lanes
        self._perc_lane_y = {
            int(midi): float(PERC_BASE_Y + idx)
            for idx, midi in enumerate(lanes)
        }

        self._perc_lane_lo = float(PERC_BASE_Y) - 0.5
        self._perc_lane_hi = float(PERC_BASE_Y + len(lanes) - 1) + 0.5
        self._perc_view_lo = self._perc_lane_lo - 1.0
        self._perc_view_hi = self._perc_lane_hi + 1.0

    def _percussive_y_for_midi(self, midi: int) -> float:
        if int(midi) in self._perc_lane_y:
            return self._perc_lane_y[int(midi)]
        lanes = self._percussive_lane_midis()
        if not lanes:
            return float(PERC_BASE_Y)
        closest = min(lanes, key=lambda m: abs(int(midi) - int(m)))
        return float(self._perc_lane_y.get(int(closest), float(PERC_BASE_Y)))

    def _percussive_lane_midis(self, visible_rows: list[tuple] | None = None) -> list[int]:
        default_lanes = [
            int(midi)
            for midi in PERC_LANES_DEFAULT
            if PERC_DRUM_MIN <= int(midi) <= PERC_DRUM_MAX
        ]
        lane_set: set[int] = set(default_lanes)
        extra_lanes: set[int] = set()

        if self._markers is not None:
            for _, midi in self._markers.model.get_drum_patterns():
                if PERC_DRUM_MIN <= int(midi) <= PERC_DRUM_MAX:
                    m = int(midi)
                    if m not in lane_set:
                        extra_lanes.add(m)
            for midi_set in self._markers._manual_drums.values():
                for midi in midi_set:
                    if PERC_DRUM_MIN <= int(midi) <= PERC_DRUM_MAX:
                        m = int(midi)
                        if m not in lane_set:
                            extra_lanes.add(m)

        if visible_rows:
            for row in visible_rows:
                if len(row) >= 3 and row[2] is not None:
                    midi = int(row[2])
                    if PERC_DRUM_MIN <= midi <= PERC_DRUM_MAX:
                        if midi not in lane_set:
                            extra_lanes.add(midi)

        return default_lanes + sorted(extra_lanes)

    def _quantize_percussive_midi(self, y_value: float) -> int:
        lanes = self._perc_lane_midis or self._percussive_lane_midis()
        if not lanes:
            return 36
        return min(
            lanes,
            key=lambda m: abs(float(y_value) - float(self._percussive_y_for_midi(int(m)))),
        )

    def _update_label_positions(self) -> None:
        if not self._y_labels:
            return
        left_scene = self.mapToScene(5, 0).x()
        for text in self._y_labels:
            y = text.data(1)
            if y is None:
                y = text.data(0)
            text.setPos(left_scene, float(y))

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
        if self._current_mode == "Percussive":
            midi_note = self._quantize_percussive_midi(scene_pos.y())
        else:
            midi_note = max(self._midi_lo, min(int(round(scene_pos.y())), self._midi_hi))
        if event.button() == Qt.MouseButton.LeftButton:
            release_rect = self._release_rect_at(event.position().toPoint())
            if release_rect is not None:
                self._release_drag_rect = release_rect
                self._release_drag_note_id = str(release_rect.data(0))
                event.accept()
                return
            self.block_clicked.emit(scene_pos.x(), midi_note)
        elif event.button() == Qt.MouseButton.RightButton:
            self.block_right_clicked.emit(scene_pos.x(), midi_note)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        scene_pos = self.mapToScene(event.pos())
        if self._release_drag_rect is not None and self._release_drag_note_id is not None:
            release = self._markers.preview_note_release(
                self._release_drag_note_id, float(scene_pos.x())
            ) if self._markers else None
            if release is not None:
                try:
                    rect = self._release_drag_rect.rect()
                    rect.setRight(float(release))
                    self._release_drag_rect.setRect(rect)
                except RuntimeError:
                    self._cancel_release_drag()
            event.accept()
            return
        if self._current_mode == "Percussive":
            midi_note = self._quantize_percussive_midi(scene_pos.y())
        else:
            midi_note = max(self._midi_lo, min(int(round(scene_pos.y())), self._midi_hi))
        self.block_hovered.emit(float(scene_pos.x()), int(midi_note))
        cursor = (
            Qt.CursorShape.SizeHorCursor
            if self._release_rect_at(event.position().toPoint()) is not None
            else Qt.CursorShape.ArrowCursor
        )
        self.viewport().setCursor(cursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._release_drag_rect is not None
            and self._release_drag_note_id is not None
        ):
            note_id = self._release_drag_note_id
            release = float(self.mapToScene(event.pos()).x())
            self._cancel_release_drag()
            self.note_release_requested.emit(note_id, release)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _cancel_release_drag(self) -> None:
        self._release_drag_rect = None
        self._release_drag_note_id = None
        self.viewport().setCursor(Qt.CursorShape.ArrowCursor)

    def _release_rect_at(self, viewport_pos) -> QGraphicsRectItem | None:
        if self._current_mode == "Percussive" or self._release_drag_rect is not None:
            return None
        scene_pos = self.mapToScene(viewport_pos)
        for rect in reversed(self._pitch_rects):
            note_id = rect.data(0)
            if not note_id:
                continue
            release = float(rect.data(2))
            release_view_x = self.mapFromScene(release, scene_pos.y()).x()
            if abs(release_view_x - viewport_pos.x()) > 7:
                continue
            note_rect = rect.rect()
            if note_rect.top() <= scene_pos.y() <= note_rect.bottom():
                return rect
        return None

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        t_start, t_end = self._time_window()
        self._fit_view_to_window(t_start, t_end)
