"""
ui/view_state.py
================
Shared view-state helpers for time-windowed timeline views.
"""

from PySide6.QtGui import QColor, QPen


def compute_time_window(current_bar: int, zoom_bars: int, bar_duration: float,
                        total_duration: float) -> tuple[float, float]:
    """Compute [t_start, t_end] for a centered bar-based viewport."""
    safe_bar = max(1, int(current_bar))
    safe_zoom = max(1, int(zoom_bars))
    safe_bar_duration = max(float(bar_duration), 1e-9)
    safe_total_duration = max(float(total_duration), 0.0)

    bar_start = float(safe_bar - 1) * safe_bar_duration
    t_start = max(0.0, bar_start - 0.5 * safe_bar_duration)
    t_end = min(safe_total_duration, t_start + safe_zoom * safe_bar_duration)
    return t_start, t_end


def compute_grid_steps(beat_duration: float, beats_per_bar: int,
                       requested_step_duration: float) -> tuple[float, int, int]:
    """Return normalized step duration and step counts for beat/bar ticks."""
    beat_dur = max(float(beat_duration), 1e-9)
    step_dur = float(requested_step_duration)
    if step_dur <= 0.0:
        step_dur = beat_dur / 4.0

    steps_per_beat = max(1, int(round(beat_dur / step_dur)))
    steps_per_bar = max(1, int(beats_per_bar) * steps_per_beat)
    return step_dur, steps_per_beat, steps_per_bar


def build_onset_pen(source: str, user_color: QColor, guess_color: QColor,
                    width: float) -> QPen:
    """Create a consistent onset marker pen for manual vs guessed events."""
    color = QColor(user_color if source == "manual" else guess_color)
    color.setAlphaF(1.0 if source == "manual" else 0.5)
    pen = QPen(color)
    pen.setWidthF(float(width))
    pen.setCosmetic(True)
    return pen
