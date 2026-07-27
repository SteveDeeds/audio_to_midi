"""
ui/view_state.py
================
Shared view-state helpers for time-windowed timeline views.
"""


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
