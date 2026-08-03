import unittest
from types import SimpleNamespace

from ui.cqt_view import CQTGraphicsView
from ui.main_window import MainWindow
from ui.view_state import compute_time_window


class TestViewState(unittest.TestCase):
    def test_rejecting_generated_pitch_does_not_mark_block_as_rest(self) -> None:
        calls = []
        markers = SimpleNamespace(
            remove_manual_pitch=lambda t, midi: calls.append(("reject", t, midi)),
            mark_as_rest=lambda t: calls.append(("rest", t)),
        )
        window = SimpleNamespace(
            _markers=markers,
            _controls=SimpleNamespace(get_mode=lambda: "Pitched"),
            _cqt_view=SimpleNamespace(
                apply_pitch_feedback=lambda t, midi, keep: calls.append(
                    ("feedback", t, midi, keep)
                )
            ),
            _pitch_timer=SimpleNamespace(start=lambda: calls.append(("timer",))),
            _status_bar=SimpleNamespace(showMessage=lambda message: None),
            _pitch_update_dirty=False,
        )

        MainWindow._on_cqt_right_clicked(window, 36.75, 36)

        self.assertIn(("reject", 36.75, 36), calls)
        self.assertNotIn(("rest", 36.75), calls)

    def test_learned_prediction_is_visible_when_off_scale(self) -> None:
        view = SimpleNamespace(
            _current_mode="Pitched",
            _view_mode="Gray off-scale",
            _scale_pcs={0, 2, 3, 5, 7, 9, 10},
            _chord_results=[],
        )

        visible = CQTGraphicsView._recommendation_visible(
            view, "ml", 64, 36.75, 36.875, 36.0, 38.0
        )

        self.assertTrue(visible)

    def test_compute_time_window_clamps_to_duration(self) -> None:
        t_start, t_end = compute_time_window(
            current_bar=1,
            zoom_bars=3,
            bar_duration=2.0,
            total_duration=4.0,
        )
        self.assertEqual(t_start, 0.0)
        self.assertEqual(t_end, 4.0)

    def test_compute_time_window_centered_bar(self) -> None:
        t_start, t_end = compute_time_window(
            current_bar=4,
            zoom_bars=3,
            bar_duration=2.0,
            total_duration=30.0,
        )
        self.assertEqual(t_start, 5.0)
        self.assertEqual(t_end, 11.0)


if __name__ == "__main__":
    unittest.main()
