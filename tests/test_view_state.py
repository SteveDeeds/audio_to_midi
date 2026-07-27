import unittest

from ui.view_state import compute_time_window


class TestViewState(unittest.TestCase):
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
