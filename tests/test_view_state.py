import inspect
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ui.cqt_view import CQTGraphicsView
from ui.controls import ControlPanel, infer_timing_from_path
from ui.main_window import MainWindow, _playback_mix_levels
from ui.view_state import compute_time_window
from core.playback import _midi_mix_messages, _stereo_audio_int16


class TestViewState(unittest.TestCase):
    def test_timing_is_inferred_from_audio_filename(self) -> None:
        path = os.path.join(
            "songs", "fallback_4-4_120bpm",
            "MySoulWillWait_FMaj_Trimmed_6-8_163bpm.wav",
        )

        bpm, meter = infer_timing_from_path(path)

        self.assertEqual(bpm, 163.0)
        self.assertEqual(meter, (6, 8))

    def test_missing_filename_timing_falls_back_to_folder(self) -> None:
        path = os.path.join("songs", "My Song 7-8 142.5BPM", "vocals.wav")

        bpm, meter = infer_timing_from_path(path)

        self.assertEqual(bpm, 142.5)
        self.assertEqual(meter, (7, 8))

    def test_filename_values_override_folder_values_per_field(self) -> None:
        path = os.path.join("songs", "session_4-4_120bpm", "bass_6-8.wav")

        bpm, meter = infer_timing_from_path(path)

        self.assertEqual(bpm, 120.0)
        self.assertEqual(meter, (6, 8))

    def test_playback_mix_slider_endpoints_and_midpoint(self) -> None:
        self.assertEqual(_playback_mix_levels(0), (1.0, 0.0))
        self.assertEqual(_playback_mix_levels(50), (0.5, 0.5))
        self.assertEqual(_playback_mix_levels(100), (0.0, 1.0))

    def test_playback_sources_use_opposite_75_percent_pan(self) -> None:
        self.assertEqual(
            _midi_mix_messages(channel=0, gain=0.5),
            ((0xB0, 10, 16), (0xB0, 7, 64)),
        )
        stereo = _stereo_audio_int16(np.array([1.0], dtype=np.float32), 1.0)
        self.assertIsNotNone(stereo)
        self.assertAlmostEqual(float(stereo[0, 0]) / float(stereo[0, 1]), 0.25, places=3)

    def test_preview_note_window_plays_all_notes_and_exact_audio_mix(self) -> None:
        calls = []
        player = SimpleNamespace(
            play=lambda blocks, t_start, audio, sr, **kwargs: calls.append(
                (blocks, t_start, audio.copy(), sr, kwargs)
            )
        )
        rows = [
            (2.0, 3.5, 60, "manual", 90),
            (2.0, 3.5, 64, "manual", 96),
        ]
        window = SimpleNamespace(
            _markers=SimpleNamespace(audio=np.arange(100), sr=10),
            _player=player,
            _controls=SimpleNamespace(
                get_play_mix=lambda: 25,
                set_playing=lambda playing: calls.append(("playing", playing)),
            ),
            _cqt_view=SimpleNamespace(rendered_note_rows=lambda *_: list(rows)),
            _playback_timer=SimpleNamespace(start=lambda: calls.append(("timer",))),
            _playback_start_time=0.0,
            _playback_start_t=0.0,
        )

        MainWindow._preview_note_window(window, 2.0, 3.5, channel=9)

        self.assertEqual(calls[0], ("timer",))
        blocks, t_start, audio, sr, kwargs = calls[1]
        self.assertEqual(blocks, rows)
        self.assertEqual(t_start, 2.0)
        np.testing.assert_array_equal(audio, np.arange(20, 35))
        self.assertEqual(sr, 10)
        self.assertEqual(kwargs, {"channel": 9, "midi_gain": 0.75, "audio_gain": 0.25})
        self.assertEqual(window._playback_start_t, 2.0)
        self.assertEqual(calls[2], ("playing", True))

    def test_transport_click_stops_when_playing(self) -> None:
        calls = []
        panel = SimpleNamespace(
            _is_playing=True,
            stop_requested=SimpleNamespace(emit=lambda: calls.append("stop")),
            play_requested=SimpleNamespace(emit=lambda mix: calls.append(("play", mix))),
            get_play_mix=lambda: 50,
        )

        ControlPanel._on_transport_clicked(panel)

        self.assertEqual(calls, ["stop"])

    def test_playback_completion_restores_play_state(self) -> None:
        calls = []
        window = SimpleNamespace(
            _player=SimpleNamespace(is_playing=False),
            _playback_timer=SimpleNamespace(stop=lambda: calls.append("timer stopped")),
            _cqt_view=SimpleNamespace(
                set_playhead_position=lambda position: calls.append(("playhead", position))
            ),
            _controls=SimpleNamespace(
                set_playing=lambda playing: calls.append(("playing", playing))
            ),
        )

        MainWindow._on_playback_tick(window)

        self.assertEqual(
            calls,
            ["timer stopped", ("playhead", -1), ("playing", False)],
        )

    def test_completed_pitch_rows_render_without_running_inference_again(self) -> None:
        calls = []
        rows = [(2.0, 3.0, 60, "nnls", 90)]
        window = SimpleNamespace(
            _markers=SimpleNamespace(),
            _pitch_suggestion_baseline=set(),
            _cqt_view=SimpleNamespace(
                refresh=lambda **kwargs: calls.append(kwargs),
            ),
            _status_bar=SimpleNamespace(showMessage=lambda message: None),
        )

        MainWindow._on_pitches_ready(window, rows, 1, False)

        self.assertEqual(calls, [{"pitch_rows": rows}])

    def test_load_completion_imports_exact_name_companion_midi(self) -> None:
        source = inspect.getsource(MainWindow._on_load_done)

        self.assertIn("_load_companion_stem_midi_if_present", source)

    def test_percussion_export_uses_published_visible_hits(self) -> None:
        source = Path("ui/main_window.py").read_text(encoding="utf-8")

        self.assertIn("notes = self._markers.get_published_drum_hits()", source)
        self.assertNotIn(
            "get_cached_drum_hits(0.0, self._markers.duration)", source
        )

    def test_rejecting_generated_pitch_does_not_mark_block_as_rest(self) -> None:
        calls = []
        markers = SimpleNamespace(
            remove_manual_pitch=lambda t, midi, origin: calls.append(
                ("reject", t, midi, origin)
            ),
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
            _learning=SimpleNamespace(request_pitch=lambda: calls.append(("timer",))),
            _status_bar=SimpleNamespace(showMessage=lambda message: None),
        )

        MainWindow._on_cqt_right_clicked(window, 36.75, 36)

        self.assertIn(("reject", 36.75, 36, "cqt_right_click"), calls)
        self.assertNotIn(("rest", 36.75), calls)

    def test_bar_navigation_rebuilds_without_inference(self) -> None:
        calls = []
        view = SimpleNamespace(
            _grid=SimpleNamespace(bar_dur=2.0),
            _markers=SimpleNamespace(duration=20.0),
            _current_bar=1,
            _rebuild_scene=lambda **kwargs: calls.append(("rebuild", kwargs)),
            bar_changed=SimpleNamespace(emit=lambda bar: calls.append(("bar", bar))),
        )

        CQTGraphicsView._navigate_to_bar(view, 3)

        self.assertEqual(view._current_bar, 3)
        self.assertEqual(calls[0], ("rebuild", {"run_inference": False}))

    def test_note_release_cursor_is_set_after_graphics_view_mouse_move(self) -> None:
        source = inspect.getsource(CQTGraphicsView.mouseMoveEvent)

        self.assertLess(
            source.rfind("super().mouseMoveEvent(event)"),
            source.rfind("self.viewport().setCursor(cursor)"),
        )

    def test_release_commit_does_not_refresh_scene(self) -> None:
        calls = []
        window = SimpleNamespace(
            _markers=SimpleNamespace(
                set_note_release=lambda note_id, release: 2.5,
                has_pending_pitch_training=lambda: True,
            ),
            _cqt_view=SimpleNamespace(
                apply_note_release=lambda note_id, release: calls.append(
                    ("release", note_id, release)
                ),
                refresh=lambda: self.fail("release commit must not rebuild the scene"),
            ),
            _learning=SimpleNamespace(request_pitch=lambda: calls.append(("timer",))),
            _status_bar=SimpleNamespace(showMessage=lambda message: None),
        )

        MainWindow._on_note_release_requested(window, "generated:1", 2.4)

        self.assertEqual(calls[0], ("release", "generated:1", 2.5))
        self.assertIn(("timer",), calls)

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
