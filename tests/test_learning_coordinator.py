import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PySide6.QtCore import QCoreApplication

from ui.learning_coordinator import LearningCoordinator


class TestLearningCoordinator(unittest.TestCase):
    _app = QCoreApplication.instance() or QCoreApplication([])

    def markers(self, pitch_pending: bool = False) -> SimpleNamespace:
        return SimpleNamespace(
            has_pending_pitch_training=lambda: pitch_pending,
            has_pending_onset_training=lambda: False,
            pending_onset_training_count=lambda: 0,
        )

    def test_onset_training_takes_priority_over_pending_pitch(self) -> None:
        coordinator = LearningCoordinator(lambda: (2.0, 4.0))
        self.addCleanup(coordinator.shutdown)
        markers = SimpleNamespace(
            has_pending_pitch_training=lambda: False,
            has_pending_onset_training=lambda: True,
            pending_onset_training_count=lambda: 12,
        )
        coordinator.set_session(markers)
        coordinator._pitch_pending = True
        coordinator._onset_pending = True

        coordinator._start_pitch()

        self.assertIsNone(coordinator._pitch_worker)
        self.assertTrue(coordinator._onset_timer.isActive())

    def test_pitch_requests_coalesce_while_debouncing(self) -> None:
        coordinator = LearningCoordinator(lambda: (2.0, 4.0))
        self.addCleanup(coordinator.shutdown)
        coordinator.set_session(self.markers())

        coordinator.request_pitch()
        coordinator.request_pitch()

        self.assertTrue(coordinator._pitch_pending)
        self.assertTrue(coordinator._pitch_timer.isActive())

    def test_onset_request_tolerates_deleted_pitch_worker(self) -> None:
        coordinator = LearningCoordinator(lambda: (2.0, 4.0))
        self.addCleanup(coordinator.shutdown)
        markers = SimpleNamespace(
            has_pending_pitch_training=lambda: False,
            has_pending_onset_training=lambda: True,
            pending_onset_training_count=lambda: 1,
        )
        coordinator.set_session(markers)
        coordinator._pitch_worker = SimpleNamespace(
            isRunning=lambda: (_ for _ in ()).throw(RuntimeError("deleted")),
            disconnect=lambda: (_ for _ in ()).throw(RuntimeError("deleted")),
        )

        coordinator.request_onset()

        self.assertTrue(coordinator._onset_pending)
        self.assertTrue(coordinator._onset_timer.isActive())

    def test_current_session_result_is_published(self) -> None:
        coordinator = LearningCoordinator(lambda: (2.0, 4.0))
        self.addCleanup(coordinator.shutdown)
        coordinator.set_session(self.markers())
        worker = SimpleNamespace()
        coordinator._pitch_worker = worker
        received = []
        coordinator.pitches_ready.connect(
            lambda rows, generation, pending: received.append(
                (rows, generation, pending)
            )
        )

        coordinator._finish_pitch(
            worker,
            coordinator._session_generation,
            [(2.0, 3.0, 60, "ml")],
            7,
            False,
        )

        self.assertEqual(received, [([(2.0, 3.0, 60, "ml")], 7, False)])

    def test_previous_session_result_is_ignored(self) -> None:
        coordinator = LearningCoordinator(lambda: (2.0, 4.0))
        self.addCleanup(coordinator.shutdown)
        coordinator.set_session(self.markers())
        old_generation = coordinator._session_generation
        coordinator.set_session(self.markers())
        received = []
        coordinator.pitches_ready.connect(lambda *args: received.append(args))

        coordinator._finish_pitch(
            SimpleNamespace(),
            old_generation,
            [(2.0, 3.0, 60, "ml")],
            7,
            False,
        )

        self.assertEqual(received, [])

    def test_scroll_interrupts_old_mono_view_and_coalesces_requests(self) -> None:
        window = [2.0, 4.0]
        coordinator = LearningCoordinator(lambda: tuple(window))
        self.addCleanup(coordinator.shutdown)
        coordinator.set_session(SimpleNamespace(
            uses_visible_pitch_model=True,
            has_pending_pitch_training=lambda: False,
        ))
        interrupted = []
        worker = SimpleNamespace(
            t_start=2.0, t_end=4.0,
            isRunning=lambda: True,
            requestInterruption=lambda: interrupted.append(True),
            disconnect=lambda: None,
        )
        coordinator._pitch_worker = worker
        window[:] = [8.0, 10.0]
        coordinator.request_pitch()
        window[:] = [10.0, 12.0]
        coordinator.request_pitch()
        self.assertTrue(interrupted)
        received = []
        coordinator.pitches_ready.connect(lambda *args: received.append(args))
        coordinator._finish_pitch(worker, coordinator._session_generation, [], 0, True)
        self.assertEqual(received, [([], 0, True)])
        self.assertTrue(coordinator._pitch_pending)
        self.assertTrue(coordinator._pitch_timer.isActive())

    def test_result_for_a_different_view_is_not_presented_as_current(self) -> None:
        coordinator = LearningCoordinator(lambda: (8.0, 10.0))
        self.addCleanup(coordinator.shutdown)
        coordinator.set_session(self.markers())
        received = []
        coordinator.pitches_ready.connect(lambda *args: received.append(args))
        coordinator._finish_pitch(
            SimpleNamespace(t_start=2.0, t_end=4.0), coordinator._session_generation,
            [(2.0, 3.0, 60, "ml")], 0, False,
        )
        self.assertTrue(received[0][2])
        self.assertTrue(coordinator._pitch_pending)

    def test_debounced_scroll_starts_only_the_latest_window(self) -> None:
        window = [2.0, 4.0]
        coordinator = LearningCoordinator(lambda: tuple(window))
        self.addCleanup(coordinator.shutdown)
        markers = self.markers()
        coordinator.set_session(markers)
        coordinator.request_pitch()
        window[:] = [4.0, 6.0]
        coordinator.request_pitch()
        window[:] = [8.0, 10.0]
        coordinator.request_pitch()
        with patch("ui.learning_coordinator.PitchWorker") as worker:
            coordinator._start_pitch()
            worker.assert_called_once_with(markers, 8.0, 10.0)
            worker.return_value.start.assert_called_once()

    def test_failed_old_view_does_not_lose_a_queued_new_view(self) -> None:
        coordinator = LearningCoordinator(lambda: (8.0, 10.0))
        self.addCleanup(coordinator.shutdown)
        coordinator.set_session(self.markers())
        worker = SimpleNamespace()
        coordinator._pitch_worker = worker
        coordinator._pitch_pending = True
        errors = []
        coordinator.error.connect(lambda *args: errors.append(args))
        coordinator._pitch_error(worker, coordinator._session_generation, "Inference failed")
        self.assertIsNone(coordinator._pitch_worker)
        self.assertTrue(coordinator._pitch_timer.isActive())
        self.assertEqual(errors, [("Pitch", "Inference failed")])


if __name__ == "__main__":
    unittest.main()