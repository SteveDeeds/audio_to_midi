import unittest
from types import SimpleNamespace

from PySide6.QtCore import QCoreApplication

from ui.learning_coordinator import LearningCoordinator


class TestLearningCoordinator(unittest.TestCase):
    _app = QCoreApplication.instance() or QCoreApplication([])

    def markers(self, pitch_pending: bool = False) -> SimpleNamespace:
        return SimpleNamespace(
            has_pending_pitch_training=lambda: pitch_pending,
            has_pending_onset_training=lambda: False,
        )

    def test_pitch_requests_coalesce_while_debouncing(self) -> None:
        coordinator = LearningCoordinator(lambda: (2.0, 4.0))
        self.addCleanup(coordinator.shutdown)
        coordinator.set_session(self.markers())

        coordinator.request_pitch()
        coordinator.request_pitch()

        self.assertTrue(coordinator._pitch_pending)
        self.assertTrue(coordinator._pitch_timer.isActive())

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


if __name__ == "__main__":
    unittest.main()