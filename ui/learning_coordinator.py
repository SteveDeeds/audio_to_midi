"""Qt coordinator for interactive training and visible-window inference."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QObject, QThread, QTimer, Signal

from core.audio_engine import AudioMarkers
from core.event_log import log_event
from workers.analysis_worker import OnsetWorker, PitchWorker


class LearningCoordinator(QObject):
    pitch_started = Signal()
    pitches_ready = Signal(list, int, bool)
    onset_started = Signal(int)
    onset_ready = Signal(int, bool)
    error = Signal(str, str)

    def __init__(
        self,
        time_window: Callable[[], tuple[float, float]],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._time_window = time_window
        self._markers: AudioMarkers | None = None
        self._session_generation = 0
        self._pitch_worker: PitchWorker | None = None
        self._onset_worker: OnsetWorker | None = None
        self._pitch_pending = False
        self._onset_pending = False

        self._pitch_timer = QTimer(self)
        self._pitch_timer.setSingleShot(True)
        self._pitch_timer.setInterval(200)
        self._pitch_timer.timeout.connect(self._start_pitch)

        self._onset_timer = QTimer(self)
        self._onset_timer.setSingleShot(True)
        self._onset_timer.setInterval(100)
        self._onset_timer.timeout.connect(self._start_onset)

    def set_session(self, markers: AudioMarkers) -> None:
        self.clear_session()
        self._markers = markers

    def clear_session(self) -> None:
        self._session_generation += 1
        self._markers = None
        self._pitch_pending = False
        self._onset_pending = False
        self._pitch_timer.stop()
        self._onset_timer.stop()
        self._stop_worker(self._pitch_worker)
        self._stop_worker(self._onset_worker)
        self._pitch_worker = None
        self._onset_worker = None

    def request_pitch(self) -> None:
        if self._markers is None:
            return
        self._pitch_pending = True
        self._pitch_timer.start()

    def request_onset(self) -> None:
        if self._markers is None:
            return
        self._onset_pending = True
        log_event(
            "onset_training_queued",
            pending_examples=self._markers.pending_onset_training_count(),
            pitch_running=self._worker_running(self._pitch_worker),
        )
        self._onset_timer.start()

    def shutdown(self) -> None:
        self.clear_session()

    def _start_pitch(self) -> None:
        markers = self._markers
        if markers is None or not self._pitch_pending:
            return
        if self._onset_pending and markers.has_pending_onset_training():
            self._onset_timer.start()
            return
        if self._worker_running(self._onset_worker):
            self._pitch_timer.start()
            return
        if self._worker_running(self._pitch_worker):
            return

        self._pitch_pending = False
        t_start, t_end = self._time_window()
        generation = self._session_generation
        worker = PitchWorker(markers, t_start, t_end)
        self._pitch_worker = worker
        self.pitch_started.emit()
        worker.finished_ok.connect(
            lambda results, edit_generation, stale: self._finish_pitch(
                worker, generation, results, edit_generation, stale
            )
        )
        worker.error.connect(
            lambda message: self._worker_error("Pitch", generation, message)
        )
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _finish_pitch(
        self,
        worker: PitchWorker,
        session_generation: int,
        results: list,
        edit_generation: int,
        stale: bool,
    ) -> None:
        if self._pitch_worker is worker:
            self._pitch_worker = None
        markers = self._markers
        if session_generation != self._session_generation or markers is None:
            return
        pending = bool(
            stale or self._pitch_pending or markers.has_pending_pitch_training()
        )
        self.pitches_ready.emit(results, edit_generation, pending)
        if self._onset_pending:
            self._onset_timer.start()
        elif pending:
            self.request_pitch()

    def _start_onset(self) -> None:
        markers = self._markers
        if markers is None or not self._onset_pending:
            return
        if self._worker_running(self._pitch_worker):
            self._onset_timer.start()
            return
        if self._worker_running(self._onset_worker):
            return
        if not markers.has_pending_onset_training():
            self._onset_pending = False
            return

        self._onset_pending = False
        generation = self._session_generation
        pending_examples = markers.pending_onset_training_count()
        worker = OnsetWorker(markers)
        self._onset_worker = worker
        self.onset_started.emit(pending_examples)
        worker.finished_ok.connect(
            lambda edit_generation, stale: self._finish_onset(
                worker, generation, edit_generation, stale
            )
        )
        worker.error.connect(
            lambda message: self._worker_error("Onset", generation, message)
        )
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _finish_onset(
        self,
        worker: OnsetWorker,
        session_generation: int,
        edit_generation: int,
        stale: bool,
    ) -> None:
        if self._onset_worker is worker:
            self._onset_worker = None
        markers = self._markers
        if session_generation != self._session_generation or markers is None:
            return
        pending = bool(
            stale or self._onset_pending or markers.has_pending_onset_training()
        )
        self.onset_ready.emit(edit_generation, pending)
        if pending:
            self.request_onset()
        elif self._pitch_pending:
            self._pitch_timer.start()

    def _worker_error(
        self,
        domain: str,
        session_generation: int,
        message: str,
    ) -> None:
        if session_generation == self._session_generation:
            self.error.emit(domain, message)

    @staticmethod
    def _worker_running(worker: QThread | None) -> bool:
        if worker is None:
            return False
        try:
            return worker.isRunning()
        except RuntimeError:
            return False

    @staticmethod
    def _stop_worker(worker: QThread | None, timeout_ms: int = 2000) -> None:
        if worker is None:
            return
        try:
            if LearningCoordinator._worker_running(worker):
                worker.requestInterruption()
                worker.quit()
                worker.wait(timeout_ms)
            worker.disconnect()
            worker.deleteLater()
        except Exception:
            pass