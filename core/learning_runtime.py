"""Pure scheduling contracts for background training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from core.chord_change_model import ChordChangeModel
from core.models import CorrectionModel
from core.pitch_models import PitchModelPolicy


class LearningDomain(str, Enum):
    PITCH = "pitch"
    ONSET = "onset"
    DRUM = "drum"
    CHORD = "chord"


class InferencePriority(str, Enum):
    VISIBLE = "visible"
    EXPORT = "export"
    OFFSCREEN = "offscreen"


@dataclass(frozen=True)
class RuntimeToken:
    session: int
    model: int
    edits: int
    settings: int


@dataclass(frozen=True)
class ModelSnapshot:
    """A published model bundle that inference treats as read-only."""

    version: int
    correction: CorrectionModel
    pitch_policy: PitchModelPolicy
    chord: ChordChangeModel

    def fork_for_training(self) -> "ModelSnapshot":
        return ModelSnapshot(
            version=self.version,
            correction=self.correction.clone(),
            pitch_policy=self.pitch_policy.clone(),
            chord=self.chord.clone(),
        )

    def with_version(self, version: int) -> "ModelSnapshot":
        return ModelSnapshot(
            version=int(version),
            correction=self.correction,
            pitch_policy=self.pitch_policy,
            chord=self.chord,
        )


@dataclass(frozen=True)
class TrainingLabel:
    domain: LearningDomain
    key: Any
    value: Any


@dataclass(frozen=True)
class TrainingBatch:
    batch_id: int
    base_model_version: int
    labels: tuple[TrainingLabel, ...]


@dataclass(frozen=True)
class InferenceRequest:
    request_id: int
    priority: InferencePriority
    token: RuntimeToken
    t_start: float = 0.0
    t_end: float = 0.0


class LearningRuntimeState:
    """Coordinate one trainer lane and one inference lane.

    The class owns no threads. A Qt coordinator asks it for the next work item
    whenever a worker lane becomes idle, then reports completion explicitly.
    """

    def __init__(self) -> None:
        self._pending_labels: list[TrainingLabel] = []
        self._training_active = False
        self._next_batch_id = 1
        self._latest_visible: InferenceRequest | None = None
        self._export_request: InferenceRequest | None = None
        self._offscreen_request: InferenceRequest | None = None
        self._active_inference: InferenceRequest | None = None
        self._awaiting_visible_model_version: int | None = None
        self._training_failed = False

    @property
    def pending_label_count(self) -> int:
        return len(self._pending_labels)

    @property
    def training_active(self) -> bool:
        return self._training_active

    def submit_labels(self, labels: Iterable[TrainingLabel]) -> None:
        self._pending_labels.extend(labels)

    def request_inference(self, request: InferenceRequest) -> None:
        if request.priority is InferencePriority.VISIBLE:
            self._latest_visible = request
        elif request.priority is InferencePriority.EXPORT:
            self._export_request = request
        else:
            self._offscreen_request = request

    def take_training_batch(self, base_model_version: int) -> TrainingBatch | None:
        if (
            self._training_active
            or self._training_failed
            or self._awaiting_visible_model_version is not None
            or self._export_request is not None
            or (
                self._active_inference is not None
                and self._active_inference.priority is InferencePriority.EXPORT
            )
            or not self._pending_labels
        ):
            return None

        batch = TrainingBatch(
            batch_id=self._next_batch_id,
            base_model_version=int(base_model_version),
            labels=tuple(self._pending_labels),
        )
        self._next_batch_id += 1
        self._pending_labels.clear()
        self._training_active = True
        return batch

    def finish_training(self, published_model_version: int) -> None:
        if not self._training_active:
            raise RuntimeError("No training batch is active")
        self._training_active = False
        self._awaiting_visible_model_version = int(published_model_version)

    def fail_training(self, batch: TrainingBatch) -> None:
        if not self._training_active:
            raise RuntimeError("No training batch is active")
        self._training_active = False
        self._pending_labels[0:0] = batch.labels
        self._training_failed = True

    def retry_failed_training(self) -> None:
        self._training_failed = False

    def take_inference_request(self) -> InferenceRequest | None:
        if self._active_inference is not None:
            return None
        if self._latest_visible is not None:
            request = self._latest_visible
            self._latest_visible = None
        elif self._export_request is not None:
            request = self._export_request
            self._export_request = None
        elif self._offscreen_request is not None:
            request = self._offscreen_request
            self._offscreen_request = None
        else:
            return None
        self._active_inference = request
        return request

    def accept_inference(self, request: InferenceRequest) -> None:
        if self._active_inference != request:
            raise RuntimeError("Inference request is not active")
        self._active_inference = None
        awaited = self._awaiting_visible_model_version
        if (
            request.priority is InferencePriority.VISIBLE
            and awaited is not None
            and request.token.model == awaited
        ):
            self._awaiting_visible_model_version = None
