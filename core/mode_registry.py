"""Registered transcription modes exposed by the application UI."""

from __future__ import annotations

from dataclasses import dataclass

from core.learning_runtime import LearningDomain
from core.pitch_models import VISIBLE_PITCH_MODEL_TYPES, PitchModelPolicy, create_pitch_model


@dataclass(frozen=True)
class ModeDefinition:
    id: str
    label: str
    ui_mode: str
    domain: LearningDomain
    display_sources: tuple[str, ...]
    pitch_model_name: str | None = None
    midi_channel: int | None = None
    drum_model: str | None = None

    def create_pitch_policy(self) -> PitchModelPolicy | None:
        if self.pitch_model_name is None:
            return None
        return create_pitch_model(self.pitch_model_name)


_MODES = (
    ModeDefinition(
        id="standard",
        label="Pitched (standard)",
        ui_mode="Pitched",
        domain=LearningDomain.PITCH,
        display_sources=("db", "spectral peel", "pitch salience", "flux novelty"),
        pitch_model_name="standard",
    ),
    ModeDefinition(
        id="percussive",
        label="Percussive",
        ui_mode="Percussive",
        domain=LearningDomain.DRUM,
        display_sources=("CQT (Percussive)", "STFT", "Mel"),
        midi_channel=9,
        drum_model="nnls",
    ),
    ModeDefinition(
        id="percussive_adtof",
        label="Percussive: ADTOF",
        ui_mode="Percussive",
        domain=LearningDomain.DRUM,
        display_sources=("CQT (Percussive)", "STFT", "Mel"),
        midi_channel=9,
        drum_model="adtof",
    ),
    *(
        ModeDefinition(
            id=model.name,
            label=model.label,
            ui_mode="Pitched",
            domain=LearningDomain.PITCH,
            display_sources=("db", "spectral peel", "pitch salience", "flux novelty"),
            pitch_model_name=model.name,
        )
        for model in VISIBLE_PITCH_MODEL_TYPES
    ),
)

_MODES_BY_ID = {mode.id: mode for mode in _MODES}


def registered_modes() -> tuple[ModeDefinition, ...]:
    return _MODES


def get_mode(mode_id: str) -> ModeDefinition:
    return _MODES_BY_ID.get(str(mode_id), _MODES_BY_ID["standard"])