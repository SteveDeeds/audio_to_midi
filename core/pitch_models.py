"""Selectable pitch-model policies applied after shared NNLS analysis."""

from __future__ import annotations

import copy
from collections import Counter


class PitchModelPolicy:
    name = "standard"
    label = "Standard"
    max_polyphony: int | None = None
    uses_pitch_gate = True

    def clone(self) -> "PitchModelPolicy":
        return copy.deepcopy(self)

    def select_nnls_notes(self, notes: list[tuple[int, float]]) -> list[tuple[int, float]]:
        return notes

    def sounding_midi(self, displayed_midi: int) -> int:
        return int(displayed_midi)

    def observe_rejection(self, block_key: tuple[int, int], midi: int) -> None:
        pass

    def observe_confirmation(self, block_key: tuple[int, int], midi: int) -> None:
        pass

    def to_dict(self) -> dict:
        return {"name": self.name}

    def load_dict(self, data: dict) -> None:
        pass


class BassPitchModel(PitchModelPolicy):
    """Monophonic bass policy mapping sounding NNLS pitch to written pitch."""

    name = "bass"
    label = "Bass (written pitch)"
    max_polyphony = 1
    uses_pitch_gate = False
    DEFAULT_OFFSET = 12
    MAX_OFFSET_SAMPLES = 64

    def __init__(self) -> None:
        self._octave_offsets: list[int] = []
        self._pending_rejections: dict[tuple[int, int], int] = {}

    @property
    def octave_offset(self) -> int:
        if not self._octave_offsets:
            return self.DEFAULT_OFFSET
        counts = Counter(self._octave_offsets)
        return int(max(counts, key=lambda offset: (counts[offset], -abs(offset))))

    def select_nnls_notes(self, notes: list[tuple[int, float]]) -> list[tuple[int, float]]:
        if not notes:
            return []
        midi, score = notes[0]
        corrected = max(0, min(127, int(midi) + self.octave_offset))
        return [(corrected, float(score))]

    def sounding_midi(self, displayed_midi: int) -> int:
        return max(0, min(127, int(displayed_midi) - self.octave_offset))

    def observe_rejection(self, block_key: tuple[int, int], midi: int) -> None:
        self._pending_rejections[block_key] = int(midi)

    def observe_confirmation(self, block_key: tuple[int, int], midi: int) -> None:
        rejected = self._pending_rejections.pop(block_key, None)
        if rejected is None:
            return
        corrected_delta = int(midi) - rejected
        octave_offset = self.octave_offset + int(round(corrected_delta / 12.0)) * 12
        if octave_offset < -24 or octave_offset > 24:
            return
        self._octave_offsets.append(octave_offset)
        self._octave_offsets = self._octave_offsets[-self.MAX_OFFSET_SAMPLES:]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "octave_offsets": list(self._octave_offsets),
        }

    def load_dict(self, data: dict) -> None:
        offsets = data.get("octave_offsets", [])
        self._octave_offsets = [
            int(offset) for offset in offsets
            if isinstance(offset, (int, float)) and -24 <= int(offset) <= 24
        ][-self.MAX_OFFSET_SAMPLES:]


PITCH_MODEL_TYPES = {
    PitchModelPolicy.name: PitchModelPolicy,
    BassPitchModel.name: BassPitchModel,
}


def create_pitch_model(name: str, data: dict | None = None) -> PitchModelPolicy:
    model_type = PITCH_MODEL_TYPES.get(str(name), PitchModelPolicy)
    model = model_type()
    if data:
        model.load_dict(data)
    return model