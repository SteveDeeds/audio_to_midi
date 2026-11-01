"""Small persisted parameter surface for the neural vocal models."""

from dataclasses import dataclass, fields
import math


VOCAL_MODELS = ("madmom_cnn", "madmom_brnn")
MODEL_DESCRIPTIONS = {
    "madmom_cnn": "MADMoM CNN syllable attacks + CREPE full pitch. Visible bars only; non-commercial onset weights.",
    "madmom_brnn": "MADMoM bidirectional RNN syllable attacks + CREPE full pitch. Visible bars only; non-commercial onset weights.",
}
CONTROLS = (
    ("neural_threshold", "Onset", 0.01, 0.99, 0.01, 2, "",
     "Neural onset threshold (start 0.30). Lower finds weaker syllable attacks; higher rejects weak proposals."),
    ("voice_db", "Voice", -60.0, -10.0, 1.0, 0, " dB",
     "Voice energy gate (start -35 dB). Exclude quiet note regions relative to the visible audio's RMS peak."),
    ("onset_wait", "Spacing", 0.02, 0.4, 0.01, 2, " s",
     "Minimum syllable spacing (start 0.12 s; try 0.10-0.16). Raise to remove close attack duplicates. The grid can merge nearby attacks."),
    ("pitch_hold", "Slur hold", 0.03, 0.4, 0.01, 2, " s",
     "Pitch-change hold (start 0.09 s; try 0.06-0.12). Lower accepts shorter slur notes; higher suppresses brief vibrato/slide excursions."),
    ("min_note", "Min split", 0.04, 0.4, 0.01, 2, " s",
     "Minimum split interval before the half-grid articulation gap (start 0.08 s; try 0.05-0.10). Lower accepts shorter proposed notes, not invented syllables."),
)


@dataclass(frozen=True)
class VocalParameters:
    neural_threshold: float = 0.30
    voice_db: float = -35.0
    onset_wait: float = 0.12
    pitch_hold: float = 0.09
    min_note: float = 0.08

    @classmethod
    def from_dict(cls, values: object) -> "VocalParameters":
        if not isinstance(values, dict):
            raise ValueError("Vocal parameters must be an object.")
        if any(not isinstance(k, str) for k in values):
            raise ValueError("Vocal parameter names must be strings.")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown vocal parameters: {', '.join(sorted(unknown))}")
        result = cls(**values)
        for name, _, low, high, *_ in CONTROLS:
            value = getattr(result, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric.")
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be between {low:g} and {high:g}.")
        return result
