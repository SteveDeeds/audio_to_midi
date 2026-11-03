"""Reviewed per-stem rankings; vocal and harmony behavior is unchanged."""

from audio_to_midi.vocal_settings import VOCAL_MODELS

MADMOM_COMBINATIONS = ("madmom_brnn_full", "madmom_cnn_full")
MODELS = {
    "bass": ("piano_hr", "madmom_brnn_full", "madmom_cnn_full"),
    "guitar": ("basic_pitch", "yourmt3", "madmom_cnn_full"),
    "piano": ("piano_hr", "yourmt3", "basic_pitch"),
    "drums": ("adtof", "yourmt3"),
    "vocals": ("pyin", "crepe_full", "crepe_tiny", *VOCAL_MODELS),
    "other": (),
}
MODEL_LABELS = {
    "piano_hr": "Piano HR", "basic_pitch": "Basic Pitch", "yourmt3": "YourMT3+",
    "madmom_brnn_full": "madmom offline BRNN + CREPE full",
    "madmom_cnn_full": "madmom CNN + CREPE full", "adtof": "ADTOF",
}


def input_source(track) -> str:
    source = track.settings.get("inference_source", "stem")
    if source not in ("mix", "stem"):
        raise ValueError("Inference source must be mix or stem.")
    if source == "mix" and track.model != "yourmt3":
        raise ValueError("Only YourMT3+ supports mix-derived track results.")
    return source
