"""Local Whisper word recognition and known-lyrics alignment, separate from pitch."""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import version
import json
from pathlib import Path
import re
import warnings

import librosa
import numpy as np
import soundfile as sf

from audio_to_midi.evidence_cache import checksum
from core.inference_runtime import neural_device


_SECTION = re.compile(
    r"(?:intro|outro|verse|chorus|pre[- ]?chorus|post[- ]?chorus|bridge|"
    r"interlude|instrumental|solo|refrain)(?:\s+\d+)?", re.IGNORECASE,
)


def clean_lyrics(text: str) -> str:
    """Remove standalone section headings, not parenthesized sung backing vocals."""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line and not _SECTION.fullmatch(line.strip("[]: ")):
            lines.append(line)
    if not lines:
        raise ValueError("Lyrics contain no words after removing section headings.")
    return "\n".join(lines)


def word_result(result, duration: float, mode: str, expected_text: str | None = None) -> dict:
    if result is None:
        raise RuntimeError(f"Whisper {mode} failed to produce a result.")
    words = []
    previous = 0.0
    for segment in result.to_dict()["segments"]:
        for item in segment.get("words", []):
            start, end = float(item["start"]), float(item["end"])
            if (not np.isfinite([start, end]).all() or start < previous or
                    start < 0 or start > duration or end < start or end > duration + 0.02):
                raise ValueError("Whisper returned invalid or out-of-order word timestamps.")
            previous = start
            probability = item.get("probability")
            if probability is not None:
                probability = float(probability)
                if not np.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError("Whisper returned an invalid word probability.")
            words.append(dict(
                word=item["word"], start=start, end=min(end, duration),
                probability=probability, aligned=min(end, duration) > start,
            ))
    if not words:
        raise RuntimeError(f"Whisper {mode} returned no words; inspect the vocal stem.")
    messages = [
        "Singing word times and text are model estimates, not verified annotations.",
        "Word boundaries are not syllable or phoneme boundaries.",
    ]
    unaligned = sum(not word["aligned"] for word in words)
    if unaligned:
        messages.append(f"{unaligned} words have zero duration and require manual alignment.")
    if expected_text is not None:
        normalize = lambda text: re.findall(r"\w+", text.casefold())
        if normalize(expected_text) != normalize(" ".join(w["word"] for w in words)):
            messages.append("Returned words differ from the supplied lyrics; alignment may be incomplete.")
    return dict(mode=mode, text=" ".join(w["word"].strip() for w in words),
                words=words, unaligned_words=unaligned, warnings=messages)


def analyze_lyrics(
    audio_path: Path, *, lyrics_path: Path | None = None, transcribe: bool = False,
    model_name: str = "base.en", language: str = "en",
    report: Callable[[str], None] = print,
    cancelled: Callable[[], bool] = lambda: False,
) -> dict:
    """Align supplied words and/or recognize words without using those lyrics as hints."""
    from audio_to_midi.vocals import check_cancel

    check_cancel(cancelled)
    if not language.strip():
        raise ValueError("An explicit lyric language is required.")
    before = checksum(audio_path)
    lyric_hash = checksum(lyrics_path) if lyrics_path else None
    text = clean_lyrics(lyrics_path.read_text(encoding="utf-8-sig")) if lyrics_path else None
    audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Expected nonempty finite audio.")
    if not np.any(audio):
        raise ValueError("Audio is silent; no lyric analysis is possible.")
    duration = len(audio) / sr
    waveform = librosa.resample(audio.mean(axis=1), orig_sr=sr, target_sr=16000)
    import torch
    import stable_whisper

    analyses = {}
    def progress(_done: float, _total: float) -> None:
        check_cancel(cancelled)

    with neural_device(cancelled=cancelled) as device, torch.inference_mode():
        report(f"Loading Whisper {model_name} on {device.upper()} (weights download on first use)...")
        model = stable_whisper.load_model(model_name, device=device)
        check_cancel(cancelled)
        if transcribe or text is None:
            report("Recognizing lyrics without a supplied-text prompt...")
            result = model.transcribe(
                waveform, language=language, fp16=device == "cuda", verbose=None,
                condition_on_previous_text=False, word_timestamps=True, temperature=0,
                progress_callback=progress,
            )
            analyses["recognized"] = word_result(result, duration, "automatic_recognition")
            check_cancel(cancelled)
        if text is not None:
            report("Aligning supplied lyrics; this does not verify that those words were sung...")
            result = model.align(
                waveform, text, language=language, verbose=None, original_split=True,
                remove_instant_words=False, nonspeech_skip=None,
                max_word_dur=None, word_dur_factor=None,
                progress_callback=progress,
            )
            analyses["aligned"] = word_result(result, duration, "known_lyrics_alignment", text)
    check_cancel(cancelled)
    if checksum(audio_path) != before:
        raise RuntimeError("Audio changed during lyric analysis; result was not published.")
    if lyrics_path and checksum(lyrics_path) != lyric_hash:
        raise RuntimeError("Lyrics changed during analysis; result was not published.")
    for analysis in analyses.values():
        for message in analysis["warnings"]:
            warnings.warn(message, stacklevel=2)
    return dict(
        schema=1, audio=str(audio_path.resolve()), audio_sha256=before, duration=duration,
        lyrics=str(lyrics_path.resolve()) if lyrics_path else None,
        lyrics_sha256=lyric_hash,
        model=model_name, language=language,
        packages={name: version(name) for name in ("stable-ts", "openai-whisper")},
        analyses=analyses,
    )


def save_analysis(result: dict, output: Path) -> None:
    content = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(content)
