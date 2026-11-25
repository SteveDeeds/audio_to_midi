"""Render beat-aligned stem self-similarity without changing the application."""

from __future__ import annotations

import argparse
from fractions import Fraction
import html
import json
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from audio_to_midi.timebase import Grid
from experimental.bass_repeat_viewer import build_viewer


RATE = 22050
HOP = 256
SILENCE_DB = -60.0


def project_grid(raw: dict) -> Grid:
    return Grid(
        bpm=raw["bpm"], numerator=raw["numerator"], denominator=raw["denominator"],
        offset_seconds=raw["offset_seconds"],
        anchors=tuple(
            (seconds, Fraction(beat["numerator"], beat["denominator"]))
            for seconds, beat in raw["anchors"]
        ),
    )


def slice_edges(grid: Grid, duration: float) -> tuple[np.ndarray, int]:
    """Return complete sixteenths from the first nonnegative bar downbeat."""
    slices_per_bar = grid.bar_beats * 4
    if slices_per_bar.denominator != 1:
        raise ValueError("The meter must contain a whole number of sixteenth notes.")
    first_bar = max(0, int(np.ceil(float(grid.seconds_to_beat(0) / grid.bar_beats))))
    start = first_bar * grid.bar_beats
    count = int(np.floor(float((grid.seconds_to_beat(duration) - start) * 4)))
    if count < int(slices_per_bar) * 4:
        raise ValueError("At least four complete bars of audio are required.")
    edges = np.array([grid.beat_to_seconds(start + Fraction(i, 4)) for i in range(count + 1)])
    return edges, first_bar + 1


def pool_slices(values: np.ndarray, frame_times: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Average only the short-time frames inside each sixteenth-note interval."""
    indices = np.searchsorted(frame_times, edges, side="left")
    counts = np.diff(indices)
    if np.any(counts == 0):
        raise ValueError("Analysis hop is too large for this sixteenth-note grid.")
    sums = np.pad(np.cumsum(values, axis=1, dtype=np.float64), ((0, 0), (1, 0)))
    return ((sums[:, indices[1:]] - sums[:, indices[:-1]]) / counts).T.astype(np.float32)


def normalize_slices(features: np.ndarray, rms: np.ndarray) -> np.ndarray:
    """Cosine-normalize each slice; silent slices supply no similarity evidence."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = np.divide(features, norms, out=np.zeros_like(features), where=norms > 0)
    peak = float(np.max(rms))
    active = rms > peak * 10 ** (SILENCE_DB / 20) if peak > 0 else np.zeros(len(rms), dtype=bool)
    normalized[~active] = 0
    return normalized


def block_similarity(normalized: np.ndarray, block_size: int, *, stride: int | None = None) -> np.ndarray:
    """Mean aligned-slice cosine, not cosine of a pooled bar spectrum."""
    if block_size < 1:
        raise ValueError("Block size must be positive.")
    if stride is None:
        stride = block_size
    if stride < 1:
        raise ValueError("Stride must be positive.")
    starts = np.arange(0, len(normalized) - block_size + 1, stride)
    if not len(starts):
        raise ValueError("No complete blocks.")
    blocks = normalized[starts[:, None] + np.arange(block_size)].reshape(len(starts), -1)
    return np.clip((blocks @ blocks.T) / block_size, 0, 1)


def extract_features(path: Path, edges: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    audio, _ = librosa.load(path, sr=RATE, mono=True)
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"Nonfinite audio samples: {path}")
    mel = librosa.feature.melspectrogram(
        y=audio, sr=RATE, n_fft=2048, hop_length=HOP, n_mels=128, fmin=30, fmax=RATE / 2,
    )
    cqt = np.abs(librosa.cqt(
        audio, sr=RATE, hop_length=HOP, fmin=librosa.note_to_hz("C1"), n_bins=96,
        bins_per_octave=12, tuning=0.0,
    )) ** 2
    rms_frames = librosa.feature.rms(y=audio, frame_length=2048, hop_length=HOP)
    rms_times = librosa.frames_to_time(np.arange(rms_frames.shape[1]), sr=RATE, hop_length=HOP)
    rms = np.sqrt(pool_slices(rms_frames ** 2, rms_times, edges)[:, 0])
    features = {}
    for kind, power in (("mel", mel), ("cqt", cqt)):
        times = librosa.frames_to_time(np.arange(power.shape[1]), sr=RATE, hop_length=HOP)
        pooled = pool_slices(power, times, edges)
        peak = float(np.max(pooled))
        # A shared per-stem reference retains relative spectral energy before cosine normalization.
        features[kind] = (
            np.log1p(1000 * pooled / peak).astype(np.float32) if peak > 0 else np.zeros_like(pooled)
        )
    return features, rms


def draw_row(axes, features, matrices, first_bar, slices_per_bar, kind):
    end_bar = first_bar + len(features) / slices_per_bar
    image = axes[0].imshow(
        features.T, origin="lower", aspect="auto", interpolation="nearest", cmap="magma",
        extent=(first_bar, end_bar, 0, features.shape[1]),
    )
    axes[0].set_title(f"{kind.upper()} - sixteenth-note slices")
    axes[0].set_ylabel("CQT semitone bin (C1-B8)" if kind == "cqt" else "Mel frequency bin")
    axes[0].figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    for axis in axes:
        axis.set_xlabel("Bar (project grid)")
    for axis, (label, matrix, step) in zip(axes[1:], matrices):
        sliding = axis is axes[-1]
        start = first_bar - 0.5 if sliding else first_bar
        stop = start + len(matrix) * step / slices_per_bar
        image = axis.imshow(
            matrix, origin="lower", interpolation="nearest", aspect="equal",
            extent=(start, stop, start, stop), cmap="magma", vmin=0, vmax=1,
        )
        axis.set_title(label)
        axis.set_ylabel("Window start bar" if sliding else "Bar")
        if sliding:
            axis.set_xlabel("Window start bar")
        axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def render(stems: Path, project: Path, output: Path) -> Path:
    data = json.loads(project.read_text(encoding="utf-8"))
    raw = data["grid"]
    grid = project_grid(raw)
    paths = sorted(stems.glob("*.wav"))
    if not paths:
        raise ValueError(f"No WAV stems found in {stems}")
    durations = [sf.info(path).duration for path in paths]
    if max(durations) - min(durations) > 1 / RATE:
        raise ValueError("Stem lengths differ; align the stems before comparing them.")
    duration = min(durations)
    edges, first_bar = slice_edges(grid, duration)
    slices_per_bar = int(grid.bar_beats * 4)
    four_bar_starts = np.arange(first_bar, first_bar + (len(edges) - 1) // slices_per_bar - 3)
    output.mkdir(parents=True, exist_ok=True)
    overviews = {
        kind: plt.subplots(len(paths), 4, figsize=(24, 5 * len(paths)), squeeze=False, layout="constrained")
        for kind in ("cqt", "mel")
    }
    results = []
    for path in paths:
        print(f"Analyzing {path.name}", flush=True)
        features, rms = extract_features(path, edges)
        for kind, values in features.items():
            normalized = normalize_slices(values, rms)
            matrices = [
                ("1/16-note SSM", block_similarity(normalized, 1), 1),
                ("1-bar SSM", block_similarity(normalized, slices_per_bar), slices_per_bar),
                ("4-bar SSM (1-bar stride)",
                 block_similarity(normalized, 4 * slices_per_bar, stride=slices_per_bar), slices_per_bar),
            ]
            np.savez_compressed(
                output / f"{path.stem}-{kind}.npz", features=values, normalized=normalized,
                rms=rms, slice_edges_seconds=edges, sixteenth=matrices[0][1],
                one_bar=matrices[1][1], four_bar=matrices[2][1], four_bar_start_bars=four_bar_starts,
            )
            fig, axes = plt.subplots(1, 4, figsize=(24, 6), layout="constrained")
            draw_row(axes, values, matrices, first_bar, slices_per_bar, kind)
            fig.suptitle(f"{data['name']} / {path.stem} / {kind.upper()}")
            fig.savefig(output / f"{path.stem}-{kind}.png", dpi=170)
            plt.close(fig)
            overview, overview_axes = overviews[kind]
            row = paths.index(path)
            draw_row(overview_axes[row], values, matrices, first_bar, slices_per_bar, kind)
            overview_axes[row, 0].set_ylabel(f"{path.stem}\n{overview_axes[row, 0].get_ylabel()}")
            results.append({
                "stem": path.stem, "feature": kind,
                "shapes": {name: list(matrix.shape) for name, matrix, _ in matrices},
            })
    for kind, (fig, _) in overviews.items():
        fig.suptitle(f"{data['name']} - all stems - {kind.upper()}", fontsize=20)
        fig.savefig(output / f"overview-{kind}.png", dpi=120)
        plt.close(fig)
    metadata = {
        "project": str(project.resolve()), "stems": str(stems.resolve()), "grid": raw,
        "first_bar": first_bar, "duration_seconds": duration, "slice_count": len(edges) - 1,
        "analysis_start_seconds": float(edges[0]), "analysis_end_seconds": float(edges[-1]),
        "slices_per_bar": slices_per_bar, "sample_rate": RATE, "hop_samples": HOP,
        "four_bar_window": {"length_bars": 4, "stride_bars": 1, "start_bars": four_bar_starts.tolist()},
        "silence_db_relative_to_stem_peak": SILENCE_DB, "results": results,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    bass_link = ""
    if any(path.stem == "bass" for path in paths):
        viewer = build_viewer(output)
        bass_link = f'<a href="{viewer.name}">Bass repeat visualizer</a>'
    intro = (
        f"{grid.bpm:g} quarter-note BPM; {grid.numerator}/{grid.denominator}; "
        f"bar 1 at {grid.offset_seconds:.4f} s. {len(edges) - 1} complete sixteenth-note slices. "
        f"Analysis: {edges[0]:.4f}-{edges[-1]:.4f} s; pickup and incomplete trailing slices excluded."
    )
    cards = []
    for path in paths:
        stem = html.escape(path.stem)
        figures = "".join(
            f'<h3>{kind.upper()}</h3><a href="{stem}-{kind}.png">'
            f'<img loading="lazy" src="{stem}-{kind}.png" alt="{stem} {kind}"></a>'
            f'<p><a href="{stem}-{kind}.npz">Download exact feature and SSM arrays (.npz)</a></p>'
            for kind in ("cqt", "mel")
        )
        cards.append(f'<section id="{stem}"><h2>{stem}</h2>{figures}</section>')
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Stem self-similarity</title><style>'
        'body{background:#15151b;color:#eee;font:16px system-ui;margin:24px}'
        'a{color:#8cd4ff}img{width:100%;height:auto}section{margin-top:40px}'
        'p{max-width:1100px;line-height:1.6}nav{display:flex;gap:20px;flex-wrap:wrap}'
        '</style><h1>Stem self-similarity: ' + html.escape(data["name"]) + '</h1>'
        '<p>' + html.escape(intro) + '</p>'
        '<p>Each feature vector averages short-time spectral power within one sixteenth note, '
        'then uses log compression and per-slice cosine normalization. '
        'The 1-bar and 4-bar matrices compare downbeat-aligned windows: '
        '<strong>mean cosine similarity of corresponding sixteenth-note slices</strong>. '
        'Four-bar windows slide by one bar (bars 1-4, 2-5, 3-6, and so on). '
        'Both axes of the four-bar matrix show window start bars. '
        'Order matters; no whole-bar spectrum, chroma folding, transposition, or time warping. '
        'Only complete blocks are included at each scale. Scores use a fixed 0-1 color scale. '
        'Slices below -60 dB RMS relative to the stem peak contribute zero, even against silence; '
        'consequently silent/partly silent block diagonals can be below 1. '
        'Spectral colors are scaled separately per stem; similarity colors are shared. '
        'Click any image for its full-resolution view.</p>'
        '<nav><a href="overview-cqt.png">All stems: CQT overview</a>'
        '<a href="overview-mel.png">All stems: Mel overview</a>'
        '<a href="metadata.json">Timing and analysis metadata</a>' + bass_link + '</nav>'
        + "".join(cards) + '</html>'
    )
    index = output / "index.html"
    index.write_text(page, encoding="utf-8")
    print(f"Report: {index.resolve()}", flush=True)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stems", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("experimental") / "outputs" / "stem_similarity")
    args = parser.parse_args()
    render(args.stems, args.project, args.output)


if __name__ == "__main__":
    main()
