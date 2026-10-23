"""BTC chord recognizer (Park et al., "A Bi-Directional Transformer for Musical Chord
Recognition", ISMIR 2019), inference only, with its published large-vocabulary weights.

The network below re-implements https://github.com/jayg996/BTC-ISMIR19 (MIT License,
Copyright (c) 2019 Jonggwon Park) with the original module names so the checkpoint loads
unchanged; training code, dropout and attention-map outputs are omitted. Features match
the original: log-magnitude CQT (144 bins, 24 per octave from C1, hop 2048 at 22050 Hz)
computed in independent 10-second chunks, standardized with the checkpoint's mean/std and
classified in non-overlapping 108-frame windows.

Original BTC code license:

    MIT License

    Copyright (c) 2019 Jonggwon Park

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import math

import numpy as np
import torch
from torch import nn

from audio_to_midi.btc_paths import CHECKPOINT, CHECKPOINT_BYTES

SAMPLE_RATE = 22050
CHUNK_SECONDS = 10.0
HOP_LENGTH = 2048
N_BINS = 144
BINS_PER_OCTAVE = 24
TIMESTEP = 108

_ROOTS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_QUALITIES = ("min", "maj", "dim", "aug", "min6", "maj6", "min7", "minmaj7", "maj7", "7", "dim7", "hdim7", "sus2", "sus4")
# BTC quality -> this app's vocabulary (sixth chords fold onto their triads).
_TO_APP = {
    "min": "m", "maj": "", "dim": "dim", "aug": "aug", "min6": "m", "maj6": "", "min7": "m7",
    "minmaj7": "mMaj7", "maj7": "maj7", "7": "7", "dim7": "dim7", "hdim7": "m7b5", "sus2": "sus2", "sus4": "sus4",
}
UNKNOWN, NO_CHORD_INDEX = 168, 169


def labels() -> list[str | None]:
    """App chord symbol per BTC class; None for BTC's 'X' (unknown chord)."""
    from audio_to_midi.chords import NO_CHORD, symbol

    out: list[str | None] = [symbol(i // 14, _TO_APP[_QUALITIES[i % 14]]) for i in range(168)]
    return out + [None, NO_CHORD]


def checkpoint_ready() -> bool:
    return CHECKPOINT.is_file() and CHECKPOINT.stat().st_size == CHECKPOINT_BYTES


# -- network (names mirror the original so state_dict keys match) -------------------------
def _bias_mask(length: int) -> torch.Tensor:
    return torch.triu(torch.full((length, length), float("-inf")), 1)[None, None]


def _timing_signal(length: int, channels: int) -> torch.Tensor:
    timescales = channels // 2
    increment = math.log(1.0e4) / (timescales - 1)
    inverse = np.exp(np.arange(timescales, dtype=np.float64) * -increment)
    scaled = np.arange(length)[:, None] * inverse[None, :]
    signal = np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1)
    return torch.from_numpy(signal.reshape(1, length, channels)).float()


class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        return self.gamma * (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + self.eps) + self.beta


class MultiHeadAttention(nn.Module):
    def __init__(self, size: int, heads: int, mask: torch.Tensor):
        super().__init__()
        self.heads, self.scale, self.mask = heads, (size // heads) ** -0.5, mask
        self.query_linear = nn.Linear(size, size, bias=False)
        self.key_linear = nn.Linear(size, size, bias=False)
        self.value_linear = nn.Linear(size, size, bias=False)
        self.output_linear = nn.Linear(size, size, bias=False)

    def _split(self, x):
        b, t, d = x.shape
        return x.view(b, t, self.heads, d // self.heads).permute(0, 2, 1, 3)

    def forward(self, x):
        q = self._split(self.query_linear(x)) * self.scale
        k, v = self._split(self.key_linear(x)), self._split(self.value_linear(x))
        logits = q @ k.transpose(-1, -2) + self.mask[:, :, :x.shape[1], :x.shape[1]]
        context = torch.softmax(logits, dim=-1) @ v
        b, h, t, d = context.shape
        return self.output_linear(context.permute(0, 2, 1, 3).reshape(b, t, h * d))


class Conv(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.pad = nn.ConstantPad1d((1, 1), 0)
        self.conv = nn.Conv1d(size, size, kernel_size=3)

    def forward(self, x):
        return self.conv(self.pad(x.permute(0, 2, 1))).permute(0, 2, 1)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.layers = nn.ModuleList([Conv(size), Conv(size)])

    def forward(self, x):
        for layer in self.layers:  # the original applies ReLU after both layers
            x = torch.relu(layer(x))
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self, size: int, heads: int, mask: torch.Tensor):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(size, heads, mask)
        self.positionwise_convolution = PositionwiseFeedForward(size)
        self.layer_norm_mha = LayerNorm(size)
        self.layer_norm_ffn = LayerNorm(size)

    def forward(self, x):
        x = x + self.multi_head_attention(self.layer_norm_mha(x))
        return x + self.positionwise_convolution(self.layer_norm_ffn(x))


class BiDirectionalSelfAttention(nn.Module):
    def __init__(self, size: int, heads: int, length: int):
        super().__init__()
        mask = _bias_mask(length)
        self.attn_block = SelfAttentionBlock(size, heads, mask)
        self.backward_attn_block = SelfAttentionBlock(size, heads, mask.transpose(2, 3))
        self.linear = nn.Linear(size * 2, size)

    def forward(self, x):
        return self.linear(torch.cat((self.attn_block(x), self.backward_attn_block(x)), dim=2))


class BiDirectionalSelfAttentionLayers(nn.Module):
    def __init__(self, features: int, size: int, layers: int, heads: int, length: int):
        super().__init__()
        self.timing_signal = _timing_signal(length, size)
        self.embedding_proj = nn.Linear(features, size, bias=False)
        self.self_attn_layers = nn.Sequential(*[BiDirectionalSelfAttention(size, heads, length) for _ in range(layers)])
        self.layer_norm = LayerNorm(size)

    def forward(self, x):
        x = self.embedding_proj(x) + self.timing_signal[:, :x.shape[1]]
        return self.layer_norm(self.self_attn_layers(x))


class OutputLayer(nn.Module):
    def __init__(self, size: int, classes: int):
        super().__init__()
        self.output_projection = nn.Linear(size, classes)
        self.lstm = nn.LSTM(size, size // 2, batch_first=True, bidirectional=True)  # unused; in the checkpoint

    def forward(self, x):
        return self.output_projection(x)


class BTC(nn.Module):
    def __init__(self, classes: int = 170):
        super().__init__()
        self.self_attn_layers = BiDirectionalSelfAttentionLayers(N_BINS, 128, 8, 4, TIMESTEP)
        self.output_layer = OutputLayer(128, classes)

    def forward(self, x):
        return self.output_layer(self.self_attn_layers(x))


@lru_cache(maxsize=1)
def _load(path: str, mtime: int):
    allowed = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype, type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = BTC()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, float(checkpoint["mean"]), float(checkpoint["std"])


def model():
    if not checkpoint_ready():
        raise RuntimeError("The BTC chord model is not downloaded yet.")
    return _load(str(CHECKPOINT), CHECKPOINT.stat().st_mtime_ns)


# -- inference ----------------------------------------------------------------------------
def features(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(log-CQT frames x 144, frame times in seconds) for 22050 Hz mono audio."""
    import librosa

    chunk = int(SAMPLE_RATE * CHUNK_SECONDS)
    parts, times = [], []
    for start in range(0, max(1, len(audio)), chunk):
        piece = audio[start:start + chunk]
        if len(piece) == 0:
            break
        cqt = np.abs(librosa.cqt(
            piece, sr=SAMPLE_RATE, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, hop_length=HOP_LENGTH,
        ))
        parts.append(np.log(cqt + 1e-6).T)
        times.append(start / SAMPLE_RATE + np.arange(cqt.shape[1]) * HOP_LENGTH / SAMPLE_RATE)
    if not parts:
        return np.zeros((0, N_BINS), dtype=np.float32), np.zeros(0)
    return np.concatenate(parts).astype(np.float32), np.concatenate(times)


def probabilities(audio: np.ndarray, cancel=lambda: False) -> tuple[np.ndarray, np.ndarray]:
    """(per-frame class probabilities frames x 170, frame times) for 22050 Hz mono audio."""
    from audio_to_midi.adapters import Cancelled

    net, mean, std = model()
    feature, times = features(np.asarray(audio, dtype=np.float32))
    count = len(feature)
    if count == 0:
        return np.zeros((0, 170), dtype=np.float32), times
    padded = np.pad((feature - mean) / std, ((0, -count % TIMESTEP), (0, 0)))
    windows = torch.from_numpy(padded.reshape(-1, TIMESTEP, N_BINS))
    out = []
    with torch.inference_mode():
        for start in range(0, len(windows), 16):
            if cancel():
                raise Cancelled()
            out.append(torch.softmax(net(windows[start:start + 16]), dim=-1).reshape(-1, 170).numpy())
    return np.concatenate(out)[:count], times


def recognize(
    audio: np.ndarray, grid, duration: float, bass_cqt: np.ndarray | None = None,
    bass_low_pitch: int = 21, bass_frame_seconds: float = 0.0, cancel=lambda: False,
):
    """Beat-level chord segments from BTC frame probabilities.

    Probabilities are averaged over each beat and summed per app symbol (BTC's
    sixth chords fold onto triads; 'X' is ignored). BTC names no inversions, so
    a separated bass stem adds a slash only when its note is another chord tone.
    """
    from audio_to_midi.adapters import Cancelled
    from audio_to_midi.chords import NO_CHORD, _bass, harmonic_power, parse, pitch_classes, symbol
    from audio_to_midi.domain import ChordSegment

    probs, times = probabilities(audio, cancel)
    names = labels()
    order = sorted({n for n in names if n is not None}, key=lambda n: names.index(n))
    column = {n: i for i, n in enumerate(order)}
    fold = np.zeros((len(names), len(order)), dtype=np.float32)
    for index, name in enumerate(names):
        if name is not None:
            fold[index, column[name]] = 1
    grouped = probs @ fold if len(probs) else np.zeros((0, len(order)), dtype=np.float32)
    bass_power = harmonic_power(bass_cqt) if bass_cqt is not None else None

    beat_q = Fraction(4, grid.denominator)
    end = grid.seconds_to_beat(duration)
    start = Fraction(0) if grid.offset_seconds >= 0 else grid.seconds_to_beat(0)
    beat = start - start % beat_q
    segments = []
    while beat < end:
        if cancel():
            raise Cancelled()
        stop = min(end, beat + beat_q)
        t0, t1 = grid.beat_to_seconds(beat), grid.beat_to_seconds(stop)
        name, confidence, alternatives = NO_CHORD, 0.0, ()
        inside = np.flatnonzero((times >= t0) & (times < t1))
        if not len(inside) and len(times):
            inside = np.array([int(np.abs(times - (t0 + t1) / 2).argmin())])
        if len(inside):
            vector = grouped[inside].mean(axis=0)
            vector = vector / max(float(vector.sum()), 1e-9)
            ranked = np.argsort(-vector)
            name, confidence = order[ranked[0]], float(vector[ranked[0]])
            alternatives = tuple(order[i] for i in ranked[1:6])
            if bass_power is not None and name != NO_CHORD and bass_frame_seconds > 0:
                lo = max(0, int(t0 / bass_frame_seconds))
                hi = min(bass_power.shape[1], int(t1 / bass_frame_seconds) + 1)
                bass = _bass(bass_power[:, lo:hi], bass_low_pitch) if hi > lo else None
                root, quality, _ = parse(name)
                if bass is not None and bass != root and bass in pitch_classes(name):
                    name = symbol(root, quality, bass)
        if segments and segments[-1].symbol == name and segments[-1].end == beat:
            last = segments[-1]
            segments[-1] = ChordSegment(last.start, stop, name, "detected", min(last.confidence, confidence), last.alternatives)
        else:
            segments.append(ChordSegment(beat, stop, name, "detected", max(0.0, min(1.0, confidence)), alternatives))
        beat = stop
    return segments
