"""Beat candidates, user-pinned labeling, and tempo fitting for the mix.

Times are source seconds in the original mix; positions are quarter notes
relative to bar 1's downbeat (quarter 0), matching ``Grid.anchors``.
The tracker proposes pulses only; bar positions come from user pins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import math

import numpy as np


ANALYSIS_RATE = 22050
HOP = 256
FINE_HOP = 64
OFF_GRID = 0.2  # max distance, in pulse units, from the expected grid for an unpinned beat to be labeled
MIN_CONFIDENCE = 0.2  # weaker unpinned candidates are shown but not counted automatically
LOCK_BEATS = 16  # labeled beats used to predict the next one when counting past the pins
MAX_DRIFT = 0.1  # local tempo may deviate this much from the pinned tempo while counting outward
SMOOTH_QUARTERS = 8.0  # half-width of the local tempo fit used for the smoothed curve
OUTLIER_SECONDS = 0.06
OUTLIER_FRACTION = 0.15  # of one pulse
STEADY_DRIFT_MS = 30.0  # below this, a constant tempo fits and no tempo map is needed


@dataclass
class Beat:
    time: float
    confidence: float
    pin: Fraction | None = None  # user-confirmed quarter position
    mark: tuple[int, int] | None = None  # user's (bar, beat in bar) for ``pin``; survives meter edits


@dataclass
class Detection:
    beats: list[Beat]
    candidate_bpm: float
    duration: float
    envelope: np.ndarray  # peak |amplitude| per ENVELOPE_STEP seconds, 0..1
    cqt: np.ndarray  # semitone rows (low first) x frames, 0..1, HOP/ANALYSIS_RATE s per frame
    frame_seconds: float = HOP / ANALYSIS_RATE
    envelope_step: float = 0.01


@dataclass
class Fit:
    bpm: float
    downbeat: float  # source seconds of quarter 0 (pinned, else on the smoothed curve)
    max_error_ms: float  # labeled beats vs. one constant tempo, including tracker jitter
    rms_error_ms: float
    drift_ms: float  # smoothed tempo curve vs. constant tempo: real drift, jitter removed
    drift_at: float = 0.0  # source seconds where drift is largest
    slope: float = field(repr=False, default=0.0)
    intercept: float = field(repr=False, default=0.0)

    @property
    def steady(self) -> bool:
        return self.drift_ms < STEADY_DRIFT_MS

    def time_of(self, quarter: float) -> float:
        return self.intercept + self.slope * quarter


def detect(audio: np.ndarray, sample_rate: int, cancel=lambda: False) -> Detection:
    """Propose beats with 0..1 confidence; also return display envelope and CQT."""
    import librosa
    from audio_to_midi.adapters import Cancelled
    from audio_to_midi.spectrogram import cqt_image

    mono = np.asarray(audio, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    if not len(mono) or not np.all(np.isfinite(mono)):
        raise ValueError("The mix has no usable audio.")
    duration = len(mono) / sample_rate
    step = max(1, round(0.01 * sample_rate))
    padded = np.pad(np.abs(mono), (0, (-len(mono)) % step))
    envelope = padded.reshape(-1, step).max(axis=1)
    envelope = envelope / max(float(envelope.max()), 1e-9)
    y = librosa.resample(mono, orig_sr=sample_rate, target_sr=ANALYSIS_RATE) if sample_rate != ANALYSIS_RATE else mono
    if cancel():
        raise Cancelled()
    strength = librosa.onset.onset_strength(y=y, sr=ANALYSIS_RATE, hop_length=HOP, aggregate=np.median)
    tempo, frames = librosa.beat.beat_track(
        onset_envelope=strength, sr=ANALYSIS_RATE, hop_length=HOP, units="frames", trim=False,
    )
    if cancel():
        raise Cancelled()
    pulse = librosa.beat.plp(onset_envelope=strength, sr=ANALYSIS_RATE, hop_length=HOP)
    # Coarse flux peaks ~13 ms late; a short-window pass places attacks within a few ms.
    fine = librosa.onset.onset_strength(y=y, sr=ANALYSIS_RATE, hop_length=FINE_HOP, n_fft=512)
    beats = []
    scale = max(float(np.percentile(strength, 95)), 1e-9)
    pulse_scale = max(float(pulse.max()), 1e-9)
    radius = max(1, round(0.04 * ANALYSIS_RATE / FINE_HOP))
    for frame in np.asarray(frames, dtype=int):
        center = round(frame * HOP / FINE_HOP)
        lo, hi = max(0, center - radius), min(len(fine), center + radius + 1)
        if hi <= lo:
            continue
        peak = lo + int(np.argmax(fine[lo:hi]))
        offset = 0.0
        if 0 < peak < len(fine) - 1:
            # Parabolic interpolation for sub-frame placement.
            a, b, c = fine[peak - 1:peak + 2]
            denominator = a - 2 * b + c
            if denominator < 0:
                offset = float(np.clip(0.5 * (a - c) / denominator, -0.5, 0.5))
        time = (peak + offset) * FINE_HOP / ANALYSIS_RATE
        coarse = min(len(strength) - 1, frame)
        onset = min(1.0, strength[coarse] / scale)
        # Periodicity alone (e.g. extrapolated into silence) is not evidence of a beat.
        confidence = 0.6 * onset + 0.4 * pulse[coarse] / pulse_scale * min(1.0, onset / 0.1)
        beats.append(Beat(float(time), float(np.clip(confidence, 0, 1))))
    beats = dedupe(beats)
    if cancel():
        raise Cancelled()
    cqt = cqt_image(y, ANALYSIS_RATE)
    return Detection(beats, float(np.asarray(tempo).reshape(-1)[0]), duration, envelope.astype(np.float32), cqt)


def dedupe(beats: list[Beat], min_gap: float = 0.05) -> list[Beat]:
    result: list[Beat] = []
    for beat in sorted(beats, key=lambda b: b.time):
        if result and beat.time - result[-1].time < min_gap:
            if beat.pin is not None or (result[-1].pin is None and beat.confidence > result[-1].confidence):
                result[-1] = beat
            continue
        result.append(beat)
    return result


def label(beats: list[Beat], unit: Fraction) -> list[Fraction | None]:
    """Assign quarter positions to beats from the user's pins.

    ``unit`` is the pulse length in quarters (normally one meter beat). Beats
    between two pins are spaced by interpolation and rounded to the unit; beats
    beyond the outermost pins are counted outward, following the local period.
    Unlabeled (None) beats are off-grid or duplicates and are not tempo-map anchors.
    """
    unit = Fraction(unit)
    if unit <= 0:
        raise ValueError("Pulse unit must be positive.")
    labels: list[Fraction | None] = [b.pin for b in beats]
    pins = [i for i, b in enumerate(beats) if b.pin is not None]
    if not pins:
        return labels
    for a, b in zip(pins, pins[1:]):
        if beats[b].time <= beats[a].time or beats[b].pin <= beats[a].pin:
            raise ValueError("Pinned positions must increase with time.")
    for a, b in zip(pins, pins[1:]):
        ta, tb, qa, qb = beats[a].time, beats[b].time, beats[a].pin, beats[b].pin
        best: dict[Fraction, tuple[float, int]] = {}
        for i in range(a + 1, b):
            if beats[i].confidence < MIN_CONFIDENCE:
                continue
            ideal = float(qa) + (beats[i].time - ta) / (tb - ta) * float(qb - qa)
            steps = round((ideal - float(qa)) / float(unit))
            q = qa + steps * unit
            error = abs(ideal - float(q)) / float(unit)
            if qa < q < qb and error <= OFF_GRID and (q not in best or error < best[q][0]):
                best[q] = (error, i)
        for q, (_, i) in best.items():
            labels[i] = q
    period = _period(beats, pins, unit)
    _extend(beats, labels, pins[-1], 1, unit, period)
    _extend(beats, labels, pins[0], -1, unit, period)
    _drop_outliers(beats, labels, unit)
    return labels


def _period(beats: list[Beat], pins: list[int], unit: Fraction) -> float:
    if len(pins) >= 2:
        first, last = pins[0], pins[-1]
        return (beats[last].time - beats[first].time) / float((beats[last].pin - beats[first].pin) / unit)
    gaps = np.diff([b.time for b in beats])
    if not len(gaps):
        raise ValueError("Pin a second beat so the tempo can be measured.")
    return float(np.median(gaps))


def _extend(beats, labels, start, direction, unit, period) -> None:
    """Count outward from the outermost pin, phase-locked to a local fit of recent labeled beats."""
    recent = [(beats[i].time, float(labels[i])) for i in range(len(beats)) if labels[i] is not None]
    recent.sort(reverse=direction < 0)
    recent = recent[-LOCK_BEATS:]
    frontier = labels[start]
    nominal = period / float(unit)
    i = start + direction
    while 0 <= i < len(beats):
        beat = beats[i]
        if beat.pin is not None:
            break
        i += direction
        if beat.confidence < MIN_CONFIDENCE:
            continue
        if len(recent) >= 2:
            quarters = np.array([q for _, q in recent])
            times = np.array([t for t, _ in recent])
            slope = float(np.clip(
                np.polyfit(quarters, times, 1)[0], nominal * (1 - MAX_DRIFT), nominal * (1 + MAX_DRIFT),
            ))
            intercept = float(np.mean(times - slope * quarters))
        else:
            slope = nominal
            intercept = recent[0][0] - slope * recent[0][1]
        estimate = (beat.time - intercept) / slope
        steps = round((estimate - float(frontier)) / float(unit))
        quarter = frontier + steps * unit
        error = abs(estimate - float(quarter)) / float(unit)
        if steps * direction < 1 or error > OFF_GRID:
            continue
        labels[i - direction] = quarter
        frontier = quarter
        recent = (recent + [(beat.time, float(quarter))])[-LOCK_BEATS:]


def fit(beats: list[Beat], labels: list[Fraction | None]) -> Fit:
    pairs = [(float(q), b.time) for b, q in zip(beats, labels) if q is not None]
    if len(pairs) < 2:
        raise ValueError("Pin at least one beat with enough detected beats around it to measure tempo.")
    quarters, times = np.array(pairs).T
    slope, intercept = np.polyfit(quarters, times, 1)
    if slope <= 0:
        raise ValueError("Labeled beats do not move forward in time.")
    errors = (times - (intercept + slope * quarters)) * 1000
    pinned = {float(b.pin): b.time for b in beats if b.pin is not None}
    if 0.0 in pinned:
        downbeat = pinned[0.0]
    elif quarters[0] <= 0 <= quarters[-1]:
        downbeat = float(smooth(beats, labels, [0.0])[0])
    else:
        downbeat = float(intercept)
    curve = smooth(beats, labels, quarters)
    drift = np.abs(curve - (intercept + slope * quarters)) * 1000
    worst = int(np.argmax(drift))
    return Fit(
        60 / slope, downbeat, float(np.abs(errors).max()), float(np.sqrt(np.mean(errors ** 2))),
        float(drift[worst]), float(times[worst]), slope, intercept,
    )


def smooth(beats: list[Beat], labels: list[Fraction | None], quarters, span: float = SMOOTH_QUARTERS) -> np.ndarray:
    """Source time at each quarter from a local linear fit of labeled beats (+/- ``span`` quarters).

    Averages out tracker jitter while following real tempo drift; pinned beats are exact.
    """
    pairs = sorted((float(q), b.time) for b, q in zip(beats, labels) if q is not None)
    known_q = np.array([q for q, _ in pairs])
    known_t = np.array([t for _, t in pairs])
    pinned = {float(b.pin): b.time for b in beats if b.pin is not None}
    result = []
    for quarter in np.asarray(quarters, dtype=float).reshape(-1):
        if quarter in pinned:
            result.append(pinned[quarter])
            continue
        near = np.abs(known_q - quarter) <= span
        if near.sum() < 4:
            near = np.zeros(len(known_q), dtype=bool)
            near[np.argsort(np.abs(known_q - quarter))[:4]] = True
        if near.sum() < 2:
            raise ValueError("Not enough labeled beats to estimate timing.")
        slope, intercept = np.polyfit(known_q[near], known_t[near], 1)
        result.append(intercept + slope * quarter)
    return np.array(result)


def _drop_outliers(beats, labels, unit) -> None:
    pairs = [(i, float(q)) for i, q in enumerate(labels) if q is not None]
    if len(pairs) < 6:
        return
    quarters = np.array([q for _, q in pairs])
    times = np.array([beats[i].time for i, _ in pairs])
    period = float(np.polyfit(quarters, times, 1)[0]) * float(unit)
    limit = max(OUTLIER_SECONDS, OUTLIER_FRACTION * period)
    for k, (i, quarter) in enumerate(pairs):
        if beats[i].pin is not None:
            continue
        near = (np.abs(quarters - quarter) <= SMOOTH_QUARTERS) & (np.arange(len(pairs)) != k)
        if near.sum() < 3:
            continue
        slope, intercept = np.polyfit(quarters[near], times[near], 1)
        if abs(times[k] - (intercept + slope * quarter)) > limit:
            labels[i] = None


def anchors(beats, labels, mode: str, bar_quarters: Fraction, downbeat: float) -> list[tuple[float, float]]:
    """Tempo-map anchors (source seconds, quarter); always includes pins and quarter 0 at ``downbeat``.

    ``smooth`` places one anchor per bar on the smoothed tempo curve (no tracker jitter);
    ``every`` uses each labeled beat's detected time exactly.
    """
    if mode not in ("every", "smooth"):
        raise ValueError("Unknown anchor mode.")
    result = {0.0: downbeat}
    labeled = [float(q) for q in labels if q is not None]
    if mode == "every":
        for beat, q in zip(beats, labels):
            if q is not None and q != 0:
                result[float(q)] = beat.time
    elif labeled:
        bar = float(bar_quarters)
        bars = np.arange(math.ceil(min(labeled) / bar), math.floor(max(labeled) / bar) + 1) * bar
        bars = bars[bars != 0]
        result.update(zip(bars.tolist(), smooth(beats, labels, bars).tolist()))
    for beat in beats:
        if beat.pin is not None and beat.pin != 0:
            result[float(beat.pin)] = beat.time
    cleaned = []
    for q, t in sorted(result.items()):
        if cleaned and t <= cleaned[-1][0]:
            raise ValueError("Labeled beats are out of order near the downbeat; adjust the pins.")
        cleaned.append((t, q))
    if len(cleaned) < 2:
        raise ValueError("Need at least one labeled beat besides the downbeat.")
    return cleaned


def bar_beat(quarter: Fraction, bar_quarters: Fraction, beat_quarters: Fraction) -> tuple[int, Fraction]:
    bar = math.floor(quarter / bar_quarters)
    return bar + 1, (quarter - bar * bar_quarters) / beat_quarters + 1


def quarter_of(bar: int, beat: int, bar_quarters: Fraction, beat_quarters: Fraction) -> Fraction:
    """Bar 1 beat 1 is quarter 0; bar 0 is a pickup bar before it."""
    if not 1 <= beat <= bar_quarters / beat_quarters:
        raise ValueError("Beat is outside the bar for this meter.")
    return (bar - 1) * Fraction(bar_quarters) + (beat - 1) * Fraction(beat_quarters)


STRONG_CONFIDENCE = 0.5  # taps snap to these candidates first
SNAP_CONFIDENCE = 0.05  # weak onsets still mark time well; zero means extrapolated through silence
MIN_TAP_LAG, MAX_TAP_LAG = -0.1, 0.25  # seconds; plausible consistent tap delay (human + audio output)
TYPICAL_TAP_LAG = 0.05
TAP_WINDOW = 0.15  # seconds from a lag-corrected tap to the beat it snaps to
TAP_BAR_TOLERANCE = 0.25  # of one bar; taps further off the running bar length are ignored
PULSE_SCORE = 0.6  # finest bar division at least this well covered (above the noise floor, vs. the best) wins
TAP_RESIDUAL = 0.06  # seconds a snapped beat may sit off the other taps' bar grid
METERS = {2: (2, 4), 3: (3, 4), 4: (4, 4), 5: (5, 4), 6: (6, 8), 7: (7, 8), 8: (4, 4),
          9: (9, 8), 10: (5, 4), 11: (11, 8), 12: (12, 8)}


@dataclass
class TapResult:
    indices: list[int]  # beat index pinned for each tap that snapped to a confident beat
    bars: list[int]  # bar number of each pinned tap, relative to the first counted tap (bar 0)
    meter: tuple[int, int]
    pulses_per_bar: int  # detected pulses between downbeats (8 in 4/4 when the detector tracks eighths)
    bar_seconds: float
    lag_ms: float  # how late the taps were on average
    counted: int  # taps used to count bars (pinned or not)
    ignored: int  # double taps and taps that do not fit the bar spacing


def from_taps(beats: list[Beat], taps, *, lag_seconds: float | None = None) -> TapResult:
    """Turn tapped downbeat times into bar numbers, snapped downbeat pins and a meter guess.

    Every tap that fits the bar spacing counts bars; only taps near a confident
    detected beat (that also agrees with the neighbouring taps) become pins.
    """
    taps = np.sort(np.asarray(taps, dtype=float))
    if len(taps) < 2:
        raise ValueError("Tap at least two downbeats (more is better).")
    candidates = [i for i, b in enumerate(beats) if b.confidence >= SNAP_CONFIDENCE]
    if not candidates:
        raise ValueError("No beat candidates to snap the taps to; detect beats first.")
    times = np.array([beats[i].time for i in candidates])
    weights = np.array([beats[i].confidence for i in candidates])
    if lag_seconds is not None and not math.isfinite(lag_seconds):
        raise ValueError("Tap lag must be finite.")
    lag = _tap_lag(taps, times, weights) if lag_seconds is None else lag_seconds
    corrected = taps - lag
    keep = np.concatenate([[True], np.diff(corrected) > 0.25])
    ignored = int((~keep).sum())
    corrected = corrected[keep]
    if len(corrected) < 2:
        raise ValueError("Tap at least two downbeats (more is better).")
    gaps = np.diff(corrected)
    bar = float(np.median(gaps))
    single = gaps[np.abs(gaps / bar - 1) < TAP_BAR_TOLERANCE]
    bar = float(np.median(single)) if len(single) else bar
    positions, bars, recent = [float(corrected[0])], [0], [bar]
    for tap in corrected[1:]:
        gap = tap - positions[-1]
        local = float(np.median(recent[-8:]))
        count = round(gap / local)
        if count < 1 or abs(gap / local - count) > TAP_BAR_TOLERANCE:
            ignored += 1
            continue
        positions.append(float(tap))
        bars.append(bars[-1] + count)
        recent.append(gap / count)
    if len(positions) < 2:
        raise ValueError("The taps are not evenly spaced bars; tap only the first beat of each bar.")
    bar = float(np.median(recent[1:]))
    positions, bars = np.array(positions), np.array(bars)
    window = min(TAP_WINDOW, 0.35 * float(np.median(np.diff(times)))) if len(times) > 1 else TAP_WINDOW
    snaps = []
    for k, (tap, number) in enumerate(zip(positions, bars)):
        distance = np.abs(times - tap)
        near = distance <= window
        if not near.any():
            continue
        strong = near & (weights >= STRONG_CONFIDENCE)
        pool = strong if strong.any() else near
        index = candidates[int(np.argmax(np.where(pool, weights * np.exp(-(distance / 0.06) ** 2), -1)))]
        others = (np.arange(len(bars)) != k) & (np.abs(bars - number) <= 4)
        expected = tap
        if others.sum() >= 2:
            slope, intercept = np.polyfit(bars[others], positions[others], 1)
            expected = intercept + slope * number
        snaps.append((index, int(number), beats[index].time - expected))
    # A snap onto a neighbouring off-beat disagrees with the other taps' bar grid; the
    # median residual absorbs any remaining error in the estimated tap lag.
    center = float(np.median([residual for _, _, residual in snaps])) if snaps else 0.0
    agreed = [(i, n) for i, n, residual in snaps if abs(residual - center) <= TAP_RESIDUAL]
    # Pins become exact tempo-map anchors, so prefer confident onsets; weak ones only if needed.
    strong = [(i, n) for i, n in agreed if beats[i].confidence >= STRONG_CONFIDENCE]
    indices: list[int] = []
    pinned: list[int] = []
    for index, number in strong if len(strong) >= 2 else agreed:
        if indices and index == indices[-1]:
            continue
        indices.append(index)
        pinned.append(number)
    if len(indices) < 2:
        raise ValueError(
            "Fewer than two taps landed near a detected beat. Tap in a section with a clear beat, "
            "or pin downbeats by hand."
        )
    pin_bars = np.array(pinned, dtype=float)
    pin_times = np.array([beats[i].time for i in indices])
    starts = {}
    for number in range(int(bars[0]), int(bars[-1]) + 1):
        near = np.abs(pin_bars - number) <= 4
        if near.sum() < 2:
            near = np.ones(len(pin_bars), dtype=bool)
        slope, intercept = np.polyfit(pin_bars[near], pin_times[near], 1)
        starts[number] = float(intercept + slope * number)
    pulses = _pulses_per_bar(beats, starts)
    return TapResult(indices, pinned, METERS.get(pulses, (4, 4)), pulses, bar, lag * 1000, len(positions), ignored)


def _tap_lag(taps, times, weights) -> float:
    """Consistent tap delay that best lines the taps up with confident beats."""
    best, best_score = 0.0, -np.inf
    for lag in np.arange(MIN_TAP_LAG, MAX_TAP_LAG + 1e-9, 0.005):
        distance = np.abs(times[None, :] - (taps[:, None] - lag))
        fit = float((weights[None, :] * np.exp(-(distance / 0.04) ** 2)).max(axis=1).sum())
        # Dense candidates (eighths) align at several lags; prefer a typical human delay.
        score = fit - len(taps) * abs(lag - TYPICAL_TAP_LAG)
        if score > best_score:
            best, best_score = float(lag), score
    return best


def _pulses_per_bar(beats, starts: dict[int, float]) -> int:
    """Most detailed even division of the tapped bars that the detected beats actually fill."""
    times = np.array([b.time for b in beats])
    weights = np.array([b.confidence for b in beats])
    numbers = sorted(starts)
    spans = [(starts[n], starts[n + 1] - starts[n]) for n in numbers if n + 1 in starts]
    if not spans:
        first, last = numbers[0], numbers[-1]
        spans = [(starts[first], (starts[last] - starts[first]) / (last - first))]
    scores = {}
    for count in METERS:
        if spans[0][1] / count < 0.12:
            continue
        hits = []
        for start, length in spans:
            for k in range(1, count):
                point = start + length * k / count
                near = np.abs(times - point) <= min(0.07, 0.2 * length / count)
                hits.append(float(weights[near].max()) if near.any() else 0.0)
        scores[count] = float(np.mean(hits))
    if not scores:
        return 4
    best = max(scores.values())
    floor = float(np.median(list(scores.values())))  # divisions that don't fit the music score about this
    return max(count for count, score in scores.items() if score - floor >= PULSE_SCORE * (best - floor))
