"""Alternative pitch adaptation models for controlled comparison."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment, nnls

from core.instrument_dictionary import InstrumentDictionary


PitchInterval = tuple[float, float, frozenset[int]]


def baseline_pitch_set(markers, t0: float, t1: float) -> set[int]:
    """Run current pitch inference for one known interval without reference count."""
    fingerprint = markers._cqt_fingerprint_for_block(t0, t1)
    suggestions, similarity = markers.model.pitch_suggestions(fingerprint)
    if suggestions:
        candidates = [
            (int(midi), float(similarity))
            for midi in suggestions[:markers._nnls_max_polyphony]
        ]
    else:
        candidates = markers._pitch_model.select_nnls_notes(
            markers._nnls_notes_for_block(t0, t1)
        )
    if not candidates or not markers._pitch_model.uses_pitch_gate:
        return {int(midi) for midi, _ in candidates}

    feats = markers._nnls_feature_vectors_for_block(t0, t1)
    energy_db = {}
    for midi, _ in candidates:
        sounding_midi = markers._pitch_model.sounding_midi(int(midi))
        energy = markers._harmonic_block_energy(t0, t1, sounding_midi)
        energy_db[int(midi)] = float(20.0 * np.log10(max(energy, 1e-8)))
    max_db = max(energy_db.values()) if energy_db else -120.0
    selected = set()
    for midi, _ in candidates:
        midi = int(midi)
        index = markers._nnls_midi_to_idx.get(midi)
        probability = None
        if feats is not None and index is not None:
            relative_db = float(energy_db[midi] - max_db)
            probability = markers.model.pitch_gate_prob_with_features(
                peel=float(feats["peel_gain"][index]),
                salience=float(feats["salience_gain"][index]),
                flux=float(feats["flux_gain"][index]),
                onset=float(feats["onset"][index]),
                energy_db=energy_db[midi],
                energy_rel_db=relative_db,
            )
            if probability is not None:
                intensity = float(np.clip((relative_db + 18.0) / 12.0, 0.0, 1.0))
                probability *= 0.60 + 0.40 * intensity
        if probability is None or probability >= markers.model.PITCH_GATE_THRESHOLD:
            selected.add(midi)
    return selected


def interval_observation(markers, t0: float, t1: float) -> np.ndarray:
    """Return a normalized non-negative CQT observation for an exact interval."""
    first = int(np.searchsorted(markers.cqt_times, float(t0), side="left"))
    last = int(np.searchsorted(markers.cqt_times, float(t1), side="right"))
    first = max(0, min(first, markers.cqt_mag.shape[1] - 1))
    last = max(first + 1, min(last, markers.cqt_mag.shape[1]))
    observation = np.percentile(markers.cqt_mag[:, first:last], 75.0, axis=1)
    observation = np.maximum(observation.astype(np.float32), 0.0)
    norm = float(np.linalg.norm(observation))
    return observation / norm if norm > 1e-8 else observation


def harmonic_mask(markers, displayed_midi: int, radius: int = 1) -> np.ndarray:
    """Create a CQT mask around the first five harmonics of a displayed MIDI."""
    sounding_midi = markers._pitch_model.sounding_midi(int(displayed_midi))
    mask = np.zeros(markers.cqt_mag.shape[0], dtype=np.float32)
    for harmonic in (1, 2, 3, 4, 5):
        harmonic_midi = sounding_midi + 12.0 * np.log2(float(harmonic))
        center = int(round(harmonic_midi - markers.cqt_fmin_midi))
        lo = max(0, center - int(radius))
        hi = min(mask.size, center + int(radius) + 1)
        mask[lo:hi] = 1.0
    return mask


def masked_atom(markers, observation: np.ndarray, displayed_midi: int) -> np.ndarray:
    atom = np.maximum(np.asarray(observation, dtype=np.float32), 0.0)
    atom = atom * harmonic_mask(markers, displayed_midi)
    norm = float(np.linalg.norm(atom))
    return atom / norm if norm > 1e-8 else atom


def select_activations(
    activations: dict[int, float], threshold_ratio: float, max_polyphony: int,
) -> set[int]:
    """Select support using learned threshold and configured cap, not reference count."""
    positive = {int(midi): max(0.0, float(value)) for midi, value in activations.items()}
    peak = max(positive.values(), default=0.0)
    if peak <= 1e-8:
        return set()
    threshold = float(threshold_ratio) * peak
    ranked = sorted(
        ((midi, value) for midi, value in positive.items() if value >= threshold),
        key=lambda item: item[1],
        reverse=True,
    )
    return {midi for midi, _ in ranked[:max(1, int(max_polyphony))]}


def _normalized_columns(matrix: np.ndarray) -> np.ndarray:
    columns = np.asarray(matrix, dtype=np.float32).copy()
    norms = np.linalg.norm(columns, axis=0)
    valid = norms > 1e-8
    columns[:, valid] /= norms[valid]
    return columns


def _base_dictionary(markers) -> tuple[np.ndarray, list[int]]:
    dictionary = _normalized_columns(markers._nnls_dictionary)
    labels = []
    for midi in markers._nnls_target_midis:
        mapped = markers._pitch_model.select_nnls_notes([(int(midi), 1.0)])
        labels.append(int(mapped[0][0]) if mapped else int(midi))
    return dictionary, labels


class OffsetCalibrationModel:
    """Learn one robust semitone correction while preserving baseline selection."""

    name = "offset_calibration"

    def __init__(self, markers) -> None:
        self.markers = markers
        self.offset = 0

    def fit(self, examples: list[PitchInterval]) -> None:
        deltas: list[int] = []
        for t0, t1, reference in examples:
            predicted = sorted(baseline_pitch_set(self.markers, t0, t1))
            expected = sorted(reference)
            if not predicted or not expected:
                continue
            costs = np.abs(
                np.asarray(expected, dtype=float)[:, None]
                - np.asarray(predicted, dtype=float)[None, :]
            )
            expected_indices, predicted_indices = linear_sum_assignment(costs)
            deltas.extend(
                expected[i] - predicted[j]
                for i, j in zip(expected_indices, predicted_indices)
                if abs(expected[i] - predicted[j]) <= 24
            )
        if deltas:
            counts = Counter(deltas)
            self.offset = int(max(counts, key=lambda value: (counts[value], -abs(value))))

    def predict(self, t0: float, t1: float) -> set[int]:
        return {
            max(0, min(127, int(midi) + self.offset))
            for midi in baseline_pitch_set(self.markers, t0, t1)
        }

    def summary(self) -> dict:
        return {"offset": self.offset}


class _DictionaryModel:
    def __init__(self, markers, threshold_ratio: float = 0.35) -> None:
        self.markers = markers
        self.threshold_ratio = float(threshold_ratio)
        self.dictionary, self.labels = _base_dictionary(markers)

    def _activations(self, t0: float, t1: float) -> dict[int, float]:
        observation = interval_observation(self.markers, t0, t1)
        if self.dictionary.shape[1] == 0 or not np.any(observation):
            return {}
        coefficients, _ = nnls(self.dictionary, observation)
        grouped: dict[int, float] = defaultdict(float)
        for label, coefficient in zip(self.labels, coefficients):
            grouped[int(label)] += float(coefficient)
        return dict(grouped)

    def predict(self, t0: float, t1: float) -> set[int]:
        return select_activations(
            self._activations(t0, t1), self.threshold_ratio,
            int(self.markers._nnls_max_polyphony),
        )

    def _calibrate_threshold(self, examples: list[PitchInterval]) -> None:
        activation_rows = [self._activations(t0, t1) for t0, t1, _ in examples]
        best = (-1.0, -1.0, self.threshold_ratio)
        for threshold in np.linspace(0.10, 0.90, 17):
            tp = fp = fn = 0
            for activations, (_, _, reference) in zip(activation_rows, examples):
                predicted = select_activations(
                    activations, float(threshold),
                    int(self.markers._nnls_max_polyphony),
                )
                tp += len(predicted & reference)
                fp += len(predicted - reference)
                fn += len(reference - predicted)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
            candidate = (f1, precision, float(threshold))
            if candidate > best:
                best = candidate
        self.threshold_ratio = best[2]

    def summary(self) -> dict:
        return {
            "threshold_ratio": self.threshold_ratio,
            "dictionary_columns": int(self.dictionary.shape[1]),
        }


class HarmonicEnvelopeModel(_DictionaryModel):
    """Learn one non-negative five-harmonic envelope for each confirmed MIDI."""

    name = "harmonic_envelope"

    def fit(self, examples: list[PitchInterval]) -> None:
        base, labels = _base_dictionary(self.markers)
        learned: dict[int, list[np.ndarray]] = defaultdict(list)
        for t0, t1, reference in examples:
            observation = interval_observation(self.markers, t0, t1)
            for midi in reference:
                atom = masked_atom(self.markers, observation, int(midi))
                if np.any(atom):
                    learned[int(midi)].append(atom)
        for midi, atoms in learned.items():
            merged = np.mean(np.column_stack(atoms), axis=1)
            norm = float(np.linalg.norm(merged))
            if norm <= 1e-8:
                continue
            merged = merged / norm
            for index, label in enumerate(labels):
                if label == midi:
                    base[:, index] = merged
        self.dictionary = _normalized_columns(base)
        self.labels = labels
        self._calibrate_threshold(examples)


class MultiAtomNNLSModel(_DictionaryModel):
    """Add multiple MIDI-labeled non-negative atoms and solve their joint sum."""

    name = "multi_atom_nnls"

    def __init__(self, markers, atom_bank: InstrumentDictionary | None = None,
                 threshold_ratio: float = 0.35) -> None:
        super().__init__(markers, threshold_ratio=threshold_ratio)
        self.atom_bank = atom_bank or InstrumentDictionary(
            max_prototypes_per_midi=3, merge_similarity=0.985
        )
        self.refresh_dictionary()

    @property
    def has_learned_atoms(self) -> bool:
        return any(
            self.atom_bank.atoms_for_midi(midi) for midi in self.markers._nnls_target_midis
        )

    def refresh_dictionary(self) -> None:
        base, labels = _base_dictionary(self.markers)
        self.dictionary, self.labels = self.atom_bank.build_augmented_dictionary(
            base, labels, include_standard=True
        )
        self.dictionary = _normalized_columns(self.dictionary)

    def observe(self, t0: float, t1: float, reference: Iterable[int]) -> None:
        observation = interval_observation(self.markers, t0, t1)
        for midi in reference:
            atom = masked_atom(self.markers, observation, int(midi))
            self.atom_bank.add_atom(
                int(midi), atom, source="user", quality=1.0
            )
        self.refresh_dictionary()

    def calibrate(self, examples: list[PitchInterval]) -> None:
        if examples:
            self._calibrate_threshold(examples)

    def fit(self, examples: list[PitchInterval]) -> None:
        for t0, t1, reference in examples:
            self.observe(t0, t1, reference)
        self.calibrate(examples)

    def summary(self) -> dict:
        summary = super().summary()
        summary["learned_atoms"] = sum(
            len(self.atom_bank.atoms_for_midi(midi)) for midi in set(self.labels)
        )
        return summary


MODEL_TYPES = {
    OffsetCalibrationModel.name: OffsetCalibrationModel,
    HarmonicEnvelopeModel.name: HarmonicEnvelopeModel,
    MultiAtomNNLSModel.name: MultiAtomNNLSModel,
}


def create_adaptation_models(markers) -> Iterable:
    return (model_type(markers) for model_type in MODEL_TYPES.values())