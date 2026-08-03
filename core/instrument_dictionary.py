"""Learned non-negative spectral atoms grouped by MIDI pitch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LearnedAtom:
    midi: int
    vector: np.ndarray
    register: int
    source: str
    support: int = 1
    quality: float = 1.0


class InstrumentDictionary:
    """Store a small set of normalized, non-negative atoms per MIDI pitch."""

    def __init__(self, max_prototypes_per_midi: int = 3,
                 merge_similarity: float = 0.995) -> None:
        self.max_prototypes_per_midi = max(1, int(max_prototypes_per_midi))
        self.merge_similarity = float(merge_similarity)
        self._atoms: dict[int, list[LearnedAtom]] = {}

    def clone(self) -> "InstrumentDictionary":
        restored = InstrumentDictionary(
            self.max_prototypes_per_midi, self.merge_similarity
        )
        restored.load_dict(self.to_dict())
        return restored

    def atoms_for_midi(self, midi: int) -> tuple[LearnedAtom, ...]:
        return tuple(self._atoms.get(int(midi), ()))

    def add_atom(self, midi: int, vector: np.ndarray, *,
                 register: int | None = None, source: str = "user",
                 quality: float = 1.0) -> bool:
        candidate = np.asarray(vector, dtype=np.float32).reshape(-1)
        if candidate.size == 0 or not np.all(np.isfinite(candidate)):
            return False
        candidate = np.maximum(candidate, 0.0)
        norm = float(np.linalg.norm(candidate))
        if norm <= 1e-8:
            return False
        candidate = candidate / norm
        midi = int(midi)
        register = int(register if register is not None else midi // 12)
        quality = float(np.clip(quality, 0.0, 1.0))
        atoms = self._atoms.setdefault(midi, [])

        for index, atom in enumerate(atoms):
            similarity = float(candidate @ atom.vector)
            if (
                atom.register == register
                and atom.source == str(source)
                and similarity >= self.merge_similarity
            ):
                support = int(atom.support) + 1
                merged = (
                    atom.vector * float(atom.support) + candidate
                ) / float(support)
                merged /= max(float(np.linalg.norm(merged)), 1e-8)
                atoms[index] = LearnedAtom(
                    midi=midi,
                    vector=merged.astype(np.float32),
                    register=register,
                    source=str(source),
                    support=support,
                    quality=(
                        atom.quality * float(atom.support) + quality
                    ) / float(support),
                )
                return True

        new_atom = LearnedAtom(
            midi=midi,
            vector=candidate.astype(np.float32),
            register=register,
            source=str(source),
            quality=quality,
        )
        if len(atoms) < self.max_prototypes_per_midi:
            atoms.append(new_atom)
            return True

        weakest = min(
            range(len(atoms)),
            key=lambda index: atoms[index].quality * atoms[index].support,
        )
        if quality <= atoms[weakest].quality:
            return False
        atoms[weakest] = new_atom
        return True

    def build_augmented_dictionary(
        self,
        base_dictionary: np.ndarray,
        target_midis: list[int],
        *,
        include_standard: bool = True,
        minimum_quality: float = 0.0,
    ) -> tuple[np.ndarray, list[int]]:
        base = np.asarray(base_dictionary, dtype=np.float32)
        if base.ndim != 2 or base.shape[1] != len(target_midis):
            raise ValueError("base dictionary columns must match target MIDIs")

        columns: list[np.ndarray] = []
        labels: list[int] = []
        if include_standard:
            for index, midi in enumerate(target_midis):
                columns.append(base[:, index])
                labels.append(int(midi))

        for midi in target_midis:
            for atom in self._atoms.get(int(midi), ()):
                if atom.vector.size != base.shape[0] or atom.quality < minimum_quality:
                    continue
                columns.append(atom.vector)
                labels.append(int(midi))

        if not columns:
            return np.zeros((base.shape[0], 0), dtype=np.float32), []
        return np.column_stack(columns).astype(np.float32), labels

    @staticmethod
    def aggregate_activations(
        activations: np.ndarray, labels: list[int]
    ) -> dict[int, np.ndarray | float]:
        values = np.asarray(activations, dtype=np.float32)
        if values.ndim not in (1, 2) or values.shape[0] != len(labels):
            raise ValueError("activation rows must match atom labels")
        grouped: dict[int, np.ndarray | float] = {}
        for index, midi in enumerate(labels):
            current = grouped.get(int(midi))
            value = values[index]
            grouped[int(midi)] = value.copy() if current is None else current + value
        return grouped

    def to_dict(self) -> dict:
        return {
            "max_prototypes_per_midi": self.max_prototypes_per_midi,
            "merge_similarity": self.merge_similarity,
            "atoms": [
                {
                    "midi": atom.midi,
                    "vector": atom.vector.tolist(),
                    "register": atom.register,
                    "source": atom.source,
                    "support": atom.support,
                    "quality": atom.quality,
                }
                for midi in sorted(self._atoms)
                for atom in self._atoms[midi]
            ],
        }

    def load_dict(self, data: dict) -> None:
        self._atoms = {}
        self.max_prototypes_per_midi = max(
            1, int(data.get("max_prototypes_per_midi", self.max_prototypes_per_midi))
        )
        self.merge_similarity = float(
            data.get("merge_similarity", self.merge_similarity)
        )
        for item in data.get("atoms", []):
            vector = np.asarray(item.get("vector", []), dtype=np.float32)
            if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
                continue
            vector = np.maximum(vector, 0.0)
            norm = float(np.linalg.norm(vector))
            if norm <= 1e-8:
                continue
            midi = int(item.get("midi", -1))
            atoms = self._atoms.setdefault(midi, [])
            if len(atoms) >= self.max_prototypes_per_midi:
                continue
            atoms.append(LearnedAtom(
                midi=midi,
                vector=(vector / norm).astype(np.float32),
                register=int(item.get("register", midi // 12)),
                source=str(item.get("source", "user")),
                support=max(1, int(item.get("support", 1))),
                quality=float(np.clip(item.get("quality", 1.0), 0.0, 1.0)),
            ))
