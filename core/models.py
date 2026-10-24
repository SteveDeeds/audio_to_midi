"""
core/models.py
==============
ML correction models trained incrementally from user edits.

  CorrectionModel
    - Unified candidate logistic regression: keep vs delete
    - Pitch cosine 1-NN: fingerprint → MIDI note suggestion
"""

import copy

import numpy as np

from core.instrument_dictionary import InstrumentDictionary
from core.logistic_utils import sigmoid_clipped, train_binary_logistic_minmax


class CorrectionModel:
    """Two sub-models trained incrementally from user corrections."""

    PITCH_SIM_THRESHOLD = 0.88
    PITCH_MARGIN_THRESHOLD = 0.015
    PITCH_SINGLE_CLASS_THRESHOLD = 0.97
    PITCH_GATE_THRESHOLD = 0.55

    def __init__(self) -> None:
        self._cand_X: list[list[float]] = []
        self._cand_y: list[int] = []
        self._cand_keys: list[int | None] = []
        self._cand_src: list[str] = []
        self._cand_w: np.ndarray = np.zeros(5)
        self._cand_norm: tuple = (np.zeros(4), np.ones(4))
        self._cand_trained: bool = False

        self._fp_vecs: list[np.ndarray] = []  # 5-dim harmonic profiles
        self._fp_notes: list[tuple[int, ...]] = []  # polyphonic: tuple of sorted MIDI notes
        self._fp_midi: list[float] = []  # median MIDI note for each fingerprint (for register weighting)
        self._fp_src: list[str] = []
        self.instrument_dictionary = InstrumentDictionary()
        self.multi_atom_threshold_ratio = 0.35

        # Pitch acceptance gate (latest model): logistic regression on
        # [peel, salience, flux, onset, energy_db, energy_rel_db].
        self._pitch_gate_X: list[list[float]] = []
        self._pitch_gate_y: list[int] = []
        self._pitch_gate_keys: list[str | None] = []
        self._pitch_gate_src: list[str] = []
        self._pitch_gate_w: np.ndarray = np.zeros(7)
        self._pitch_gate_norm: tuple = (np.zeros(6), np.ones(6))
        self._pitch_gate_trained: bool = False

        # Drum pattern storage: up to 3 representative prototypes per drum MIDI
        # _drum_patterns_dict[drum_midi] = [(proto_vec, support_count), ...]
        self._drum_patterns_dict: dict[int, list[tuple[np.ndarray, int]]] = {}  # drum_midi -> [(vec, support), ...]
        self._drum_training_X: list[list[float]] = []  # Training features for pattern acceptance
        self._drum_training_y: list[int] = []  # Accept (1) or reject (0)
        self._drum_keys: list[str | None] = []
        self._drum_src: list[str] = []
        self._drum_w: np.ndarray = np.zeros(10)  # Logistic regression weights + bias (9 features + bias)
        self._drum_norm: tuple = (np.zeros(9), np.ones(9))  # 9 percussive features
        self._drum_trained: bool = False
        self._MAX_PROTOTYPES_PER_DRUM = 3  # Cap prototypes to limit NNLS dictionary size

        self._seed_bootstrap_candidates()
        self._seed_bootstrap_pitch_gate()

    def clone(self) -> "CorrectionModel":
        """Return an isolated training copy, including non-persisted live state."""
        return copy.deepcopy(self)

    def learning_metrics(self) -> dict[str, int]:
        """Return training-set and learned-state sizes for diagnostics."""
        atoms = [
            atom
            for midi_atoms in self.instrument_dictionary._atoms.values()
            for atom in midi_atoms
        ]
        fixed_weight_scalars = int(
            self._cand_w.size + self._pitch_gate_w.size + self._drum_w.size
        )
        fingerprint_scalars = int(sum(vector.size for vector in self._fp_vecs))
        atom_scalars = int(sum(atom.vector.size for atom in atoms))
        return {
            "candidate_examples": len(self._cand_X),
            "pitch_gate_examples": len(self._pitch_gate_X),
            "fingerprint_examples": len(self._fp_vecs),
            "drum_examples": len(self._drum_training_X),
            "learned_atoms": len(atoms),
            "fixed_weight_scalars": fixed_weight_scalars,
            "fingerprint_scalars": fingerprint_scalars,
            "atom_scalars": atom_scalars,
            "learned_scalar_values": (
                fixed_weight_scalars + fingerprint_scalars + atom_scalars
            ),
        }

    # ── Candidate model ──────────────────────────────────────────────────

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return sigmoid_clipped(x)

    def _retrain_binary(self, X_rows, y_rows):
        if not X_rows:
            return False, np.zeros(3), (np.zeros(2), np.ones(2))
        return train_binary_logistic_minmax(
            X_rows,
            y_rows,
            learning_rate=1.0,
            n_iter_min=200,
            n_iter_max=800,
            n_iter_factor=40,
        )

    def _append_candidate_sample(self, feat: list[float], keep: bool,
                                 candidate_key=None, source: str = "user") -> None:
        y_val = 1 if keep else 0
        if candidate_key is not None and candidate_key in self._cand_keys:
            idx = self._cand_keys.index(candidate_key)
            self._cand_X[idx] = feat
            self._cand_y[idx] = y_val
            self._cand_src[idx] = source
        else:
            # Deduplicate unlabeled-key samples (e.g. imported profile rows)
            # so toggling/merging does not create repeated training points.
            if candidate_key is None:
                for i, (row, key) in enumerate(zip(self._cand_X, self._cand_keys)):
                    if key is None and np.allclose(np.array(row), np.array(feat), atol=1e-9):
                        self._cand_y[i] = y_val
                        self._cand_src[i] = source
                        return
            self._cand_X.append(feat)
            self._cand_y.append(y_val)
            self._cand_keys.append(candidate_key)
            self._cand_src.append(source)

    @staticmethod
    def _map_candidate_row(row: list[float] | tuple[float, ...]) -> list[float]:
        """Map legacy candidate row shapes to the active 4-feature format."""
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            raise ValueError("Candidate rows require at least three numeric features.")
        vals = [float(v) for v in row]
        if not all(np.isfinite(v) for v in vals):
            raise ValueError("Candidate features must be finite.")
        if len(vals) >= 5:
            return [vals[0], vals[1], vals[2], vals[4]]
        if len(vals) == 4:
            return [vals[0], vals[1], vals[2], vals[3]]
        while len(vals) < 4:
            vals.append(0.0)
        return vals[:4]

    @classmethod
    def candidate_rows_from_payload(cls, data: dict, user_only: bool = False) -> list[dict]:
        """Return normalized candidate rows from new or legacy profile payloads."""
        out: list[dict] = []
        if "cand" in data:
            candidates = data["cand"]
        else:
            # Backward compatibility with older saved profiles.
            rows, labels = data.get("cand_X", []), data.get("cand_y", [])
            if len(rows) != len(labels):
                raise ValueError("Legacy candidate rows and labels must have matching lengths.")
            candidates = [{"row": row, "label": label} for row, label in zip(rows, labels)]
        for item in candidates:
            source = str(item.get("source", "user"))
            if user_only and source != "user":
                continue
            label = item.get("label")
            if label not in (0, 1):
                raise ValueError("Candidate labels must be binary (0 or 1).")
            out.append({
                "row": cls._map_candidate_row(item.get("row", [])),
                "label": int(label),
                "key": item.get("key"),
                "source": source,
            })
        return out

    def _seed_bootstrap_candidates(self) -> None:
        if self._cand_X:
            return
        # Two obvious positive examples and two obvious negative examples.
        # Positive = note-like energy / transient evidence.
        self._append_candidate_sample([0.92, 0.74, -14.0, 0.18], keep=True, source="bootstrap")
        self._append_candidate_sample([0.68, 0.49, -20.0, 0.08], keep=True, source="bootstrap")
        # Negative = essentially silence / no amplitude / no clear onset.
        self._append_candidate_sample([0.02, 0.01, -78.0, -0.10], keep=False, source="bootstrap")
        self._append_candidate_sample([0.07, 0.02, -64.0, -0.04], keep=False, source="bootstrap")
        trained, w, norm = self._retrain_binary(self._cand_X, self._cand_y)
        self._cand_trained, self._cand_w, self._cand_norm = trained, w, norm

    def record_candidate(self, transient, chroma_delta, energy_db,
                         percussive_spike, keep, candidate_key=None, source: str = "user"):
        y_val = 1 if keep else 0
        feat = [float(transient), float(chroma_delta), float(energy_db), float(percussive_spike)]
        self._append_candidate_sample(feat, bool(keep), candidate_key=candidate_key, source=source)
        trained, w, norm = self._retrain_binary(self._cand_X, self._cand_y)
        self._cand_trained, self._cand_w, self._cand_norm = trained, w, norm

    def record_candidate_batch(self, examples: list[tuple]) -> None:
        """Record candidate labels and retrain once after the complete batch."""
        for transient, chroma_delta, energy_db, percussive_spike, keep, key, source in examples:
            feat = [
                float(transient), float(chroma_delta),
                float(energy_db), float(percussive_spike),
            ]
            self._append_candidate_sample(
                feat, bool(keep), candidate_key=key, source=str(source)
            )
        trained, w, norm = self._retrain_binary(self._cand_X, self._cand_y)
        self._cand_trained, self._cand_w, self._cand_norm = trained, w, norm

    def candidate_keep_prob(self, transient, chroma_delta, energy_db, percussive_spike):
        if not self._cand_trained:
            return None
        lo, rng = self._cand_norm
        xn = (np.array([transient, chroma_delta, energy_db, percussive_spike]) - lo) / rng
        return float(self._sigmoid(float(np.append(xn, 1.0) @ self._cand_w)))

    # ── Pitch model (polyphonic) ────────────────────────────────────────

    def record_pitch(self, fingerprint: np.ndarray, midi: int, source: str = "user"):
        """Record a single pitch for a fingerprint (legacy monophonic API)."""
        self.record_pitches(fingerprint, [midi], source)

    def record_pitches(self, fingerprint: np.ndarray, midis: list[int],
                       source: str = "user") -> None:
        """Record all pitches for one block's fingerprint (5-harmonic profile).
        If the same fingerprint (cosine >= 0.995) already exists from the same
        source, update its notes in-place rather than appending duplicates.
        Stores median MIDI note for register-based weighting during queries."""
        norm = np.linalg.norm(fingerprint)
        if norm < 1e-8 or not midis:
            return
        fp_normed = fingerprint / norm
        notes = tuple(sorted(set(int(m) for m in midis)))
        median_midi = float(np.median([float(m) for m in midis]))

        # Check for existing near-duplicate fingerprint from same source
        for i, (vec, src) in enumerate(zip(self._fp_vecs, self._fp_src)):
            if src == source and float(fp_normed @ vec) >= 0.995:
                self._fp_notes[i] = notes
                self._fp_midi[i] = median_midi
                return

        self._fp_vecs.append(fp_normed)
        self._fp_notes.append(notes)
        self._fp_midi.append(median_midi)
        self._fp_src.append(str(source))

    def pitch_suggestions(self, fingerprint: np.ndarray) -> tuple[list[int], float]:
        """Return all MIDI notes for the best-matching fingerprint block.
        Uses register-based weighting: prefers matches clustered with other 
        high-similarity matches and penalizes isolated matches far from the cluster.
        Returns (midi_list, similarity).  Empty list if no confident match."""
        if not self._fp_vecs:
            return [], 0.0
        norm = np.linalg.norm(fingerprint)
        if norm < 1e-8:
            return [], 0.0
        q = fingerprint / norm
        sims = np.array([float(q @ v) for v in self._fp_vecs])

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim < self.PITCH_SIM_THRESHOLD:
            return [], best_sim

        # Check margin: is there a conflicting entry (different note-set) close behind?
        best_notes = self._fp_notes[best_idx]
        second_best_sim = 0.0
        for i, sim in enumerate(sims):
            if self._fp_notes[i] != best_notes and sim > second_best_sim:
                second_best_sim = sim

        if second_best_sim > 0 and best_sim - second_best_sim < self.PITCH_MARGIN_THRESHOLD:
            return [], best_sim

        if len(self._fp_vecs) == 1 or second_best_sim == 0.0:
            if best_sim < self.PITCH_SINGLE_CLASS_THRESHOLD:
                return [], best_sim

        # Register affinity: prefer matches clustered with others at similar register
        best_midi = self._fp_midi[best_idx]
        register_affinity = self._compute_register_affinity(best_idx, best_midi)
        adjusted_sim = best_sim * register_affinity

        return list(best_notes), adjusted_sim

    def _compute_register_affinity(self, match_idx: int, query_midi: float) -> float:
        """Return 0.7-1.0 affinity based on register clustering.
        
        High affinity: best match surrounded by similar-register matches.
        Low affinity: best match isolated from its register peers.
        """
        if len(self._fp_vecs) <= 1:
            return 1.0
        
        # Find all high-similarity matches (>0.90) from same note-set as best
        best_notes = self._fp_notes[match_idx]
        same_note_sims = []
        same_register_count = 0
        
        q_vec = self._fp_vecs[match_idx]
        for i, (vec, notes, midi) in enumerate(zip(self._fp_vecs, self._fp_notes, self._fp_midi)):
            if i == match_idx:
                continue
            if notes != best_notes:
                continue
            sim = float(q_vec @ vec)
            if sim > 0.90:
                same_note_sims.append(sim)
                # Register proximity: within 3 semitones
                if abs(midi - query_midi) <= 3:
                    same_register_count += 1
        
        # Affinity boost if we have nearby register matches
        if same_register_count > 0:
            return 1.0  # High confidence
        elif len(same_note_sims) > 0:
            return 0.95  # Good confidence, different registers
        else:
            return 0.85  # Lower confidence, isolated match

    def pitch_shape_similarity_for_midi(self, fingerprint: np.ndarray, midi: int) -> float:
        """Return best cosine similarity to stored fingerprints that contain midi."""
        if not self._fp_vecs:
            return 0.0
        norm = float(np.linalg.norm(fingerprint))
        if norm < 1e-8:
            return 0.0
        q = fingerprint / norm
        best = 0.0
        target = int(midi)
        for vec, notes in zip(self._fp_vecs, self._fp_notes):
            if target not in notes:
                continue
            best = max(best, float(q @ vec))
        return float(np.clip(best, 0.0, 1.0))

    def _append_pitch_gate_sample(self, feat: list[float], keep: bool,
                                  sample_key: str | None = None,
                                  source: str = "user") -> None:
        y_val = 1 if keep else 0
        if sample_key is not None and sample_key in self._pitch_gate_keys:
            idx = self._pitch_gate_keys.index(sample_key)
            self._pitch_gate_X[idx] = feat
            self._pitch_gate_y[idx] = y_val
            self._pitch_gate_src[idx] = source
            return
        if sample_key is None:
            for i, (row, key) in enumerate(zip(self._pitch_gate_X, self._pitch_gate_keys)):
                if key is None and np.allclose(np.array(row), np.array(feat), atol=1e-9):
                    self._pitch_gate_y[i] = y_val
                    self._pitch_gate_src[i] = source
                    return
        self._pitch_gate_X.append(feat)
        self._pitch_gate_y.append(y_val)
        self._pitch_gate_keys.append(sample_key)
        self._pitch_gate_src.append(source)

    def record_pitch_gate_example_with_features(self, peel: float, salience: float,
                                                 flux: float, onset: float,
                                                 energy_db: float, energy_rel_db: float,
                                                 keep: bool, sample_key: str | None = None,
                                                 source: str = "user") -> None:
        """Record one labeled pitch-acceptance example.

        Features (all NNLS signals from the block):
        - peel: spectral peel level [0, 1]
        - salience: pitch salience [0, 1]
        - flux: onset flux novelty [0, 1]
        - onset: onset pre/post novelty [0, 1]
        - energy_db: absolute dB level of note
        - energy_rel_db: note dB minus strongest-note dB in this block
        """
        peel_c = float(np.clip(peel, 0.0, 1.0))
        salience_c = float(np.clip(salience, 0.0, 1.0))
        flux_c = float(np.clip(flux, 0.0, 1.0))
        onset_c = float(np.clip(onset, 0.0, 1.0))
        en_db = float(energy_db)
        en_rel_db = float(energy_rel_db)
        self._append_pitch_gate_sample(
            [peel_c, salience_c, flux_c, onset_c, en_db, en_rel_db],
            bool(keep),
            sample_key=sample_key,
            source=source,
        )
        trained, w, norm = self._retrain_binary(self._pitch_gate_X, self._pitch_gate_y)
        self._pitch_gate_trained, self._pitch_gate_w, self._pitch_gate_norm = trained, w, norm

    def record_pitch_gate_batch(self, examples: list[tuple]) -> None:
        """Record pitch-gate labels and retrain once after the complete batch."""
        for peel, salience, flux, onset, energy_db, energy_rel_db, keep, key, source in examples:
            self._append_pitch_gate_sample(
                [
                    float(np.clip(peel, 0.0, 1.0)),
                    float(np.clip(salience, 0.0, 1.0)),
                    float(np.clip(flux, 0.0, 1.0)),
                    float(np.clip(onset, 0.0, 1.0)),
                    float(energy_db),
                    float(energy_rel_db),
                ],
                bool(keep),
                sample_key=key,
                source=str(source),
            )
        trained, w, norm = self._retrain_binary(self._pitch_gate_X, self._pitch_gate_y)
        self._pitch_gate_trained, self._pitch_gate_w, self._pitch_gate_norm = trained, w, norm

    def _seed_bootstrap_pitch_gate(self) -> None:
        if self._pitch_gate_X:
            return
        # Keep: strong peel/salience/onset and reasonable intensity.
        # [peel, salience, flux, onset, energy_db, energy_rel_db]
        self._append_pitch_gate_sample([0.85, 0.72, -0.15, 0.60, -18.0, -1.5], True, source="bootstrap")
        self._append_pitch_gate_sample([0.68, 0.55, -0.08, 0.42, -24.0, -3.0], True, source="bootstrap")
        # Reject: weak structure or low relative intensity in block.
        self._append_pitch_gate_sample([0.15, 0.08, -0.85, 0.05, -22.0, -10.0], False, source="bootstrap")
        self._append_pitch_gate_sample([0.72, 0.60, -0.02, 0.55, -48.0, -14.0], False, source="bootstrap")
        trained, w, norm = self._retrain_binary(self._pitch_gate_X, self._pitch_gate_y)
        self._pitch_gate_trained, self._pitch_gate_w, self._pitch_gate_norm = trained, w, norm

    def pitch_gate_prob_with_features(self, peel: float, salience: float,
                                       flux: float, onset: float,
                                       energy_db: float, energy_rel_db: float) -> float | None:
        """Return P(accept) for a pitch suggestion using NNLS features.

        Args:
            peel: spectral peel [0, 1]
            salience: pitch salience [0, 1]
            flux: flux novelty [0, 1]
            onset: onset novelty [0, 1]
            energy_db: absolute dB level
            energy_rel_db: dB relative to strongest candidate in block
        """
        if not self._pitch_gate_trained:
            return None
        peel_c = float(np.clip(peel, 0.0, 1.0))
        salience_c = float(np.clip(salience, 0.0, 1.0))
        flux_c = float(np.clip(flux, 0.0, 1.0))
        onset_c = float(np.clip(onset, 0.0, 1.0))
        en_db = float(energy_db)
        en_rel_db = float(energy_rel_db)
        lo, rng = self._pitch_gate_norm
        xn = (np.array([peel_c, salience_c, flux_c, onset_c, en_db, en_rel_db]) - lo) / rng
        return float(self._sigmoid(float(np.append(xn, 1.0) @ self._pitch_gate_w)))

    # ── Drum pattern model ────────────────────────────────────────────────

    def _append_drum_training_sample(self, feat: list[float], keep: bool,
                                     sample_key: str | None = None,
                                     source: str = "user") -> None:
        """Append or update drum training sample."""
        y_val = 1 if keep else 0
        if sample_key is not None and sample_key in self._drum_keys:
            idx = self._drum_keys.index(sample_key)
            self._drum_training_X[idx] = feat
            self._drum_training_y[idx] = y_val
            self._drum_src[idx] = source
            return
        if sample_key is None:
            for i, (row, key) in enumerate(zip(self._drum_training_X, self._drum_keys)):
                if key is None and np.allclose(np.array(row), np.array(feat), atol=1e-9):
                    self._drum_training_y[i] = y_val
                    self._drum_src[i] = source
                    return
        self._drum_training_X.append(feat)
        self._drum_training_y.append(y_val)
        self._drum_keys.append(sample_key)
        self._drum_src.append(source)

    def record_drum_pattern(self, drum_midi: int, fingerprint: np.ndarray,
                           attack_slope: float, impulsivity: float,
                           decay_rate: float, energy_db: float, energy_rel_db: float,
                           band_low: float, band_mid: float, band_high: float,
                           keep: bool, sample_key: str | None = None,
                           source: str = "user") -> None:
        """Record a drum pattern with 9 percussive-only features.
        
        Args:
            drum_midi: MIDI note for this drum (e.g., 36 for kick)
            fingerprint: STFT feature vector for the drum sample
            attack_slope: dB rise per ms (transient sharpness)
            impulsivity: peak / RMS ratio (impulse strength)
            decay_rate: dB drop per 100ms (decay character)
            energy_db: absolute dB level
            energy_rel_db: dB relative to loudest in beat
            band_low: energy in low freq band [0, 1]
            band_mid: energy in mid freq band [0, 1]
            band_high: energy in high freq band [0, 1]
            keep: True if user accepted, False if rejected
            sample_key: unique identifier for this sample (for deduplication)
            source: source label ('user', 'ml', 'bootstrap', etc.)
        """
        # Normalize and store the fingerprint
        norm = float(np.linalg.norm(fingerprint))
        if norm > 1e-8:
            fingerprint_normed = fingerprint / norm
        else:
            fingerprint_normed = fingerprint
        
        # Normalize features
        attack_slope_c = float(np.clip(attack_slope, 0.0, 1.0))
        impulsivity_c = float(np.clip(impulsivity, 0.0, 1.0))
        decay_rate_c = float(np.clip(decay_rate, 0.0, 1.0))
        en_db = float(energy_db)
        en_rel_db = float(energy_rel_db)
        band_low_c = float(np.clip(band_low, 0.0, 1.0))
        band_mid_c = float(np.clip(band_mid, 0.0, 1.0))
        band_high_c = float(np.clip(band_high, 0.0, 1.0))
        
        self._append_drum_training_sample(
            [attack_slope_c, impulsivity_c, decay_rate_c, en_db, en_rel_db, 
             band_low_c, band_mid_c, band_high_c, 0.0],
            bool(keep),
            sample_key=sample_key,
            source=source,
        )
        
        # Store or update drum patterns (capped at 3 prototypes per drum)
        if keep:
            drum_midi_int = int(drum_midi)
            if drum_midi_int not in self._drum_patterns_dict:
                self._drum_patterns_dict[drum_midi_int] = []
            
            prototypes = self._drum_patterns_dict[drum_midi_int]
            
            if len(prototypes) < self._MAX_PROTOTYPES_PER_DRUM:
                # Add new prototype
                prototypes.append((fingerprint_normed.copy(), 1))
            else:
                # Assign to nearest prototype and update via EMA
                best_idx = -1
                best_sim = -1.0
                for i, (proto_vec, _) in enumerate(prototypes):
                    if proto_vec.shape == fingerprint_normed.shape:
                        sim = float(fingerprint_normed @ proto_vec)
                        if sim > best_sim:
                            best_sim = sim
                            best_idx = i
                
                if best_idx >= 0:
                    # Update via weighted EMA
                    proto_vec, support = prototypes[best_idx]
                    # EMA weight: inverse of support count (newer updates matter more)
                    alpha = 1.0 / (1.0 + support)
                    updated = (1.0 - alpha) * proto_vec + alpha * fingerprint_normed
                    updated_norm = float(np.linalg.norm(updated))
                    if updated_norm > 1e-8:
                        updated = updated / updated_norm
                    prototypes[best_idx] = (updated, support + 1)
                elif best_sim < 0.5:
                    # Far from all prototypes: replace weakest (lowest support)
                    weakest_idx = min(range(len(prototypes)), key=lambda i: prototypes[i][1])
                    prototypes[weakest_idx] = (fingerprint_normed.copy(), 1)
        
        # Retrain acceptance model (now 9 features)
        trained, w, norm = self._retrain_binary(self._drum_training_X, self._drum_training_y)
        self._drum_trained, self._drum_w, self._drum_norm = trained, w, norm

    def get_drum_acceptance_prob(self, attack_slope: float, impulsivity: float,
                                  decay_rate: float, energy_db: float, energy_rel_db: float,
                                  band_low: float, band_mid: float, band_high: float) -> float | None:
        """Return P(accept) for a drum candidate using the trained logistic regression.
        
        Uses 9-feature percussive-only space.
        Returns None if model not yet trained (will use default behavior).
        """
        if not self._drum_trained:
            return None
        attack_slope_c = float(np.clip(attack_slope, 0.0, 1.0))
        impulsivity_c = float(np.clip(impulsivity, 0.0, 1.0))
        decay_rate_c = float(np.clip(decay_rate, 0.0, 1.0))
        en_db = float(energy_db)
        en_rel_db = float(energy_rel_db)
        band_low_c = float(np.clip(band_low, 0.0, 1.0))
        band_mid_c = float(np.clip(band_mid, 0.0, 1.0))
        band_high_c = float(np.clip(band_high, 0.0, 1.0))
        lo, rng = self._drum_norm
        xn = (np.array([attack_slope_c, impulsivity_c, decay_rate_c, en_db, en_rel_db,
                       band_low_c, band_mid_c, band_high_c, 0.0]) - lo) / rng
        return float(self._sigmoid(float(np.append(xn, 1.0) @ self._drum_w)))

    def get_drum_patterns(self, expected_size: int | None = None) -> list[tuple[np.ndarray, int]]:
        """Return all learned drum patterns as (fingerprint, drum_midi) tuples.
        
        Flattens the prototype dict into a list for NNLS solver.
        Each drum contributes up to 3 prototypes.
        """
        patterns = []
        for drum_midi, prototypes in self._drum_patterns_dict.items():
            for proto_vec, _ in prototypes:
                patterns.append((proto_vec, drum_midi))
        
        if expected_size is None:
            return patterns
        return [
            (vec, midi)
            for vec, midi in patterns
            if int(np.size(vec)) == int(expected_size)
        ]

    # ── Persistence ──────────────────────────────────────────────────────

    def to_dict(self, user_only: bool = True) -> dict:
        pitch_items = []
        for vec, notes, midi, src in zip(self._fp_vecs, self._fp_notes, self._fp_midi, self._fp_src):
            if user_only and src != "user":
                continue
            pitch_items.append({
                "vec": vec.tolist(),
                "notes": list(notes),
                "midi": float(midi),
                "source": src
            })
        pitch_gate_rows = []
        for row, label, key, src in zip(self._pitch_gate_X, self._pitch_gate_y,
                                        self._pitch_gate_keys, self._pitch_gate_src):
            if user_only and src != "user":
                continue
            pitch_gate_rows.append({
                "row": [float(v) for v in row],
                "label": int(label),
                "key": key,
                "source": src,
            })
        cand_rows = []
        for row, label, key, src in zip(self._cand_X, self._cand_y, self._cand_keys, self._cand_src):
            if user_only and src != "user":
                continue
            cand_rows.append({
                "row": [float(v) for v in row],
                "label": int(label),
                "key": int(key) if key is not None else None,
                "source": src,
            })
        drum_patterns = []
        for drum_midi, prototypes in self._drum_patterns_dict.items():
            for vec, support in prototypes:
                drum_patterns.append({
                    "vec": vec.tolist(),
                    "drum_midi": int(drum_midi),
                    "support": int(max(1, support)),
                })
        return {
            "schema": 12,
            "cand": cand_rows,
            "pitch": pitch_items,
            "pitch_gate": pitch_gate_rows,
            "drum_patterns": drum_patterns,
            "instrument_dictionary": self.instrument_dictionary.to_dict(),
            "multi_atom_threshold_ratio": self.multi_atom_threshold_ratio,
        }

    def load_dict(self, data: dict, merge: bool = False):
        candidate_rows = self.candidate_rows_from_payload(data, user_only=False)
        if not merge:
            self.__init__()

        for item in candidate_rows:
            self._append_candidate_sample(
                item["row"],
                bool(int(item["label"])),
                candidate_key=item.get("key"),
                source=str(item.get("source", "user")),
            )
        trained, w, norm = self._retrain_binary(self._cand_X, self._cand_y)
        self._cand_trained, self._cand_w, self._cand_norm = trained, w, norm
        for item in data.get("pitch", []):
            vec = np.array(item["vec"], dtype=np.float32)
            # Support both old ("midi": int) and new ("notes": list) formats
            if "notes" in item:
                self.record_pitches(vec, item["notes"], item.get("source", "user"))
            else:
                self.record_pitch(vec, item["midi"], item.get("source", "user"))

        instrument_dictionary = data.get("instrument_dictionary")
        if isinstance(instrument_dictionary, dict):
            self.instrument_dictionary.load_dict(instrument_dictionary)
        self.multi_atom_threshold_ratio = float(np.clip(
            data.get("multi_atom_threshold_ratio", 0.35), 0.10, 0.90
        ))

        for item in data.get("pitch_gate", []):
            row = item.get("row")
            if not isinstance(row, (list, tuple)) or len(row) != 6:
                continue  # Latest model only: require exactly 6 features
            vals = [float(v) for v in row]
            self._append_pitch_gate_sample(
                vals,
                bool(int(item.get("label", 0))),
                sample_key=item.get("key"),
                source=str(item.get("source", "user")),
            )
        trained, w, norm = self._retrain_binary(self._pitch_gate_X, self._pitch_gate_y)
        self._pitch_gate_trained, self._pitch_gate_w, self._pitch_gate_norm = trained, w, norm

        # Load drum patterns (schema v10+)
        for item in data.get("drum_patterns", []):
            vec_list = item.get("vec")
            drum_midi = item.get("drum_midi")
            if isinstance(vec_list, (list, tuple)) and isinstance(drum_midi, int):
                vec = np.array(vec_list, dtype=np.float32)
                # Normalize to unit length
                norm = float(np.linalg.norm(vec))
                if norm > 1e-8:
                    vec = vec / norm
                support = int(item.get("support", 1))
                support = max(1, support)
                drum_midi_int = int(drum_midi)
                self._drum_patterns_dict.setdefault(drum_midi_int, []).append((vec, support))

        # Keep only the strongest prototypes per drum after merges.
        for drum_midi, prototypes in list(self._drum_patterns_dict.items()):
            if len(prototypes) <= self._MAX_PROTOTYPES_PER_DRUM:
                continue
            ordered = sorted(prototypes, key=lambda item: int(item[1]), reverse=True)
            self._drum_patterns_dict[drum_midi] = ordered[:self._MAX_PROTOTYPES_PER_DRUM]
