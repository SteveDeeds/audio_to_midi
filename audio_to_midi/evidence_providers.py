"""Feed reusable evidence into the existing onset-grid/edit providers."""

import numpy as np

from audio_to_midi.model_evidence import (
    check_cancel, file_evidence, onset_evidence, pitch_evidence, replay_pitch, replay_poly, threshold,
)
from core.mono_pitch import MonoTranscriber
from core.poly_pitch import PolyNote, PolyTranscriber


class EvidencePolyTranscriber(PolyTranscriber):
    def __init__(self, audio, sr, path, parameters):
        super().__init__(audio, sr, path)
        self.parameters = parameters
        self.report = lambda text: None

    def prepare(self, t_start, t_end, stem, tracker, should_cancel=None, max_batches=None):
        cancelled = should_cancel or (lambda: False)
        check_cancel(cancelled)
        for a, b in self._missing(t_start, t_end, stem, tracker):
            check_cancel(cancelled)
            context = self.CONTEXT_SECONDS[tracker]
            padded_start, padded_end = max(0.0, a - context), min(self.duration, b + context)
            arrays = file_evidence(str(self.wav_path), f"stem:{stem}", tracker, cancelled,
                                   self.report, interval=(padded_start, padded_end))
            events = replay_poly(tracker, arrays, stem, self.parameters)
            notes = []
            for onset, stop, pitch, velocity, confidence in events:
                onset, stop = onset + padded_start, min(stop + padded_start, padded_end)
                if a <= onset < b:
                    notes.append(PolyNote(onset, stop, pitch, velocity, True, confidence))
                elif onset < a < stop:
                    notes.append(PolyNote(a, stop, pitch, velocity, False, confidence))
            self._chunks[(stem, tracker, (a, b))] = notes
            self._rebuild(stem, tracker)
        check_cancel(cancelled)
        return True


class InstrumentOnsetTranscriber(MonoTranscriber):
    """Instrument-range CREPE + offline madmom, not the vocal syllable pipeline."""

    def __init__(self, audio, sr, path, parameters):
        super().__init__(audio, sr, path)
        self.parameters = parameters
        self.report = lambda text: None
        self._attack_chunks: dict[tuple[str, str, float, float], np.ndarray] = {}

    def prepare(self, t_start, t_end, stem, tracker, should_cancel=None, max_batches=None):
        cancelled = should_cancel or (lambda: False)
        check_cancel(cancelled)
        for a, b in self._missing(t_start, t_end, stem, tracker):
            check_cancel(cancelled)
            padded_start = max(0.0, a - self.CONTEXT_SECONDS)
            padded_end = min(self.duration, b + self.CONTEXT_SECONDS)
            first, last = round(padded_start * self.sr), round(padded_end * self.sr)
            offset = first / self.sr
            audio = self.audio[first:last]
            pitch = pitch_evidence(audio, self.sr, stem, "crepe_full", cancelled, report=self.report)
            self.report("Detecting attacks in this chunk...")
            onsets = onset_evidence(audio, self.sr, stem, tracker, cancelled)
            times, midi = replay_pitch("crepe_full", pitch, self.parameters)
            times = times + offset
            keep = (times >= a) & (times < b)
            times, midi = times[keep], midi[keep]
            # These instrument pipelines use the measured range, not vocal key folding.
            allowed = np.ones((12, len(times)), dtype=bool)
            self._contours[(stem, tracker, (a, b))] = times, midi, allowed
            self._clean_cached(stem, tracker)
            from madmom.features.onsets import OnsetPeakPickingProcessor
            attacks = np.asarray(OnsetPeakPickingProcessor(
                fps=100, threshold=threshold(self.parameters, "onset_threshold", .3),
                smooth=0, pre_max=.03, post_max=.03, combine=.12)(onsets["onset"].copy())) + offset
            self._attack_chunks[(stem, tracker, a, b)] = attacks[(attacks >= a) & (attacks < b)]
            self._onsets[stem] = np.unique(np.concatenate([
                values for (s, t, _, _), values in self._attack_chunks.items()
                if (s, t) == (stem, tracker)
            ]))
        check_cancel(cancelled)
        return True
