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

    def prepare(self, t_start, t_end, stem, tracker, should_cancel=None, max_batches=None):
        cancelled = should_cancel or (lambda: False)
        check_cancel(cancelled)
        if (stem, tracker) not in self._runs:
            arrays = file_evidence(str(self.wav_path), f"stem:{stem}", tracker, cancelled)
            events = replay_poly(tracker, arrays, stem, self.parameters)
            self._chunks[(stem, tracker, (0.0, self.duration))] = [
                PolyNote(a, b, p, v, True, confidence) for a, b, p, v, confidence in events]
            self._rebuild(stem, tracker)
        check_cancel(cancelled)
        return True


class InstrumentOnsetTranscriber(MonoTranscriber):
    """Instrument-range CREPE + offline madmom, not the vocal syllable pipeline."""

    def __init__(self, audio, sr, path, parameters):
        super().__init__(audio, sr, path)
        self.parameters = parameters

    def prepare(self, t_start, t_end, stem, tracker, should_cancel=None, max_batches=None):
        cancelled = should_cancel or (lambda: False)
        check_cancel(cancelled)
        if (stem, tracker) not in self._clean:
            pitch = pitch_evidence(self.audio, self.sr, stem, "crepe_full", cancelled)
            onsets = onset_evidence(self.audio, self.sr, stem, tracker, cancelled)
            times, midi = replay_pitch("crepe_full", pitch, self.parameters)
            # These instrument pipelines use the measured range, not vocal key folding.
            allowed = np.ones((12, len(times)), dtype=bool)
            self._contours[(stem, tracker, (0.0, self.duration))] = times, midi, allowed
            self._clean_cached(stem, tracker)
            from madmom.features.onsets import OnsetPeakPickingProcessor
            self._onsets[stem] = np.asarray(OnsetPeakPickingProcessor(
                fps=100, threshold=threshold(self.parameters, "onset_threshold", .3),
                smooth=0, pre_max=.03, post_max=.03, combine=.12)(onsets["onset"].copy()))
        check_cancel(cancelled)
        return True
