import unittest

import numpy as np

from core.adtof_drums import (
    CLASS_MIDI, DEFAULT_THRESHOLDS, AdtofDrumTranscriber, drum_class, fit_threshold,
    peak_novelty, pick_peaks,
)

KICK, SNARE, TOM, HIHAT, CYMBAL = range(5)

def activations(seconds: float, hits: list[tuple[float, int, float]], extra=None) -> np.ndarray:
    """Synthetic ADTOF output: one-frame spikes with a short decay."""
    act = np.zeros((int(seconds * 100), 5), dtype=np.float32)
    for t, cls, value in hits:
        f = int(round(t * 100))
        act[f:f + 3, cls] = np.maximum(act[f:f + 3, cls], [value, value * 0.5, value * 0.2])
    for t, cls, value in extra or ():
        act[int(round(t * 100)), cls] = value
    return act

def transcriber(act: np.ndarray) -> AdtofDrumTranscriber:
    tr = AdtofDrumTranscriber("drums.wav")
    tr.set_activations(act)
    return tr

class TestAdtofPeakPicking(unittest.TestCase):
    def test_matches_adtof_reference_peak_picker(self):
        try:
            import adtof_pytorch as adtof
        except ModuleNotFoundError:
            self.skipTest("adtof-pytorch not installed")
        rng = np.random.default_rng(0)
        act = (rng.random((3000, 5)) ** 6).astype(np.float32)
        ref = adtof.PeakPicker(thresholds=adtof.FRAME_RNN_THRESHOLDS, fps=100).pick(
            act[None], labels=adtof.LABELS_5)[0]
        novelty = peak_novelty(act)
        for cls, times in enumerate(ref.values()):
            mine = pick_peaks(novelty[:, cls], DEFAULT_THRESHOLDS[cls]) / 100.0
            np.testing.assert_allclose(mine, times)

    def test_gm_mapping_covers_kit_pieces(self):
        self.assertEqual([drum_class(m) for m in CLASS_MIDI], [0, 1, 2, 3, 4])
        self.assertEqual(drum_class(40), SNARE)
        self.assertEqual(drum_class(51), CYMBAL)
        self.assertIsNone(drum_class(54))

class TestAdtofAdaptation(unittest.TestCase):
    def test_fit_threshold_prefers_default_and_separates_feedback(self):
        self.assertEqual(fit_threshold([], [], 0.24), 0.24)
        self.assertEqual(fit_threshold([0.5], [], 0.24), 0.24)
        self.assertAlmostEqual(fit_threshold([0.6], [0.3], 0.24), 0.301)
        self.assertAlmostEqual(fit_threshold([0.15], [], 0.24), 0.149)
        # An unreachable rejection never silences the whole class.
        self.assertEqual(fit_threshold([], [0.99], 0.24), 0.24)
        self.assertAlmostEqual(fit_threshold([], [0.9], 0.24), 0.901)

    def test_rejecting_a_ghost_note_raises_that_class_threshold_only(self):
        tr = transcriber(activations(4, [(0.5, SNARE, 0.9), (1.5, SNARE, 0.3), (2.5, KICK, 0.3)]))
        self.assertEqual(len(tr.hits()), 3)
        tr.record_feedback(1.45, 1.6, 38, keep=False)
        self.assertEqual([(h.time, h.midi) for h in tr.hits()], [(0.5, 38), (2.5, 36)])
        self.assertGreater(tr.thresholds[SNARE], DEFAULT_THRESHOLDS[SNARE])
        self.assertEqual(tr.thresholds[KICK], DEFAULT_THRESHOLDS[KICK])

    def test_confirming_a_quiet_hit_lowers_threshold(self):
        tr = transcriber(activations(4, [(0.5, HIHAT, 0.6), (1.5, HIHAT, 0.15), (2.5, HIHAT, 0.12)]))
        self.assertEqual(len(tr.hits()), 1)
        tr.record_feedback(1.45, 1.6, 42, keep=True)
        self.assertEqual([h.time for h in tr.hits()], [0.5, 1.5])

    def test_confirming_silence_does_not_flood_the_class(self):
        tr = transcriber(activations(4, [(0.5, TOM, 0.8)], extra=[(2.0, TOM, 0.1)]))
        tr.record_feedback(3.0, 3.15, 47, keep=True)
        self.assertEqual(tr.thresholds[TOM], DEFAULT_THRESHOLDS[TOM])

    def test_relabel_teaches_snare_like_toms_but_keeps_real_toms(self):
        # This kit's snare fires the tom output with a weak snare response; real toms do not.
        snare_like = [(t, TOM, 0.7) for t in (0.5, 1.5, 2.5)]
        snare_resp = [(t, SNARE, 0.2) for t in (0.5, 1.5, 2.5)]
        real_toms = [(t, TOM, 0.7) for t in (3.5, 4.5)]
        tr = transcriber(activations(6, snare_like + snare_resp + real_toms))
        self.assertEqual({h.midi for h in tr.hits()}, {47})
        tr.record_feedback(0.45, 0.6, 47, keep=False)
        tr.record_feedback(0.45, 0.6, 38, keep=True)
        tr.record_feedback(3.45, 3.6, 47, keep=True)
        labels = {h.time: h.midi for h in tr.hits()}
        self.assertEqual(labels, {0.5: 38, 1.5: 38, 2.5: 38, 3.5: 47, 4.5: 47})
        self.assertEqual(tr.thresholds[TOM], DEFAULT_THRESHOLDS[TOM])

    def test_feedback_before_inference_is_applied_once_ready(self):
        tr = AdtofDrumTranscriber("drums.wav")
        self.assertEqual(tr.hits(), ())
        tr.record_feedback(1.45, 1.6, 38, keep=False)
        tr.set_activations(activations(3, [(0.5, SNARE, 0.9), (1.5, SNARE, 0.3)]))
        self.assertEqual([h.time for h in tr.hits()], [0.5])

class TestAdtofInference(unittest.TestCase):
    def test_real_model_runs_on_audio(self):
        try:
            import adtof_pytorch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("adtof-pytorch not installed")
        from core.adtof_drums import compute_activations

        sr = 44100
        audio = np.zeros(sr * 3, dtype=np.float32)
        rng = np.random.default_rng(1)
        for t in (0.5, 1.5, 2.5):
            n = int(0.1 * sr)
            kick = np.sin(2 * np.pi * 55 * np.arange(n) / sr) * np.exp(-np.arange(n) / (0.03 * sr))
            audio[int(t * sr):int(t * sr) + n] += kick.astype(np.float32)
            audio[int(t * sr):int(t * sr) + n] += 0.05 * rng.standard_normal(n).astype(np.float32)
        act = compute_activations(audio)
        self.assertEqual(act.shape[1], 5)
        self.assertAlmostEqual(len(act) / 100.0, 3.0, delta=0.05)
        self.assertTrue(np.all((act >= 0) & (act <= 1)))

if __name__ == "__main__":
    unittest.main()
