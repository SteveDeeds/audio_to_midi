import numpy as np

from core.evaluation import score_notes
from experimental.benchmark import threshold_sweep as sweep


def test_per_pitch_matching_equals_full_one_to_one_scoring():
    rng = np.random.default_rng(7)
    for _ in range(30):
        reference = np.column_stack([np.sort(rng.uniform(0, 5, 40)), np.zeros(40),
                                     rng.integers(60, 64, 40), np.ones(40)])
        reference[:, 1] = reference[:, 0] + 0.2
        predictions = reference[rng.random(40) < 0.7].copy()
        predictions[:, 0] += rng.normal(0, 0.03, len(predictions))
        extra = reference[:15].copy()
        extra[:, 2] += rng.integers(-1, 2, 15)
        predictions = np.vstack([predictions, extra])
        assert sweep.true_positives(reference, predictions) == score_notes(
            reference.tolist(), predictions.tolist()).true_positives


def test_threshold_choice_never_uses_the_held_fold():
    thresholds = (0.2, 0.8)
    totals = np.array([
        [[10, 0, 0], [0, 0, 10]],     # held-out fold strongly prefers the low threshold
        [[10, 30, 0], [8, 1, 2]],
        [[10, 30, 0], [8, 1, 2]],
    ])
    result = sweep.cross_fold(totals, thresholds, cost=2, default=0)
    assert result["fold_thresholds"][0] == 0.8
    assert result["tuned"]["true_positives"] == 16
    assert result["default"]["false_positives"] == 60


def test_counts_drop_low_scores_and_report_wrong_notes():
    reference = np.array([[1.0, 1.2, 60, 1], [2.0, 2.2, 62, 1]])
    predictions = np.array([[1.01, 1.2, 60, 0.9], [2.0, 2.2, 61, 0.1], [3.0, 3.2, 64, 0.6]])
    result = sweep.counts(reference, predictions, predictions[:, 3], (0.0, 0.5, 0.95))
    assert result.tolist() == [[1, 2, 1], [1, 1, 1], [0, 0, 2]]
    assert sweep.rates(1, 2, 1, cost=2)["net"] == -3
