import numpy as np
import pytest
from scipy.special import expit

from core.evaluation import score_notes
from experimental.benchmark import adaptation_study as study
from experimental.benchmark import fresh_experiment as fx


def toy_view():
    rng = np.random.default_rng(3)
    onsets = np.sort(rng.uniform(0, 40, 120))
    pitches = rng.integers(60, 66, 120).astype(float)
    candidates = np.column_stack([onsets, onsets + 0.2, pitches, rng.uniform(0.1, 1, 120)])
    channels = {"own:basic_pitch": candidates}
    reference = candidates[rng.random(120) < 0.5].copy()
    reference[:, 3] = 80
    record = {"clip": "c", "source_group": "g", "split": "train"}
    return fx.View(record, "piano", "single_basic_pitch", channels, candidates,
                   fx.features(candidates, channels), reference.tolist())


@pytest.mark.parametrize("architecture", study.ARCHITECTURES)
def test_every_architecture_fits_and_adapts_without_reading_query_truth(architecture):
    view = toy_view()
    head = study.fit([view], architecture, iterations=20)
    rows = study.evaluate_view(view, head, architecture)
    assert rows[0]["policy"] == "frozen" and rows[0]["gain"] == 0
    assert all(np.isfinite(row["f1"]) and 0 <= row["f1"] <= 1 for row in rows)
    if architecture != "boosted_trees":
        before = head.pack().copy()
        study.evaluate_view(view, head, architecture)
        assert np.array_equal(head.pack(), before)
    changed = fx.View(view.record, view.role, view.config, view.channels, view.candidates, view.X,
                      [row for row in view.reference if row[0] < fx.SUPPORT_SECONDS])
    stream = fx.edit_stream(view, head)
    for budget in study.BUDGETS:
        window = study.reviewed_window(view, stream, budget)
        assert np.all(view.candidates[window, 0] < fx.SUPPORT_SECONDS)
        assert np.array_equal(view.y[window], changed.y[window])


def test_edit_stream_matches_experiment_at_half_and_follows_threshold():
    view = toy_view()
    head = study.fit([view], "mlp4", iterations=20)
    expected = fx.edit_stream(view, head)
    actual = study.edit_stream(view, head, 0.5)
    assert len(actual) == len(expected)
    assert all(np.array_equal(a[0], b[0]) and a[1] == b[1] for a, b in zip(actual, expected))
    strict = study.edit_stream(view, head, 0.95)
    assert sum(label == 0 for _, label in strict) <= sum(label == 0 for _, label in expected)
    rows = study.evaluate_view(view, head, "mlp4", 0.8)
    assert all(row["threshold"] == 0.8 for row in rows)
    frozen = study.score(view, head.probabilities(view.X) >= 0.8)
    assert (rows[0]["tp"], rows[0]["fp"], rows[0]["fn"]) == frozen[:3]


@pytest.mark.parametrize("architecture", study.ARCHITECTURES)
def test_views_without_candidates_score_reference_as_missed(architecture):
    view = toy_view()
    head = study.fit([view], architecture, iterations=20)
    empty = np.zeros((0, 4))
    blank = fx.View(view.record, view.role, view.config, {"own:basic_pitch": empty}, empty,
                    fx.features(empty, {"own:basic_pitch": empty}), view.reference)
    rows = study.evaluate_view(blank, head, architecture)
    assert rows and all(row["tp"] == 0 and row["fp"] == 0 and row["f1"] == 0 for row in rows)


def test_reviewed_window_ends_at_last_edit_or_support_end():
    view = toy_view()
    stream = [(np.array([5.0, 5.2, 60, 1]), 1.0), (np.array([9.0, 9.2, 61, 1]), 0.0)]
    window = study.reviewed_window(view, stream, 1)
    assert window.any() and view.candidates[window, 0].max() <= 5.0
    assert not study.reviewed_window(view, stream, 0).any()
    full = study.reviewed_window(view, stream, 2 + 1)
    assert np.array_equal(full, view.candidates[:, 0] < fx.SUPPORT_SECONDS)


def test_platt_recalibration_moves_threshold_toward_window_labels():
    z = np.linspace(-3, 3, 200)
    y = (z > 1.5).astype(float)
    adapted = study.platt(z, z, y, lam=0.1)
    assert np.mean((adapted >= 0) == y) > np.mean((z >= 0) == y)
    assert np.allclose(study.platt(z, z[:0], y[:0], lam=10), z)


def test_gate_rejects_updates_that_do_not_help_later_reviewed_notes():
    class Fixed:
        def logits(self, X):
            return X[:, 0] * 4
    X = np.linspace(-1, 1, 30)[:, None]
    good = (X[:, 0] > 0).astype(float)
    misleading_early = np.where(X[:, 0] < 0.3, (X[:, 0] < 0).astype(float), good)
    logits, accepted = study.gated(Fixed(), "platt_window", 0.01, X, X, misleading_early, X[:, 0])
    assert not accepted and np.array_equal(logits, X[:, 0] * 4)
    shifted = (X[:, 0] > 0.5).astype(float)
    onsets = np.random.default_rng(0).permutation(len(X)).astype(float)
    logits, accepted = study.gated(Fixed(), "platt_window", 0.01, X, X, shifted, onsets)
    assert accepted and np.mean((logits >= 0) == shifted) > np.mean((X[:, 0] >= 0) == shifted)
    logits, accepted = study.gated(Fixed(), "platt_window", 1.0, X, X[:3], good[:3], X[:3, 0])
    assert not accepted


def test_fast_scoring_matches_reference_f1():
    view = toy_view()
    keep = np.random.default_rng(1).random(len(view.candidates)) < 0.6
    tp, fp, fn, f1 = study.score(view, keep)
    reference = [row for row in view.reference if row[0] >= fx.QUERY_START]
    predicted = view.candidates[keep & (view.candidates[:, 0] >= fx.QUERY_START)].tolist()
    expected = score_notes(reference, predicted)
    assert (tp, fp, fn) == (expected.true_positives, expected.false_positives, expected.false_negatives)
    assert f1 == pytest.approx(expected.f1)
    assert expit(0) == 0.5
