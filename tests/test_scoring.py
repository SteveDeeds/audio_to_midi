import json

import numpy as np
import pytest

from experimental.benchmark import scoring


def test_perfect_and_empty_scores_are_one_and_nothing_right_is_zero():
    assert scoring.f_beta(10, 0, 0) == 1.0
    assert scoring.f_beta(0, 0, 0) == 1.0
    assert scoring.f_beta(0, 3, 5) == 0.0


def test_wrong_note_costs_four_missed_notes_at_beta_half():
    assert scoring.f_beta(10, 1, 0) == pytest.approx(scoring.f_beta(10, 0, 4))
    assert scoring.f_beta(10, 1, 0) < scoring.f_beta(10, 0, 1)
    assert scoring.f_beta(10, 1, 0, beta=1.0) == pytest.approx(scoring.f_beta(10, 0, 1, beta=1.0))


def test_prefers_few_wrong_notes_over_more_right_with_many_wrong():
    cautious = scoring.f_beta(50, 2, 50)    # half found, two wrong
    greedy = scoring.f_beta(90, 45, 10)     # 90% found, 45 wrong
    assert cautious > greedy
    assert scoring.f_beta(50, 2, 50, beta=1.0) < scoring.f_beta(90, 45, 10, beta=1.0)


def test_array_matches_scalar():
    counts = np.array([[5, 2, 3], [0, 0, 0], [0, 4, 0], [7, 0, 1]])
    assert np.allclose(scoring.f_beta_array(counts), [scoring.f_beta(*row) for row in counts])


def test_song_pooling_counts_silent_stem_wrong_notes_against_the_song():
    counts = np.array([[8, 0, 2], [0, 0, 0], [0, 5, 0]])
    names, scores = scoring.song_scores(counts, ["a", "b", "c"])
    assert names == ["a", "b", "c"] and scores[1] == 1.0 and scores[2] == 0.0
    _, pooled = scoring.song_scores(counts, ["a", "a", "a"])
    assert pooled[0] == pytest.approx(scoring.f_beta(8, 5, 2))


def test_tuned_threshold_comes_only_from_other_folds():
    # Two thresholds per view, each view in its own fold.
    good_low, good_high = [[10, 0, 0], [5, 20, 5]], [[5, 0, 5], [10, 0, 0]]
    counts = np.array([good_high, good_low, good_low])
    # Fold 0's own view prefers index 1, but its threshold comes from the other views.
    assert scoring.tune_thresholds(counts, ["a", "b", "c"], [0, 1, 2]) == [0, 0, 0]
    counts = np.array([good_low, good_high, good_high])
    assert scoring.tune_thresholds(counts, ["a", "b", "c"], [0, 1, 2]) == [1, 0, 0]


def test_summary_reports_song_gains_against_baseline():
    rows = [
        {"source_group": "a", "policy": "frozen", "budget": 0, "tp": 5, "fp": 5, "fn": 5},
        {"source_group": "b", "policy": "frozen", "budget": 0, "tp": 5, "fp": 0, "fn": 5},
        {"source_group": "a", "policy": "edit", "budget": 1, "tp": 5, "fp": 0, "fn": 5},
        {"source_group": "b", "policy": "edit", "budget": 1, "tp": 4, "fp": 1, "fn": 6},
    ]
    frozen = scoring.song_counts([r for r in rows if r["policy"] == "frozen"], ())
    summary = scoring.summarize(rows, ("policy", "budget"), baseline=lambda key: frozen[()])
    edit = next(s for s in summary if s["policy"] == "edit")
    assert edit["improved_fraction"] == 0.5 and edit["regression_fraction"] == 0.5
    assert edit["gain"] == pytest.approx(np.mean([scoring.f_beta(5, 0, 5) - scoring.f_beta(5, 5, 5),
                                                  scoring.f_beta(4, 1, 6) - scoring.f_beta(5, 0, 5)]))


def test_streaming_array_reader_handles_small_chunks(tmp_path):
    from experimental.benchmark.fbeta_report import iter_json_array

    items = [{"i": i, "text": "x" * (i % 7), "nested": [i, {"k": "]"}]} for i in range(50)]
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(items, indent=2), encoding="utf-8")
    assert list(iter_json_array(path, chunk=13)) == items
    path.write_text("[]", encoding="utf-8")
    assert list(iter_json_array(path, chunk=13)) == []
