from fractions import Fraction as F

import numpy as np

from experimental.benchmark.ensemble import PriorHead
from experimental.benchmark.harness import corrections


def test_prior_head_reproduces_baseline_and_resists_few_clicks():
    rng = np.random.default_rng(0)
    base = rng.random(200) > 0.6
    head = PriorHead(rng.normal(size=(200, 3)), base, lam=1.0)
    assert np.array_equal(head.active({}), base)
    flipped = int(np.flatnonzero(base)[0])
    assert head.active({flipped: (False, 1.0)}).sum() >= base.sum() - 5


def test_pitched_corrections_are_onset_clicks_before_split():
    predicted = {(F(0), 40), (F(1), 40), (F(9), 40)}
    reference = {(F(0), 41), (F(2), 40), (F(10), 40)}
    assert corrections("bass", predicted, reference, F(8)) == [
        ("false_onset", F(1), None, None),
        ("missed_onset", F(2), None, None),
    ]


def test_drum_wrong_class_is_one_relabel():
    predicted = {(F(0), 36), (F(1), 38), (F(2), 42)}
    reference = {(F(0), 36), (F(1), 47), (F(3), 49)}
    assert corrections("drums", predicted, reference, F(8)) == [
        ("relabel", F(1), 38, 47),
        ("remove", F(2), 42, None),
        ("add", F(3), None, 49),
    ]
