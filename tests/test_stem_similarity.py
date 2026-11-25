import numpy as np
import pytest

from audio_to_midi.timebase import Grid
from experimental.stem_similarity import (
    block_similarity, normalize_slices, pool_slices, project_grid, slice_edges,
)


def test_block_similarity_preserves_slice_order_and_matches_diagonal_average():
    sequence = np.eye(4, dtype=np.float32)
    features = np.concatenate((sequence, sequence, sequence[::-1], sequence[[1, 0, 2, 3]]))
    slices = block_similarity(features, 1)
    blocks = block_similarity(features, 4)
    assert blocks[0, 1] == pytest.approx(1)
    assert blocks[0, 2] == pytest.approx(0)
    assert blocks[0, 3] == pytest.approx(0.5)
    for i in range(4):
        for j in range(4):
            assert blocks[i, j] == pytest.approx(np.trace(slices[i*4:(i+1)*4, j*4:(j+1)*4]) / 4)
    np.testing.assert_array_equal(block_similarity(features[:15], 4), blocks[:3, :3])


def test_four_bar_similarity_keeps_all_64_positions():
    phrase = np.eye(64, dtype=np.float32)
    features = np.concatenate((phrase, phrase, np.roll(phrase, 16, axis=0)))
    actual = block_similarity(features, 64)
    np.testing.assert_allclose(actual, [[1, 1, 0], [1, 1, 0], [0, 0, 1]])


def test_four_bar_windows_start_every_bar_and_drop_incomplete_tail():
    bars = np.eye(3, dtype=np.float32)[[0, 1, 0, 1, 0, 2, 0]]
    features = np.repeat(bars, 16, axis=0)[:6 * 16 + 15]
    actual = block_similarity(features, 64, stride=16)
    np.testing.assert_allclose(actual, [[1, 0, 0.75], [0, 1, 0], [0.75, 0, 1]])


def test_silence_is_not_similarity_evidence():
    normalized = normalize_slices(np.ones((4, 3), dtype=np.float32), np.array([1, 0, 1e-5, 1]))
    actual = block_similarity(normalized, 1)
    assert actual[0, 3] == pytest.approx(1)
    assert np.all(actual[1:3] == 0)
    assert block_similarity(normalized, 4)[0, 0] == pytest.approx(0.5)
    assert not np.any(normalize_slices(np.ones((2, 3)), np.zeros(2)))


def test_pooling_uses_half_open_slice_intervals():
    values = np.array([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]], dtype=np.float32)
    pooled = pool_slices(values, np.arange(5) * 0.1, np.array([0, 0.2, 0.4]))
    np.testing.assert_array_equal(pooled, [[2, 3], [6, 7]])
    with pytest.raises(ValueError, match="hop"):
        pool_slices(values, np.arange(5), np.array([0.1, 0.2]))


def test_edges_follow_project_downbeat_and_omit_partial_tail():
    grid = Grid(120, offset_seconds=0.3)
    edges, first = slice_edges(grid, 8.49)
    assert first == 1
    assert len(edges) == 66
    assert edges[0] == pytest.approx(0.3)
    assert edges[-1] == pytest.approx(8.425)
    np.testing.assert_allclose(np.diff(edges), 0.125)


def test_edges_follow_variable_tempo_anchors():
    grid = project_grid({
        "bpm": 120, "numerator": 4, "denominator": 4, "offset_seconds": 0,
        "anchors": [[s, {"numerator": q, "denominator": 1}] for s, q in ((0, 0), (4, 8), (12, 16))],
    })
    edges, first = slice_edges(grid, 12)
    assert first == 1
    assert len(edges) == 65
    np.testing.assert_allclose(np.diff(edges[:33]), 0.125)
    np.testing.assert_allclose(np.diff(edges[32:]), 0.25)
