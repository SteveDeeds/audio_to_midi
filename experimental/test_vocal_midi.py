from pathlib import Path

import mido
import numpy as np
import pytest

from experimental.vocal_midi import (
    Detector, NEURAL_PEAK_PARAMETERS, Note, articulation_release, coalesce, export_midi, loudest_window,
    fixed_window, make_notes, stable_changes, stereo_comparison, synthesize, voice_regions,
    write_review,
)


def grid():
    times = np.arange(0, 2, 0.01)
    return times, np.full(len(times), 60.0), np.full(len(times), -5.0)


def test_loudest_window_is_exact_not_hop_aligned():
    audio = np.zeros((100, 2))
    audio[37:67, 0] = 1
    audio[37:67, 1] = -1
    assert loudest_window(audio, 10, 3) == (37, 67, 1.0)


def test_loudest_window_matches_every_brute_force_window():
    audio = np.random.default_rng(41).normal(size=(143, 2))
    power = (audio ** 2).mean(axis=1)
    expected = np.argmax([power[a:a + 30].mean() for a in range(114)])
    a, b, rms = loudest_window(audio, 10, 3)
    assert a == expected
    assert b - a == 30
    assert rms ** 2 == pytest.approx(power[a:b].mean())


@pytest.mark.parametrize("audio", [np.zeros(100), np.ones(20), np.full(100, np.nan)])
def test_invalid_or_silent_source_is_explicit_error(audio):
    with pytest.raises(ValueError):
        loudest_window(audio, 10, 3)


def test_fixed_window_selects_requested_time_not_louder_audio():
    audio = np.ones((500, 2))
    audio[100:200] = 10
    assert fixed_window(audio, 10, 10, 25.5) == (255, 355, 1.0)


def test_fixed_window_rounds_start_to_sample_grid_and_allows_exact_eof():
    audio = np.ones(100)
    assert fixed_window(audio, 10, 3, 7.04) == (70, 100, 1.0)


@pytest.mark.parametrize(("start", "seconds"), [
    (-1, 1), (float("nan"), 1), (float("inf"), 1), (0, 0),
    (0, -1), (0, float("nan")), (0, float("inf")), (9.5, 1), (0, 0.001),
])
def test_fixed_window_rejects_invalid_or_truncated_intervals(start, seconds):
    with pytest.raises(ValueError):
        fixed_window(np.ones(100), 10, seconds, start)


@pytest.mark.parametrize("audio", [np.zeros(100), np.full(100, np.nan)])
def test_fixed_window_rejects_silent_or_nonfinite_audio(audio):
    with pytest.raises(ValueError):
        fixed_window(audio, 10, 1, 0)


def test_review_describes_fixed_selection_without_claiming_loudest(tmp_path: Path):
    report = {"source": {"path": "vocals.mp3"}, "methods": {},
              "window": {"mode": "fixed", "start_seconds": 25.5, "end_seconds": 35.5}}
    write_review(tmp_path, report)
    page = (tmp_path / "review.html").read_text(encoding="utf-8")
    assert "Selected excerpt 10s" in page
    assert "25.500000-35.500000s" in page
    assert "Loudest continuous" not in page


def test_voice_regions_bridge_only_short_internal_dropouts():
    times, pitch, db = grid()
    pitch[:10] = np.nan
    pitch[40:43] = np.nan
    pitch[100:110] = np.nan
    pitch[180:] = np.nan
    regions = voice_regions(times, pitch, db, 2)
    assert np.asarray(regions) == pytest.approx(np.array([[0.1, 1.0], [1.1, 1.8]]))


def test_vibrato_does_not_create_slurred_notes():
    times, pitch, _ = grid()
    pitch += 0.55 * np.sin(2 * np.pi * 6 * times)
    assert not len(stable_changes(times, pitch))


def test_stable_pitch_change_is_backdated_and_short_excursion_ignored():
    times, pitch, _ = grid()
    pitch[30:34] = 62
    pitch[100:] = 64
    assert stable_changes(times, pitch) == pytest.approx([1.0])


def test_release_detects_consonant_voicing_gap_not_vibrato():
    times, pitch, db = grid()
    pitch[96:99] = np.nan
    release, reason = articulation_release(0, 1, times, pitch, db)
    assert release == pytest.approx(0.96)
    assert reason == "pre-attack voicing gap"
    pitch[96:99] = 60
    assert articulation_release(0, 1, times, pitch, db) == (1, "connected")


def test_release_detects_energy_valley():
    times, pitch, db = grid()
    db[95:99] = -15
    release, reason = articulation_release(0, 1, times, pitch, db)
    assert release == pytest.approx(0.95)
    assert reason == "pre-attack energy dip"


def test_notes_encode_slurs_ties_and_articulations():
    times, pitch, db = grid()
    pitch[50:100] = 62
    pitch[100:] = 62
    pitch[146:149] = np.nan
    detector = Detector(np.array([0, 0.5, 1, 1.5]), times, np.zeros(len(times)), {})
    notes = make_notes(detector, times, pitch, db, [(0, 2)])
    assert [n.link_to_next for n in notes] == ["slurred", "tied", "articulated", "phrase_end"]
    assert notes[0].end == notes[1].start
    assert notes[1].end == notes[2].start
    assert notes[2].end == pytest.approx(1.46)
    assert notes[2].end < notes[3].start
    assert notes[-1].end == 2
    assert all(n.end > n.start for n in notes)


def test_notes_end_at_voice_release_not_the_next_distant_onset():
    times, pitch, db = grid()
    pitch[70:130] = np.nan
    detector = Detector(np.array([0, 1.3]), times, np.zeros(len(times)), {})
    notes = make_notes(detector, times, pitch, db, [(0, 0.7), (1.3, 2)])
    assert [n.end for n in notes] == [0.7, 2]
    assert notes[0].gap_to_next_ms == 600


def test_close_raw_onsets_do_not_generate_tiny_notes():
    assert coalesce(np.array([0.5, 0.52, 0.58, 0.60])) == pytest.approx([0.5, 0.58])


def test_neural_peak_picker_preserves_probability_scale():
    from madmom.features.onsets import OnsetPeakPickingProcessor

    activation = np.zeros(100, dtype=np.float32)
    activation[[20, 50, 80]] = 0.95
    # madmom's unnormalized 2-frame Hamming smoothing attenuates peaks below .3.
    assert OnsetPeakPickingProcessor(**NEURAL_PEAK_PARAMETERS)(activation) == pytest.approx(
        [0.2, 0.5, 0.8],
    )


def test_synth_ties_do_not_retrigger_and_gaps_are_silent():
    sr = 8000
    joined = [Note(0, 0.5, 60, 100, "", ""), Note(0.5, 1, 60, 100, "", "")]
    whole = [Note(0, 1, 60, 100, "", "")]
    assert synthesize(joined, sr, sr) == pytest.approx(synthesize(whole, sr, sr), abs=1e-11)
    separated = [Note(0.1, 0.4, 60, 100, "", ""), Note(0.5, 0.9, 64, 100, "", "")]
    audio = synthesize(separated, sr, sr)
    assert not np.any(audio[3200:4000])
    assert not np.any(audio[:800])
    assert not np.any(audio[7200:])
    freqs = np.fft.rfftfreq(2000, 1 / sr)
    peak = freqs[np.argmax(np.abs(np.fft.rfft(audio[1000:3000])))]
    assert peak == pytest.approx(261.626, abs=3)


def test_stereo_alignment_pan_and_peak():
    vocal = np.linspace(-0.3, 0.3, 100)
    synth = np.sin(np.arange(100))
    stereo, gains = stereo_comparison(vocal, synth)
    assert stereo.shape == (100, 2)
    assert stereo[:, 0] == pytest.approx(vocal * gains["vocal_gain"])
    assert stereo[:, 1] == pytest.approx(synth * gains["synth_gain"])
    assert np.abs(stereo).max() <= 0.85000001


def test_midi_exact_connected_ticks_and_positive_articulation_gap(tmp_path: Path):
    notes = [Note(0.123456, 0.5, 60, 100, "", ""),
             Note(0.5, 0.99, 62, 100, "", ""), Note(1, 1.5, 62, 100, "", "")]
    path = tmp_path / "notes.mid"
    export_midi(notes, path, 150)
    midi = mido.MidiFile(path)
    assert midi.ticks_per_beat == 9600
    assert midi.tracks[0][0].tempo == 400000
    tick, events = 0, []
    for message in midi.tracks[0]:
        tick += message.time
        if message.type in ("note_on", "note_off"):
            events.append((message.type, tick))
    assert events[1] == ("note_off", 12000)
    assert events[2] == ("note_on", 12000)
    assert events[3][1] < events[4][1]
