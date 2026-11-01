from dataclasses import asdict, replace
from http import HTTPStatus
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import mido
import numpy as np
import pytest
import soundfile as sf

from experimental.tuner_server import TunerServer, byte_range
from experimental.tuning import (
    CONTROL_SPECS, GUI_CONTROL_SPECS, Evidence, TuningParameters, load_evidence,
    pick_cached, retune, tuning_defaults,
)
from experimental.vocal_midi import (
    Detector, Note, NoteSettings, articulation_release, make_notes, stable_changes,
    stereo_comparison, synthesize,
)
from experimental.parameter_sweep import measure


def contour():
    times = np.arange(200) / 100
    pitch = np.where(times < 1, 60.0, 64.0)
    db = np.full(200, -5.0)
    detector = Detector(np.array([0.0]), times, np.zeros(200), {})
    return times, pitch, db, detector


def fake_evidence(tmp_path: Path) -> Evidence:
    times, pitch, db, _ = contour()
    env = np.zeros(200)
    env[[5, 50, 120]] = [0.4, 0.8, 0.9]
    arrays = {"times": times, "pitch": pitch, "db": db,
              "periodicity": np.full(200, 0.8), "tracker_gate": np.ones(200, dtype=bool),
              "vocal": 0.1 * np.sin(np.arange(2000) * 0.3),
              "cqt_db": np.full((144, 200), -40.0), "cqt_times": times}
    for name in ("librosa", "superflux", "madmom_cnn", "madmom_brnn"):
        arrays[f"{name}_times"] = times
        arrays[f"{name}_envelope"] = env.copy()
    metadata = {"duration_seconds": 2, "sample_rate": 1000, "start_seconds": 0,
                "end_seconds": 2, "pitch_low": 48, "pitch_high": 76}
    return Evidence(metadata, arrays, tmp_path)


def test_pitch_splits_fix_long_onset_blocks():
    times, pitch, db, detector = contour()
    baseline = make_notes(detector, times, pitch, db, [(0, 2)])
    enhanced = make_notes(detector, times, pitch, db, [(0, 2)], NoteSettings(split_pitch=True))
    assert len(baseline) == 1
    assert [n.midi for n in enhanced] == [60, 64]
    assert enhanced[0].end == enhanced[1].start == 1
    assert enhanced[0].link_to_next == "slurred"
    assert enhanced[1].start_evidence == "pitch change"


def test_pitch_certainty_rejects_mixed_notes_but_accepts_split_notes():
    times, pitch, db, detector = contour()
    strict = NoteSettings(min_pitch_agreement=0.8)
    assert not make_notes(detector, times, pitch, db, [(0, 2)], strict)
    split = make_notes(detector, times, pitch, db, [(0, 2)], replace(strict, split_pitch=True))
    assert len(split) == 2 and all(n.pitch_agreement == 1 for n in split)


def test_pitch_hold_and_hysteresis_are_effective():
    times, pitch, _, _ = contour()
    pitch[30:35] = 62
    assert len(stable_changes(times, pitch)) == 1
    assert len(stable_changes(times, pitch, NoteSettings(pitch_hold=0.03))) == 3
    pitch[:] = 60
    pitch[100:] = 61
    assert len(stable_changes(times, pitch, NoteSettings(pitch_hysteresis=0.5))) == 1
    assert not len(stable_changes(times, pitch, NoteSettings(pitch_hysteresis=1.5)))


def test_key_assignment_is_optional_limited_and_uses_raw_pitch():
    times, pitch, db, detector = contour()
    pitch[:] = 66.2
    uncorrected = make_notes(detector, times, pitch, db, [(0, 2)])
    corrected = make_notes(detector, times, pitch, db, [(0, 2)], NoteSettings(key="C"))
    limited = make_notes(detector, times, pitch, db, [(0, 2)], NoteSettings(key="C", key_limit=0))
    assert uncorrected[0].midi == limited[0].midi == 66
    assert corrected[0].midi == 67 and corrected[0].key_adjusted
    assert corrected[0].raw_midi == pytest.approx(66.2)
    assert corrected[0].pitch_agreement == 1


def test_articulation_parameters_change_release_not_fixed_fake_gap():
    times, pitch, db, _ = contour()
    db[95:99] = -15
    assert articulation_release(0, 1, times, pitch, db)[0] == 0.95
    assert articulation_release(0, 1, times, pitch, db, NoteSettings(articulation_dip=15))[0] == 1
    assert articulation_release(0, 1, times, pitch, db, NoteSettings(articulation_gap=0.1))[0] == 1


@pytest.mark.parametrize("data", [
    {"periodicity": float("nan")}, {"periodicity": -1}, {"periodicity": "0.3"},
    {"periodicity": True}, {"min_note": 0}, {"split_pitch": 1}, {"key": "H"},
    {"method": "fake"}, {"method": []}, {"key": []}, {"surprise": 1}, {"syllabic": 1},
])
def test_invalid_controls_are_explicit_errors(data):
    with pytest.raises(ValueError):
        TuningParameters.from_dict(data)


def test_every_numeric_control_has_an_effective_parameter():
    parameters = TuningParameters.from_dict({})
    for name, _, low, high, _, _, _, _ in CONTROL_SPECS:
        assert low <= getattr(parameters, name) <= high
        updated = TuningParameters.from_dict({name: high})
        assert getattr(updated, name) == high


def test_cached_onset_threshold_and_spacing_are_effective(tmp_path):
    evidence = fake_evidence(tmp_path)
    default = TuningParameters()
    lower = pick_cached(evidence, "madmom_cnn", replace(default, neural_threshold=0.3))
    higher = pick_cached(evidence, "madmom_cnn", replace(default, neural_threshold=0.85))
    assert len(lower.times) == 3 and len(higher.times) == 1
    spaced = pick_cached(evidence, "madmom_cnn", replace(default, onset_wait=0.4))
    evidence.arrays["madmom_cnn_envelope"][20] = 0.9
    spaced = pick_cached(evidence, "madmom_cnn", replace(default, onset_wait=0.4))
    assert len(spaced.times) < len(pick_cached(evidence, "madmom_cnn", default).times)
    assert len(pick_cached(evidence, "superflux", replace(default, flux_delta=0.7)).times) < len(
        pick_cached(evidence, "superflux", replace(default, flux_delta=0.05)).times,
    )


def test_periodicity_gate_and_minimum_note_length_change_output(tmp_path):
    evidence = fake_evidence(tmp_path)
    evidence.arrays["periodicity"][70:130] = 0.2
    normal, _, _ = retune(evidence, TuningParameters())
    lower, _, _ = retune(evidence, TuningParameters(periodicity=0.1))
    assert normal["stats"]["voiced_fraction"] < lower["stats"]["voiced_fraction"]
    none, _, _ = retune(evidence, TuningParameters(periodicity=0.95))
    assert not none["notes"]
    short, _, _ = retune(evidence, TuningParameters(min_note=0.04))
    longer, _, _ = retune(evidence, TuningParameters(min_note=0.4))
    assert len(longer["notes"]) <= len(short["notes"])


@pytest.fixture
def api(tmp_path):
    evidence = fake_evidence(tmp_path)
    server = TunerServer(("127.0.0.1", 0), evidence)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def request(url, values=None, token=None):
    if values is None:
        return urlopen(url)
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Tuner-Token"] = token
    return urlopen(Request(url, data=json.dumps(values).encode(), headers=headers))


def test_api_updates_and_reports_invalid_parameters(api):
    server, url = api
    with request(url + "/api/state") as response:
        state = json.load(response)
    assert state["defaults"]["split_pitch"]
    with request(url + "/api/tune", {"neural_threshold": 0.85}, state["token"]) as response:
        result = json.load(response)
    assert len(result["onsets"]) == 1
    with pytest.raises(HTTPError) as error:
        request(url + "/api/tune", {"periodicity": 2}, server.token)
    assert error.value.code == HTTPStatus.BAD_REQUEST
    assert "periodicity" in json.load(error.value)["error"]
    with pytest.raises(HTTPError) as error:
        request(url + "/api/tune", {})
    assert error.value.code == HTTPStatus.FORBIDDEN


@pytest.mark.parametrize(("header", "expected"), [
    (None, (0, 99, 200)), ("bytes=10-19", (10, 19, 206)),
    ("bytes=10-", (10, 99, 206)), ("bytes=-20", (80, 99, 206)),
    ("bytes=10-999", (10, 99, 206)), ("bytes=-999", (0, 99, 206)),
])
def test_media_byte_ranges(header, expected):
    assert byte_range(header, 100) == expected


@pytest.mark.parametrize("header", ["bytes=-0", "bytes=100-", "bytes=2-1",
                                  "bytes=0-1,5-6", "bytes=-", "bytes=foo"])
def test_invalid_media_ranges_are_explicit(header):
    with pytest.raises(ValueError):
        byte_range(header, 100)


@pytest.mark.parametrize("syllabic", [False, True])
def test_api_preview_and_export_match_current_notes(api, syllabic):
    server, url = api
    values = asdict(TuningParameters(split_pitch=True, key="C", syllabic=syllabic))
    with request(url + "/api/preview", values, server.token) as response:
        preview = json.load(response)
    audio, sr = sf.read(server.work / preview["url"].split("/")[-1])
    _, notes, _ = retune(server.evidence, TuningParameters.from_dict(values))
    expected, _ = stereo_comparison(server.evidence.arrays["vocal"], synthesize(notes, sr, len(audio)))
    assert audio.shape == (2000, 2) and np.max(np.abs(audio - expected)) < 2e-7
    with request(url + "/api/export", values, server.token) as response:
        exported = json.load(response)
    directory = Path(exported["directory"])
    preset = json.loads((directory / "settings.json").read_text())
    assert preset["parameters"] == values
    assert preset["notes"] == [asdict(n) for n in notes]
    midi = mido.MidiFile(directory / "notes.mid")
    assert sum(m.type == "note_on" for m in midi.tracks[0]) == len(notes)
    tick = 0
    events = []
    for event in midi.tracks[0]:
        tick += event.time
        if event.type in ("note_on", "note_off"):
            events.append((tick, event.type, event.note))
    expected_events = sorted(
        [(round(n.start * 150 / 60 * 9600), "note_on", n.midi) for n in notes] +
        [(round(n.end * 150 / 60 * 9600), "note_off", n.midi) for n in notes],
        key=lambda event: (event[0], event[1] == "note_on"),
    )
    assert events == expected_events
    assert (directory / "madmom_cnn_cqt.png").is_file()
    assert (directory / "AB.wav").is_file()
    with request(url + exported["files"]["notes.mid"]) as response:
        assert response.read() == (directory / "notes.mid").read_bytes()
    with urlopen(Request(url + exported["files"]["notes.mid"], headers={"Range": "bytes=5-14"})) as response:
        assert response.status == 206
        assert response.headers["Content-Range"].startswith("bytes 5-14/")
        assert response.read() == (directory / "notes.mid").read_bytes()[5:15]


def test_real_cache_reproduces_all_five_original_baselines():
    base = Path(__file__).parent / "outputs"
    cache = base / "tuner_cache_39_to_end"
    if not (cache / "evidence.json").exists():
        pytest.skip("Prepare the real vocal cache to run this integration check.")
    evidence = load_evidence(cache)
    original = json.loads((base / "vocals_39_to_end" / "report.json").read_text())
    for method in original["methods"]:
        result, notes, _ = retune(evidence, TuningParameters(method=method))
        previous = original["methods"][method]
        assert result["stats"]["raw_onsets"] == previous["raw_onsets"], method
        assert len(notes) == previous["notes"], method
        for current, old in zip(notes, previous["notes_detail"]):
            assert current.start == pytest.approx(old["start"], abs=1e-9)
            assert current.end == pytest.approx(old["end"], abs=1e-9)
            assert current.midi == old["midi"]
            assert current.link_to_next == old["link_to_next"]


def test_five_controls_per_model_keep_hidden_values_validated():
    for neural in (False, True):
        visible = [s for s in GUI_CONTROL_SPECS
                   if s[6] == "all" or s[6] == ("neural" if neural else "flux")]
        assert len(visible) == 5
    values = asdict(tuning_defaults())
    values.update(key_limit=0, periodicity=0.9, pitch_hysteresis=1.25)
    assert asdict(TuningParameters.from_dict(values)) == values
    with pytest.raises(ValueError, match="pitch_hysteresis"):
        TuningParameters.from_dict({**values, "pitch_hysteresis": 3})


def test_sweep_records_exact_settings_and_measurable_shapes(tmp_path):
    evidence = fake_evidence(tmp_path)
    parameters = tuning_defaults()
    row = measure(evidence, parameters)
    assert row["parameters"] == asdict(parameters)
    assert row["notes"] == row["syllable_starts"] + row["pitch_only_starts"]
    assert 0 <= row["mean_pitch_agreement"] <= 1
    assert row["minimum_duration_ms"] >= parameters.min_note * 1000 - 1e-6
    assert row["tied"] == 0
    empty = measure(evidence, replace(parameters, periodicity=0.95))
    assert empty["notes"] == 0
    assert empty["mean_pitch_agreement"] is None
    assert empty["minimum_duration_ms"] is None


def test_syllable_attack_retriggers_same_pitch_without_acoustic_gap():
    times, pitch, db, detector = contour()
    pitch[:] = 60
    detector.times = np.array([0, 1.0])
    notes = make_notes(detector, times, pitch, db, [(0, 2)], NoteSettings(syllabic=True))
    assert len(notes) == 2
    assert notes[0].end == 0.99 and notes[1].start == 1
    assert notes[0].link_to_next == "articulated"
    assert notes[0].gap_to_next_ms == 10
    assert "rendering gap" in notes[0].end_evidence


def test_pitch_only_split_stays_slurred_despite_energy_dip():
    times, pitch, db, detector = contour()
    db[95:99] = -15
    notes = make_notes(detector, times, pitch, db, [(0, 2)], NoteSettings(syllabic=True))
    assert [n.midi for n in notes] == [60, 64]
    assert notes[0].end == notes[1].start == 1
    assert notes[0].link_to_next == "slurred"
    assert notes[0].end_evidence == "within-syllable pitch change"


def test_nearby_syllable_attack_takes_priority_over_pitch_proposal():
    times, pitch, db, detector = contour()
    detector.times = np.array([0, 1.04])
    notes = make_notes(detector, times, pitch, db, [(0, 2)], NoteSettings(syllabic=True))
    assert len(notes) == 2
    assert [n.midi for n in notes] == [60, 64]
    assert notes[1].start == 1.04
    assert notes[1].start_evidence == "detector"
    assert notes[0].link_to_next == "articulated"


def test_same_pitch_legato_segments_merge_and_remeasure_after_key_correction():
    times, pitch, db, detector = contour()
    pitch[times >= 1] = 60.8
    notes = make_notes(detector, times, pitch, db, [(0, 2)], NoteSettings(syllabic=True, key="C"))
    assert len(notes) == 1 and notes[0].midi == 60
    assert notes[0].start == 0 and notes[0].end == 2
    assert notes[0].raw_midi == pytest.approx(60.8)
    assert notes[0].pitch_agreement == pytest.approx(100 / 196)
    assert notes[0].key_adjusted


def test_energy_backed_extra_syllable_is_separate_from_slur(tmp_path):
    evidence = fake_evidence(tmp_path)
    evidence.arrays["pitch"][:] = 60
    evidence.arrays["db"][95:99] = -12
    evidence.arrays["madmom_cnn_envelope"][[50, 120]] = 0
    evidence.arrays["superflux_envelope"][:] = 0
    evidence.arrays["superflux_envelope"][100] = 0.9
    result, notes, _ = retune(evidence, tuning_defaults())
    assert result["stats"]["extra_syllable_proposals"] == 1
    assert len(notes) == 2
    assert notes[1].start_evidence == "SuperFlux + energy recovery"
    assert notes[0].link_to_next == "articulated"
    assert notes[1].start == pytest.approx(result["energy_onsets"][0])
    assert notes[0].end < notes[1].start


def test_energy_fallback_does_not_duplicate_neural_attack(tmp_path):
    evidence = fake_evidence(tmp_path)
    evidence.arrays["db"][45:49] = -12
    result, _, _ = retune(evidence, tuning_defaults())
    assert not result["energy_onsets"]


@pytest.mark.parametrize("method", ["librosa", "superflux", "madmom_cnn", "madmom_brnn", "pitch_aware"])
def test_all_methods_keep_syllable_and_slur_timing_consistent(tmp_path, method):
    result, notes, _ = retune(fake_evidence(tmp_path), replace(tuning_defaults(), method=method))
    assert notes
    for previous, following in zip(notes, notes[1:]):
        assert previous.end <= following.start
        assert previous.end > previous.start
        if following.start_evidence == "pitch change":
            if previous.end == following.start:
                assert previous.link_to_next == "slurred"
                assert previous.midi != following.midi
        else:
            assert following.start - previous.end >= 0.01 - 1e-9
            assert previous.link_to_next == "articulated"
    assert result["stats"]["links"]["tied"] == 0
