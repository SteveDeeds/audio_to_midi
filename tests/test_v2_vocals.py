from copy import deepcopy
from dataclasses import asdict, replace
from fractions import Fraction as F
from unittest.mock import patch

import mido
import numpy as np
import pytest
import soundfile as sf

from audio_to_midi import application as app
from audio_to_midi.adapters import Cancelled, Prediction, TrackBackend
from audio_to_midi.domain import Note, Project
from audio_to_midi.midi import export_midi
from audio_to_midi.preview import render_notes
from audio_to_midi.storage import load_project, save_project
from audio_to_midi.timebase import Grid
from audio_to_midi.vocal_settings import CONTROLS, VOCAL_MODELS, VocalParameters
from audio_to_midi.vocals import VocalEvidence, VocalTranscriber, grid_notes
from core.vocal_boundaries import BoundarySettings


def evidence():
    times = np.arange(400) / 100
    pitch = np.select([times < 1, times < 3], [60.0, 64.0], 67.0)
    db = np.full(400, -5.0)
    activation = np.zeros(400)
    activation[[0, 200]] = [0.9, 0.8]
    return VocalEvidence(times, pitch, db, times, activation, times, np.zeros(400))


@pytest.fixture
def project_track(tmp_path):
    path = tmp_path / "vocal.wav"
    sf.write(path, np.sin(np.arange(64000) / 10) * .1, 8000)
    project = Project.create("Vocals", Grid(120))
    track = app.add_stem(project, path, "vocals")
    track.model = "madmom_cnn"
    track.settings["madmom_license_acknowledged"] = True
    track.settings["vocal_parameters"] = asdict(VocalParameters())
    return project, track


@pytest.mark.parametrize("values", [None, [], {"neural_threshold": True}, {"min_note": float("nan")},
                                   {"pitch_hold": "0.09"}, {"voice_db": -70}, {"surprise": 1}, {1: 2}])
def test_invalid_vocal_controls_are_explicit(values):
    with pytest.raises(ValueError):
        VocalParameters.from_dict(values)


def test_all_five_controls_validate_their_endpoints():
    assert len(CONTROLS) == 5
    for name, _, low, high, *_ in CONTROLS:
        assert getattr(VocalParameters.from_dict({name: low}), name) == low
        assert getattr(VocalParameters.from_dict({name: high}), name) == high
        with pytest.raises(ValueError):
            VocalParameters.from_dict({name: high + 1})


@pytest.mark.parametrize("step", [F(1), F(1, 2), F(1, 3), F(1, 4), F(1, 6), F(1, 8)])
def test_articulated_syllables_have_exact_half_grid_gap_and_slurs_touch(step):
    notes, onsets = grid_notes(evidence(), np.array([0, 2]), Grid(120, subdivision=step),
                               F(0), F(8), BoundarySettings(), "v")
    assert [n.pitch for n in notes] == [60, 64, 64, 67]
    assert notes[0].end == notes[1].start == F(2)
    assert notes[1].end == notes[2].start - step / 2
    assert notes[2].end == notes[3].start == F(6)
    assert onsets == [F(0), F(4)]
    assert all(F(0) <= n.start < n.end <= F(8) for n in notes)


def test_gap_is_beats_not_constant_milliseconds_in_a_tempo_map():
    grid = Grid(subdivision=F(1, 3), anchors=((0, F(0)), (2, F(4)), (4, F(12))))
    notes, onsets = grid_notes(evidence(), np.array([0, 2]), grid, F(0), F(12), BoundarySettings(), "v")
    assert F(4) in onsets
    previous = next(n for n in notes if n.end == F(4) - F(1, 6))
    assert grid.beat_to_seconds(F(4)) - grid.beat_to_seconds(previous.end) == pytest.approx(1 / 12)
    assert any(a.end == b.start for a, b in zip(notes, notes[1:]))


def test_same_pitch_slur_merges_but_same_pitch_syllables_do_not():
    data = evidence()
    data.pitch[:] = np.where(data.times < 1, 60.0, 60.8)
    notes, onsets = grid_notes(data, np.array([0, 2]), Grid(120), F(0), F(8), BoundarySettings(), "v", "C")
    assert len(notes) == 2 and [n.pitch for n in notes] == [60, 60]
    assert notes[0].start == 0 and notes[1].start == 4
    assert notes[1].start - notes[0].end == F(1, 8)
    assert onsets == [F(0), F(4)]


def test_rejected_syllable_stays_connected_and_forced_attack_retriggers():
    arguments = (evidence(), np.array([0, 2]), Grid(120), F(0), F(8), BoundarySettings(), "v")
    notes, onsets = grid_notes(*arguments, labels={F(4): False})
    assert F(4) not in onsets
    assert notes[1].start == 2 and notes[1].end == 6
    forced, onsets = grid_notes(evidence(), np.array([0]), Grid(120), F(0), F(8), BoundarySettings(),
                                "v", labels={F(4): True})
    assert F(4) in onsets
    assert next(n for n in forced if n.start == 2).end == F(4) - F(1, 8)


def test_pitch_hold_and_min_split_reject_brief_changes():
    data = evidence()
    data.pitch[:] = 60
    data.pitch[100:108] = 64
    quick, _ = grid_notes(data, np.array([0]), Grid(120, subdivision=F(1, 8)), F(0), F(8),
                           BoundarySettings(pitch_hold=.04, min_note=.04), "v")
    held, _ = grid_notes(data, np.array([0]), Grid(120, subdivision=F(1, 8)), F(0), F(8),
                          BoundarySettings(pitch_hold=.16), "v")
    assert len(quick) > len(held) == 1


def test_main_provider_reuses_visible_evidence_when_controls_change():
    pytest.importorskip("madmom")
    provider = VocalTranscriber(np.ones(4000), 1000, "madmom_cnn", "")
    data = evidence()
    provider._cache[(0, 4000)] = data
    with patch.object(provider, "_pitch") as pitch, patch.object(provider, "_activation") as activation:
        first, _ = provider.transcribe("v", Grid(120), F(0), F(8), VocalParameters(), {}, lambda: False)
        second, _ = provider.transcribe("v", Grid(120), F(0), F(8),
                                        VocalParameters(neural_threshold=.85), {}, lambda: False)
    assert len(first) > len(second)
    pitch.assert_not_called()
    activation.assert_not_called()


def test_evidence_crop_and_pitch_context_never_read_outside_requested_audio():
    audio = np.zeros(22050 * 3, dtype=np.float32)
    audio[22050:44100] = 1
    provider = VocalTranscriber(audio, 22050, "madmom_cnn", "")

    def pitch(y, cancelled):
        assert len(y) == 22050 and np.all(y == 1)
        times = np.arange(100) / 100
        return times, np.full(100, 60)

    def activation(y):
        assert len(y) == 22050 and np.all(y == 1)
        return np.zeros(100)

    with patch.object(provider, "_pitch", side_effect=pitch) as pitch_call, \
            patch.object(provider, "_activation", side_effect=activation) as onset_call:
        first = provider.evidence(1, 2, lambda: False)
        assert provider.evidence(1, 2, lambda: False) is first
    assert pitch_call.call_count == onset_call.call_count == 1
    assert min(first.times) >= 1 and max(first.times) < 2


def test_cancelled_evidence_is_not_published_or_cached():
    provider = VocalTranscriber(np.ones(22050), 22050, "madmom_cnn", "")
    with pytest.raises(Cancelled):
        provider.evidence(0, 1, lambda: True)
    assert provider._cache == {}


def test_pitch_batches_keep_context_inside_visible_crop():
    pytest.importorskip("torch")
    provider = VocalTranscriber(np.ones(22050 * 3), 22050, "madmom_cnn", "")

    def pitch(y, sr, role, model):
        assert sr == 22050 and role == "vocals" and model == "crepe_full"
        times = np.arange(round(len(y) / sr * 100)) / 100
        return times, np.full(len(times), 60)

    with patch("audio_to_midi.vocals.track_pitch", side_effect=pitch) as calls:
        times, pitches = provider._pitch(provider.audio, lambda: False)
    assert [len(call.args[0]) for call in calls.call_args_list] == [22050 * 2 + 5512, 22050 + 5512]
    assert len(times) == len(pitches) == 300
    assert np.all(np.diff(times) > 0) and 0 <= times[0] < times[-1] < 3


def test_cancelling_after_first_pitch_batch_does_not_cache_partial_results():
    pytest.importorskip("torch")
    provider = VocalTranscriber(np.ones(22050 * 3), 22050, "madmom_cnn", "")
    cancelled = False

    def pitch(*args):
        nonlocal cancelled
        cancelled = True
        return np.arange(200) / 100, np.full(200, 60)

    with patch("audio_to_midi.vocals.track_pitch", side_effect=pitch) as calls:
        with pytest.raises(Cancelled):
            provider.evidence(0, 3, lambda: cancelled)
    assert calls.call_count == 1 and not provider._cache


@pytest.mark.parametrize("model", VOCAL_MODELS)
def test_actual_onset_models_resample_without_an_external_ffmpeg_dependency(model):
    pytest.importorskip("madmom")
    provider = VocalTranscriber(np.ones(2205, dtype=np.float32), 22050, model, "")
    with patch("madmom.io.audio.subprocess.Popen", side_effect=AssertionError("External ffmpeg requested")):
        activation = provider._activation(provider.audio)
    assert 9 <= len(activation) <= 11 and np.isfinite(activation).all()


def test_controls_and_models_round_trip_without_rewriting_saved_notes(project_track, tmp_path):
    project, track = project_track
    track.notes.generated = [Note("old", F(10), F(11), 60)]
    track.settings["vocal_parameters"] = asdict(VocalParameters(neural_threshold=.55, pitch_hold=.12))
    for model in VOCAL_MODELS:
        track.model = model
        save_project(project, tmp_path / model)
        loaded = load_project(tmp_path / model).track(track.id)
        assert loaded.model == model
        assert loaded.settings["vocal_parameters"] == track.settings["vocal_parameters"]
        assert loaded.notes.generated == track.notes.generated


def test_main_publication_never_changes_offscreen_fragments_or_onsets(project_track):
    project, track = project_track
    track.notes.generated = [Note("spanning", F(0), F(16), 60)]
    track.notes.onsets = [F(0), F(2), F(12), F(14)]
    before = deepcopy(track.notes)
    prediction = Prediction(F(4), F(12), (Note("new", F(4), F(12), 64),), (F(4), F(8)))
    assert app.publish(project, app.token(project, track), prediction)
    for a, b in [(F(0), F(4)), (F(12), F(16))]:
        prior = [replace(n, start=max(n.start, a), end=min(n.end, b))
                 for n in before.resolved() if n.start < b and a < n.end]
        after = [replace(n, start=max(n.start, a), end=min(n.end, b))
                 for n in track.notes.resolved() if n.start < b and a < n.end]
        assert before.onsets == [F(0), F(2), F(12), F(14)]
        assert prior == after
    assert track.notes.onsets == [F(0), F(2), F(4), F(12), F(14)]


def test_control_change_invalidates_pending_result_but_not_hidden_notes(project_track):
    project, track = project_track
    before = app.token(project, track)
    track.notes.generated = [Note("offscreen", F(10), F(12), 60)]
    track.settings["vocal_parameters"] = asdict(VocalParameters(pitch_hold=.15))
    prediction = Prediction(F(0), F(8), (Note("new", F(0), F(4), 62),), ())
    assert not app.publish(project, before, prediction)
    assert track.notes.generated == [Note("offscreen", F(10), F(12), 60)]


def test_crepe_license_flag_cannot_acknowledge_madmom_license(project_track):
    project, track = project_track
    track.settings["madmom_license_acknowledged"] = False
    track.settings["license_acknowledged"] = True
    backend = TrackBackend.create(track, project)
    with pytest.raises(ValueError, match="license"):
        backend.predict(track, F(0), F(8), lambda: False)


def test_authored_vocal_note_blocks_generated_alternatives_of_any_pitch(project_track):
    project, track = project_track
    track.notes.author(F(1), F(3), 60)
    backend = TrackBackend.create(track, project)

    class Provider:
        def transcribe(self, *args):
            return [Note("g", F(0), F(8), 64)], [F(0)]

    backend.provider = Provider()
    result = backend.predict(track, F(0), F(8), lambda: False)
    assert [(n.start, n.end) for n in result.notes] == [(F(0), F(1)), (F(3), F(8))]
    assert track.notes.authored[0].pitch == 60


def test_midi_keeps_half_grid_gap_and_exact_slur_ticks(project_track, tmp_path):
    project, track = project_track
    notes, _ = grid_notes(evidence(), np.array([0, 2]), project.grid, F(0), F(8), BoundarySettings(), track.id)
    track.notes.generated = notes
    path = tmp_path / "vocal.mid"
    export_midi(project, path, tracks=[track], include_chords=False)
    midi = mido.MidiFile(path)
    events, tick = [], 0
    for event in midi.tracks[1]:
        tick += event.time
        if event.type in ("note_on", "note_off"):
            events.append((tick, event.type, event.note))
    assert events == sorted([(round(n.start * midi.ticks_per_beat), "note_on", n.pitch) for n in notes] +
                            [(round(n.end * midi.ticks_per_beat), "note_off", n.pitch) for n in notes],
                            key=lambda row: (row[0], row[1] == "note_on"))


def test_preview_sustains_connected_pitches_and_retriggers_after_gap():
    sr = 16000
    rows = [(0, .4, 60, 100), (.4, .8, 64, 100)]
    slur = render_notes(rows, 0, sr, sr, legato=True)
    separate = render_notes(rows, 0, sr, sr)
    boundary = round(.4 * sr)
    phase = 2 * np.pi * (440 * 2 ** ((60 - 69) / 12)) * .4
    expected = .3 * 100 / 127 * sum(weight * np.sin(k * phase)
                                   for k, weight in [(1, 1), (2, .3), (3, .15)])
    assert slur[boundary] == pytest.approx(expected, abs=1e-7)
    assert separate[boundary] == 0
    gapped = render_notes([(0, .35, 60, 100), (.4, .8, 64, 100)], 0, sr, sr, legato=True)
    assert not np.any(gapped[round(.35 * sr):boundary])
    assert gapped[boundary] == 0


def test_legato_option_does_not_change_drum_rendering():
    rows = [(0, .4, 36, 100), (.4, .8, 38, 100)]
    assert np.array_equal(render_notes(rows, 0, 16000, 16000, drums=True),
                          render_notes(rows, 0, 16000, 16000, drums=True, legato=True))
