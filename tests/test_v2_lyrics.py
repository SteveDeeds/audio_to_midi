from types import SimpleNamespace
from unittest.mock import Mock, patch
import sys

import numpy as np
import pytest
import soundfile as sf

from audio_to_midi.lyrics import analyze_lyrics, clean_lyrics, save_analysis, word_result
from audio_to_midi.vocal_words import annotation_from_analysis, note_key, note_words, validate_words


def result(words):
    return SimpleNamespace(to_dict=lambda: {"segments": [{"words": words}]})


def word(text="hello", start=0.1, end=0.5):
    return dict(word=text, start=start, end=end, probability=.9)


def test_section_headings_are_not_aligned_but_backing_vocals_are():
    assert clean_lyrics("[Verse 1]\nhello\nChorus:\n(world)\nBridge\nOutro") == "hello\n(world)"
    assert clean_lyrics("Chorus of voices") == "Chorus of voices"
    with pytest.raises(ValueError, match="no words"):
        clean_lyrics("[Intro]\n\n")


def test_unaligned_and_missing_words_remain_explicit():
    data = word_result(result([word(), word("world", .5, .5)]), 1, "known_lyrics_alignment", "hello world again")
    assert data["unaligned_words"] == 1
    assert not data["words"][1]["aligned"]
    assert any("incomplete" in message for message in data["warnings"])


@pytest.mark.parametrize("words", [
    [word(start=float("nan"))], [word(start=-1)], [word(end=2)],
    [word(start=.7, end=.6)], [word(), word(start=.05)],
])
def test_invalid_word_times_are_errors(words):
    with pytest.raises(ValueError, match="timestamps"):
        word_result(result(words), 1, "automatic_recognition")


def test_empty_alignment_is_not_success_shaped():
    with pytest.raises(RuntimeError, match="failed"):
        word_result(None, 1, "alignment")
    with pytest.raises(RuntimeError, match="no words"):
        word_result(result([]), 1, "alignment")


def test_recognition_is_independent_of_supplied_lyrics_and_needs_no_ffmpeg(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    path = tmp_path / "vocals.wav"
    sf.write(path, np.full((8000, 2), .1), 8000)
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_text("[Verse 1]\nhello world", encoding="utf-8")
    network = SimpleNamespace(
        transcribe=Mock(return_value=result([word("different")])),
        align=Mock(return_value=result([word(), word("world", .5, .9)])),
    )
    load = Mock(return_value=network)
    monkeypatch.setitem(sys.modules, "stable_whisper", SimpleNamespace(load_model=load))
    with patch("audio_to_midi.lyrics.version", return_value="test"), \
            pytest.warns(UserWarning):
        data = analyze_lyrics(path, lyrics_path=lyrics, transcribe=True, report=lambda _: None)
    args, kwargs = network.transcribe.call_args
    assert args[0].shape == (16000,) and args[0].dtype == np.float32
    assert "initial_prompt" not in kwargs
    assert network.align.call_args.args[1] == "hello world"
    assert data["analyses"]["recognized"]["text"] == "different"
    assert data["analyses"]["aligned"]["mode"] == "known_lyrics_alignment"
    assert data["lyrics_sha256"]
    output = tmp_path / "words.json"
    save_analysis(data, output)
    with pytest.raises(FileExistsError):
        save_analysis(data, output)


def annotation(words):
    return dict(model="small.en", mode="automatic_recognition", audio_sha256="a" * 64, words=words)


def test_words_map_once_by_overlap_with_melismas_repeats_and_multiple_words():
    from fractions import Fraction as F
    from audio_to_midi.domain import Note
    from audio_to_midi.timebase import Grid

    notes = [Note("a", F(0), F(1), 60), Note("b", F(1), F(3), 64), Note("c", F(4), F(6), 62)]
    data = annotation([word("long", .3, 1.4), word("word", 1.4, 1.5),
                       word("again", 2.1, 2.4), word("lost", 3.2, 3.5), word("instant", 4, 4)])
    labels = note_words(notes, Grid(120), data)
    assert labels == {note_key(notes[1]): "long word", note_key(notes[2]): "again"}
    # A viewport containing only the later note cannot steal words from an earlier note.
    assert len(labels) == 2
    assert note_words(notes, Grid(60), data)[note_key(notes[0])] == "long"


@pytest.mark.parametrize("changes", [
    {"start": True}, {"start": -1}, {"end": float("nan")},
    {"end": 10}, {"word": ""}, {"probability": 2},
])
def test_persisted_word_annotations_validate(changes):
    item = word()
    item.update(changes)
    with pytest.raises(ValueError):
        validate_words(annotation([item]), 2)


def test_import_requires_matching_audio_and_copies_selected_analysis():
    data = dict(schema=1, model="small.en", audio_sha256="a" * 64,
                analyses={"recognized": {"mode": "automatic_recognition", "words": [word()]}})
    imported = annotation_from_analysis(data, "recognized", "a" * 64, 1)
    data["analyses"]["recognized"]["words"][0]["word"] = "changed"
    assert imported["words"][0]["word"] == "hello"
    with pytest.raises(ValueError, match="match"):
        annotation_from_analysis(data, "recognized", "b" * 64, 1)
    with pytest.raises(ValueError, match="available"):
        annotation_from_analysis(data, "aligned", "a" * 64, 1)


def test_saved_words_round_trip_and_replacement_clears_them(tmp_path):
    from audio_to_midi.application import add_stem, replace_stem
    from audio_to_midi.domain import Project
    from audio_to_midi.storage import load_project, save_project
    from audio_to_midi.timebase import Grid

    audio = tmp_path / "vocals.wav"
    sf.write(audio, np.ones(8000), 8000)
    project = Project.create("Words", Grid())
    track = add_stem(project, audio, "vocals")
    track.settings["vocal_words"] = annotation([word()])
    save_project(project, tmp_path / "project")
    loaded = load_project(tmp_path / "project")
    assert loaded.track(track.id).settings["vocal_words"] == track.settings["vocal_words"]
    replace_stem(loaded, loaded.track(track.id), audio)
    assert "vocal_words" not in loaded.track(track.id).settings
