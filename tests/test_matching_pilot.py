from copy import deepcopy
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import mido
import numpy as np
import pytest
import soundfile as sf

from experimental.benchmark import matching_pilot as pilot


def tone(pitch, rate=22050, seconds=.25):
    time = np.arange(round(rate * seconds)) / rate
    frequency = 440 * 2 ** ((pitch - 69) / 12)
    audio = .1 * np.sin(2 * np.pi * frequency * time)
    audio += .035 * np.sin(2 * np.pi * frequency * 2 * time)
    return np.column_stack((audio, -audio))


def evidence_fixture():
    guitar, piano = tone(60), tone(73)
    clean = {"guitar": guitar, "piano": piano}
    stems = {stem: np.zeros_like(guitar) for stem in pilot.STEMS}
    stems["bass"] = guitar * .8
    stems["piano"] = piano
    return clean, stems


def test_pitch_evidence_finds_misrouted_note_not_louder_unrelated_energy():
    clean, stems = evidence_fixture()
    stems["piano"] *= 8
    evidence = pilot.spectral_evidence(clean, stems, "guitar", 60, 22050)
    candidates, reasons = pilot.propose(evidence, False)
    assert candidates == ["bass"]
    assert evidence["stems"]["bass"]["score"] > .7
    assert evidence["stems"]["piano"]["score"] < .1
    assert "weak_spectral_evidence" not in reasons


def test_multi_stem_leakage_is_not_forced_into_one_destination():
    clean, stems = evidence_fixture()
    stems["guitar"] = clean["guitar"] * .7
    evidence = pilot.spectral_evidence(clean, stems, "guitar", 60, 22050)
    candidates, reasons = pilot.propose(evidence, False)
    assert set(candidates) == {"guitar", "bass"}
    assert "possible_multi_stem_leakage" in reasons


def test_stereo_antiphase_and_short_windows_keep_real_energy():
    audio = tone(60)[:200]
    assert np.max(np.abs(audio.mean(axis=1))) == 0
    assert pilot.spectrum(audio, 22050).max() > 0


def test_absent_audio_produces_no_suggestion_not_a_fake_label():
    clean, stems = evidence_fixture()
    stems = {stem: np.zeros_like(clean["guitar"]) for stem in pilot.STEMS}
    candidates, reasons = pilot.propose(
        pilot.spectral_evidence(clean, stems, "guitar", 60, 22050), True)
    assert candidates == []
    assert "weak_spectral_evidence" in reasons
    assert "overlapping_same_family_notes_or_release" in reasons


def test_confusable_families_flagged_instead_of_claiming_certainty():
    clean, stems = evidence_fixture()
    clean["piano"] = clean["guitar"].copy()
    _, reasons = pilot.propose(
        pilot.spectral_evidence(clean, stems, "guitar", 60, 22050), False)
    assert "confusable_clean_families" in reasons


def test_silent_clean_reference_remains_uncertain():
    clean, stems = evidence_fixture()
    clean["guitar"][:] = 0
    candidates, reasons = pilot.propose(
        pilot.spectral_evidence(clean, stems, "guitar", 60, 22050), False)
    assert not candidates
    assert "quiet_clean_reference" in reasons


@pytest.mark.parametrize("audio", [np.empty((0, 2)), np.ones((10,)), np.full((10, 2), np.nan)])
def test_invalid_audio_fails_explicitly(audio):
    with pytest.raises(ValueError, match="Audio window"):
        pilot.spectrum(audio, 22050)


def record(group, split, variant="base", flagged=False):
    return {"source_group": group, "split": split, "composition_id": f"{group}_{variant}",
            "flagged": {"guitar": flagged}}


def test_selection_uses_distinct_original_training_sources_only():
    records = [record("easy", "train"), record("hard", "train", flagged=True),
               record("easy", "train", "aug"), record("validation", "validation"),
               record("test", "test")]
    selected = pilot.select_records(records, 2)
    assert {item["composition_id"] for item in selected} == {"easy_base", "hard_base"}
    with pytest.raises(ValueError, match="Only 2"):
        pilot.select_records(records, 3)
    with pytest.raises(ValueError, match="positive"):
        pilot.select_records(records, 0)
    records.append(record("easy", "test", "aug"))
    with pytest.raises(ValueError, match="leakage"):
        pilot.select_records(records, 1)


def make_report(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "pairs.json").write_text("[]")
    wav = dataset / "audio.wav"
    sf.write(wav, tone(60, seconds=1), 22050, subtype="FLOAT")
    midi = dataset / "guitar.mid"
    file = mido.MidiFile()
    track = mido.MidiTrack()
    track.extend([mido.Message("note_on", note=60, velocity=100, time=0),
                  mido.Message("note_off", note=60, time=240)])
    file.tracks.append(track)
    file.save(midi)
    report = {
        "schema": pilot.SCHEMA, "code_sha256": pilot.code_hash(),
        "dataset": str(dataset), "dataset_sha256": {"pairs.json": pilot.digest(dataset / "pairs.json")},
        "status": "ready_for_review", "split": "train", "summary": {"notes": 3, "source_groups": 1},
        "records": [{
            "clip": "source", "composition_id": "source_base",
            "source_group": "group", "split": "train", "duration": 1., "sample_rate": 22050,
            "license": ["Original test fixture"], "provenance": ["Synthetic test"],
            "source_midi": {"guitar": str(midi)},
            "source_midi_sha256": {"guitar": pilot.digest(midi)},
            "audio": {"guitar": str(wav)}, "audio_sha256": {"guitar": pilot.digest(wav)},
            "notes": [
                {"id": "guitar", "family": "guitar", "onset": 0., "offset": .2,
                 "pitch": 60, "channel": 0, "velocity": 100},
                {"id": "drums", "family": "drums", "onset": .1, "offset": .2,
                 "pitch": 35, "channel": 9, "velocity": 80},
                {"id": "unknown", "family": "piano", "onset": .4, "offset": .7,
                 "pitch": 72, "channel": 1, "velocity": 60},
            ],
        }],
    }
    item = report["records"][0]
    item["audio"] = {name: str(wav) for name in (
        "mix", *pilot.STEMS, "clean_guitar", "clean_piano", "clean_drums", "clean_bass")}
    item["audio_sha256"] = {name: pilot.digest(wav) for name in item["audio"]}
    for note in item["notes"]:
        note.update({
            "legacy_stem": "guitar", "suggested_stems": ["guitar"], "review_reasons": [],
            "evidence": {"clean_rms_dbfs": -20, "stems": {
                stem: {"score": .8 if stem == "guitar" else 0, "spectral_similarity": .8,
                       "estimated_retention": .8, "fit_residual": .1}
                for stem in pilot.STEMS}},
            "model_support": {stem: [] for stem in pilot.STEMS},
        })
    output = tmp_path / "pilot"
    output.mkdir()
    pilot.write_json(output / "report.json", report)
    pilot.write_json(output / "reviews.json", pilot.empty_reviews(report))
    (output / "index.html").write_text("<h1>Review</h1>")
    (output / "review.js").write_text("")
    return output, report


def test_no_export_without_human_confirmation(tmp_path):
    output, report = make_report(tmp_path)
    with pytest.raises(ValueError, match="No human-confirmed"):
        pilot.export_supervision(output, tmp_path / "labels.json")
    assert not (tmp_path / "labels.json").exists()


def test_export_keeps_multiple_destinations_and_masks_all_unknown_negatives(tmp_path):
    output, report = make_report(tmp_path)
    reviews = pilot.empty_reviews(report)
    reviews["decisions"] = {
        "guitar": {"status": "confirmed", "stems": ["bass", "guitar"], "scope": "exact_note"},
        "drums": {"status": "confirmed", "stems": ["other"], "scope": "exact_note"},
        "unknown": {"status": "uncertain", "stems": []},
    }
    pilot.write_json(output / "reviews.json", reviews)
    destination = tmp_path / "labels.json"
    data = pilot.export_supervision(output, destination)
    assert data["confirmed_notes"] == 2
    assert data["positive_stem_assignments"] == 3
    assert data["negative_supervision_allowed"] is False
    targets = data["records"][0]["targets"]
    assert targets["bass"]["pitched_onsets"][0]["note_id"] == "guitar"
    assert targets["other"]["drum_onsets"][0]["class_index"] == 0
    assert "piano" not in targets
    assert data["source_offsets_are_acoustic_targets"] is False
    with pytest.raises(ValueError, match="rights"):
        pilot.load_positive_supervision(destination)
    assert pilot.load_positive_supervision(destination, rights_cleared=True) == data
    with pytest.raises(FileExistsError):
        pilot.export_supervision(output, destination)


@pytest.mark.parametrize("decision", [
    {"status": "confirmed", "stems": []},
    {"status": "confirmed", "stems": ["guitar", "guitar"]},
    {"status": "confirmed", "stems": ["invented"]},
    {"status": "absent", "stems": ["guitar"]},
    {"status": "uncertain", "stems": ["bass"]},
    {"status": "auto_confirmed", "stems": ["bass"]},
])
def test_invalid_review_decisions_rejected(tmp_path, decision):
    _, report = make_report(tmp_path)
    reviews = pilot.empty_reviews(report)
    reviews["decisions"]["guitar"] = decision
    with pytest.raises(ValueError):
        pilot.validate_reviews(report, reviews)


def test_stale_unknown_and_nontraining_reviews_rejected(tmp_path):
    output, report = make_report(tmp_path)
    reviews = pilot.empty_reviews(report)
    reviews["report_sha256"] = "stale"
    with pytest.raises(ValueError, match="different report"):
        pilot.validate_reviews(report, reviews)
    reviews = pilot.empty_reviews(report)
    reviews["decisions"]["unknown_id"] = {"status": "uncertain", "stems": []}
    with pytest.raises(ValueError, match="Unknown"):
        pilot.validate_reviews(report, reviews)
    report["records"][0]["split"] = "test"
    pilot.write_json(output / "report.json", report)
    with pytest.raises(ValueError, match="training"):
        pilot.load_report(output)


def test_changed_audio_or_midi_rejected_at_export(tmp_path):
    output, report = make_report(tmp_path)
    midi = report["records"][0]["source_midi"]["guitar"]
    with open(midi, "ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="source MIDI changed"):
        pilot.load_report(output, verify_audio=True)


def test_cached_model_support_is_one_to_one_and_hash_checked(tmp_path):
    notes = [
        {"id": "a", "family": "guitar", "pitch": 60, "onset": 1., "offset": 1.2},
        {"id": "b", "family": "guitar", "pitch": 60, "onset": 1.01, "offset": 1.2},
        {"id": "kick", "family": "drums", "pitch": 35, "onset": 2., "offset": 2.1},
    ]
    item = {"clip": "clip"}
    hashes = {stem: stem for stem in pilot.STEMS}
    path = pilot.prediction_path(tmp_path, item, "bass", "basic_pitch")
    pilot.write_json(path, {"model": "basic_pitch", "audio_sha256": "bass",
                            "events": [[1., 1.2, 60, .8]]})
    drum_path = pilot.prediction_path(tmp_path, item, "drums", "adtof")
    pilot.write_json(drum_path, {"model": "adtof", "audio_sha256": "drums",
                                 "events": [[2., 2.18, 36, .9]]})
    support, provenance = pilot.cached_support(tmp_path, item, notes, hashes)
    assert sum(bool(support[note]["bass"]) for note in ("a", "b")) == 1
    assert support["kick"]["drums"][0]["model"] == "adtof"
    assert sum(row["status"] == "available" for row in provenance) == 2
    hashes["bass"] = "changed"
    with pytest.raises(ValueError, match="provenance mismatch"):
        pilot.cached_support(tmp_path, item, notes, hashes)


def test_original_reader_retains_time_zero_channel_velocity_and_duplicates(tmp_path):
    output, report = make_report(tmp_path)
    directory = tmp_path / "dataset"
    midi = directory / "guitar.mid"
    pairs = {("song", "guitar"): {"midi": "guitar.mid", "note_count": 1}}
    item = {"composition_id": "song", "clip": "clip", "notes": {"guitar": []},
            "family_destinations": {"guitar": "bass"}}
    notes = pilot.original_notes(directory, item, pairs)
    assert len(notes) == 1
    assert notes[0]["onset"] == 0
    assert notes[0]["velocity"] == 100 and notes[0]["channel"] == 0
    assert notes[0]["legacy_stem"] == "bass"


def test_real_audio_prepare_uses_clean_render_midi_hash_not_legacy_pair_hash(tmp_path, monkeypatch):
    _, fixture = make_report(tmp_path)
    dataset = tmp_path / "dataset"
    directory = dataset / "song_base"
    directory.mkdir()
    audio = fixture["records"][0]["audio"]["guitar"]
    midi = fixture["records"][0]["source_midi"]["guitar"]
    (directory / "mix.wav").write_bytes(open(audio, "rb").read())
    pilot.write_json(directory / "clean-render.json", {
        "audio_sha256": {"guitar": pilot.digest(pilot.Path(audio))},
        "midi_sha256": {"guitar": pilot.digest(pilot.Path(midi))},
    })
    pilot.write_json(directory / "separation.json", {
        "mix_sha256": pilot.digest(directory / "mix.wav"),
        "stem_sha256": {stem: pilot.digest(pilot.Path(audio)) for stem in pilot.STEMS},
    })
    pair = {"composition_id": "song_base", "instrument": "guitar",
            "midi": "guitar.mid", "note_count": 1, "license": "Fixture", "provenance": "Synthetic test"}
    assert "midi_sha256" not in pair
    pilot.write_json(dataset / "pairs.json", [pair])
    for name, data in {
        "progress.json": {"phase": "complete"}, "verification.json": {"status": "complete"},
        "failures.json": [], "training_pairs.json": [], "splits.json": {},
    }.items():
        pilot.write_json(dataset / name, data)
    item = {
        "composition_id": "song_base", "clip": "song", "source_group": "group",
        "split": "train", "notes": {"guitar": []}, "flagged": {"guitar": False},
        "family_destinations": {"guitar": "guitar"},
        "audio": {"clean": {"guitar": audio}, "stems": {stem: audio for stem in pilot.STEMS}},
    }
    monkeypatch.setattr(pilot, "balanced_index", lambda dataset: [item])
    output = tmp_path / "prepared-pilot"
    report = pilot.prepare(dataset, output, 1)
    assert report["summary"]["notes"] == 1
    assert report["summary"]["hard_training_labels"] == 0
    assert report["records"][0]["source_midi_sha256"]["guitar"] == pilot.digest(pilot.Path(midi))
    assert pilot.load_report(output, verify_audio=True) == report
    assert (output / "index.html").exists()
    with pytest.raises(FileExistsError):
        pilot.prepare(dataset, output, 1)
    with pytest.raises(ValueError, match="No human-confirmed"):
        pilot.export_supervision(output, tmp_path / "forbidden-labels.json")


def test_positive_consumer_rejects_duplicate_targets_wrong_classes_and_changed_audio(tmp_path):
    output, report = make_report(tmp_path)
    reviews = pilot.empty_reviews(report)
    reviews["decisions"] = {"drums": {"status": "confirmed", "stems": ["other"], "scope": "exact_note"}}
    pilot.write_json(output / "reviews.json", reviews)
    destination = tmp_path / "labels.json"
    exported = pilot.export_supervision(output, destination)
    altered = deepcopy(exported)
    altered["records"][0]["targets"]["other"]["drum_onsets"][0]["class_index"] = 4
    pilot.write_json(destination, altered)
    with pytest.raises(ValueError, match="drum label"):
        pilot.load_positive_supervision(destination, rights_cleared=True)
    altered = deepcopy(exported)
    altered["records"].append(deepcopy(altered["records"][0]))
    pilot.write_json(destination, altered)
    with pytest.raises(ValueError, match="Duplicated source group"):
        pilot.load_positive_supervision(destination, rights_cleared=True)
    pilot.write_json(destination, exported)
    with open(exported["records"][0]["audio"]["guitar"], "ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="audio changed"):
        pilot.load_positive_supervision(destination, rights_cleared=True)


def test_review_server_streams_ranges_and_persists_only_valid_same_origin_decisions(tmp_path, monkeypatch):
    output, report = make_report(tmp_path)
    servers = []
    real_server = pilot.ThreadingHTTPServer

    def factory(address, handler):
        server = real_server(address, handler)
        servers.append(server)
        return server

    monkeypatch.setattr(pilot, "ThreadingHTTPServer", factory)
    worker = threading.Thread(target=pilot.serve, args=(output, 0))
    worker.start()
    try:
        for _ in range(200):
            if servers:
                break
            threading.Event().wait(.01)
        assert servers
        base = f"http://127.0.0.1:{servers[0].server_port}"
        with urlopen(base + "/report.json", timeout=10) as response:
            bootstrap = json.load(response)
        with urlopen(base + "/health", timeout=10) as response:
            assert json.load(response) == {"status": "ok", "revision": 0, "updated_at": None}
        with urlopen(base + "/click/source.json", timeout=10) as response:
            assert json.load(response) == {"beats": [
                {"time": 0., "accent": True}, {"time": .5, "accent": False},
            ]}
        assert pilot.read_json(output / "reviews.json") == bootstrap["reviews"]
        with pytest.raises(HTTPError) as unknown_click:
            urlopen(base + "/click/not-a-song.json")
        assert unknown_click.value.code == 404
        with urlopen(Request(base + "/audio/source/guitar.wav", headers={"Range": "bytes=0-31"})) as response:
            assert response.status == 206
            assert len(response.read()) == 32
            assert response.headers["Content-Range"].startswith("bytes 0-31/")
        with pytest.raises(HTTPError) as protected_cue:
            urlopen(base + "/cue/guitar.wav")
        assert protected_cue.value.code == 403
        with pytest.raises(HTTPError) as unknown_cue:
            urlopen(Request(base + "/cue/not-a-note.wav",
                            headers={"X-Review-Token": bootstrap["token"]}))
        assert unknown_cue.value.code == 404
        from experimental.benchmark.matching_cues import CueRenderer
        monkeypatch.setattr(CueRenderer, "render",
                            lambda self, note_id: pilot.Path(report["records"][0]["audio"]["guitar"]))
        with urlopen(Request(base + "/cue/guitar.wav",
                            headers={"X-Review-Token": bootstrap["token"], "Range": "bytes=0-31"})) as response:
            assert response.status == 206 and len(response.read()) == 32
        data = json.dumps({"revision": 0, "note_id": "guitar",
                           "decision": {"status": "confirmed", "stems": ["bass", "guitar"],
                                        "scope": "exact_note"}}).encode()
        with pytest.raises(HTTPError) as blocked:
            urlopen(Request(base + "/reviews", data=data))
        assert blocked.value.code == 403
        headers = {"Content-Type": "application/json", "X-Review-Token": bootstrap["token"],
                   "Origin": base}
        with urlopen(Request(base + "/reviews", data=data, headers=headers)) as response:
            saved = json.load(response)
        assert saved["revision"] == 1
        assert saved["updated_at"] == saved["decisions"]["guitar"]["saved_at"]
        assert pilot.read_json(output / "reviews.json") == saved
        with pytest.raises(HTTPError) as conflict:
            urlopen(Request(base + "/reviews", data=data, headers=headers))
        assert conflict.value.code == 409
        comment_payload = {"revision": 1, "note_id": "unknown", "comment": "<b>Quiet \u03c0 note</b>"}
        with urlopen(Request(base + "/reviews", data=json.dumps(comment_payload).encode(), headers=headers)) as response:
            annotated = json.load(response)
        assert annotated["revision"] == 2
        assert annotated["decisions"] == saved["decisions"]
        assert "unknown" not in annotated["decisions"]
        assert annotated["comments"]["unknown"]["text"] == comment_payload["comment"]
        original_bytes = (output / "reviews.json").read_bytes()
        for invalid in ("x" * 4001, [], None):
            payload = {"revision": 2, "note_id": "guitar", "comment": invalid}
            with pytest.raises(HTTPError) as rejected:
                urlopen(Request(base + "/reviews", data=json.dumps(payload).encode(), headers=headers))
            assert rejected.value.code == 400
            assert (output / "reviews.json").read_bytes() == original_bytes
        payload = {"revision": 2, "note_id": "guitar", "comment": "\U0001f600" * 4000}
        with urlopen(Request(base + "/reviews", data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                             headers=headers)) as response:
            long_comment = json.load(response)
        assert long_comment["revision"] == 3
        assert len(long_comment["comments"]["guitar"]["text"]) == 4000
        payload = {"revision": 3, "note_id": "guitar", "decision": None}
        with urlopen(Request(base + "/reviews", data=json.dumps(payload).encode(), headers=headers)) as response:
            undone = json.load(response)
        assert undone["revision"] == 4 and "guitar" not in undone["decisions"]
        assert undone["comments"] == long_comment["comments"]
        payload = {"revision": 4, "note_id": "guitar", "comment": ""}
        with urlopen(Request(base + "/reviews", data=json.dumps(payload).encode(), headers=headers)) as response:
            cleared = json.load(response)
        assert "guitar" not in cleared["comments"] and "unknown" in cleared["comments"]
        for path in ("/../../dataset/audio.wav", "/audio/source/unknown.wav"):
            with pytest.raises(HTTPError) as missing:
                urlopen(base + path)
            assert missing.value.code == 404
        source = pilot.Path(report["records"][0]["source_midi"]["guitar"])
        with source.open("ab") as stream:
            stream.write(b"changed")
        with pytest.raises(HTTPError) as changed_click:
            urlopen(base + "/click/source.json")
        assert changed_click.value.code == 500
        assert "Source MIDI changed" in json.load(changed_click.value)["error"]
    finally:
        if servers:
            servers[0].shutdown()
        worker.join(timeout=10)
        assert not worker.is_alive()


def test_legacy_decisions_preserved_but_not_exported_until_note_level_recheck(tmp_path):
    output, report = make_report(tmp_path)
    reviews = pilot.empty_reviews(report)
    reviews["decisions"] = {"guitar": {"status": "confirmed", "stems": ["guitar"]}}
    pilot.write_json(output / "reviews.json", reviews)
    before = (output / "reviews.json").read_bytes()
    assert pilot.validate_reviews(report, reviews) == reviews["decisions"]
    with pytest.raises(ValueError, match="exact-note scope"):
        pilot.export_supervision(output, tmp_path / "labels.json")
    assert (output / "reviews.json").read_bytes() == before
    reviews["decisions"]["guitar"]["scope"] = "exact_note"
    pilot.write_json(output / "reviews.json", reviews)
    exported = pilot.export_supervision(output, tmp_path / "labels.json")
    assert exported["confirmed_notes"] == 1
    assert exported["review_scope"] == "exact_note"
    exported.pop("review_scope")
    pilot.write_json(tmp_path / "labels.json", exported)
    with pytest.raises(ValueError, match="exact-note review scope"):
        pilot.load_positive_supervision(tmp_path / "labels.json", rights_cleared=True)


def test_legacy_report_compatibility_requires_unchanged_evidence(tmp_path, monkeypatch):
    output, report = make_report(tmp_path)
    report["code_sha256"] = pilot.LEGACY_CODE_SHA256
    pilot.write_json(output / "report.json", report)
    assert pilot.evidence_hash() == pilot.LEGACY_EVIDENCE_SHA256
    assert pilot.load_report(output) == report
    monkeypatch.setattr(pilot, "MIN_SCORE", .31)
    with pytest.raises(ValueError, match="algorithm/schema changed"):
        pilot.load_report(output)


@pytest.mark.parametrize("comments", [
    None, [], {"not-a-note": {"text": "Hello"}}, {"guitar": "Wrong shape"},
    {"guitar": {"text": ""}}, {"guitar": {"text": "x" * 4001}}, {"guitar": {"text": 4}},
])
def test_invalid_comment_mappings_are_rejected(tmp_path, comments):
    _, report = make_report(tmp_path)
    reviews = pilot.empty_reviews(report)
    reviews["comments"] = comments
    with pytest.raises(ValueError, match="Comments"):
        pilot.validate_reviews(report, reviews)


def test_comments_are_provenance_not_training_confirmations(tmp_path):
    output, report = make_report(tmp_path)
    reviews = pilot.empty_reviews(report)
    reviews["comments"] = {"guitar": {"text": "Could be two sources", "saved_at": "2026-11-14T16:00:00+00:00"}}
    pilot.write_json(output / "reviews.json", reviews)
    with pytest.raises(ValueError, match="No human-confirmed notes"):
        pilot.export_supervision(output, tmp_path / "labels.json")
    reviews["decisions"]["guitar"] = {"status": "confirmed", "stems": ["guitar"], "scope": "exact_note"}
    pilot.write_json(output / "reviews.json", reviews)
    exported = pilot.export_supervision(output, tmp_path / "labels.json")
    assert exported["review_comments"] == reviews["comments"]
    assert exported["confirmed_notes"] == 1
