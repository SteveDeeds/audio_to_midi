from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import check_grad

from experimental.benchmark import fresh_experiment as experiment
from experimental.benchmark.small_heads import ARCHITECTURES, SmallHead


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_small_head_gradient_and_parameter_counts(architecture):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(8, 3))
    y = (X[:, 0] > 0).astype(float)
    head = SmallHead.initialize(X, architecture, 3)
    parameters = head.pack()
    anchor = np.zeros_like(parameters)
    args = (X, y, anchor, 0.3, np.ones(len(y)))
    error = check_grad(lambda values: head.objective(values, *args)[0],
                       lambda values: head.objective(values, *args)[1], parameters)
    assert error < 1e-4
    expected = sum(a * b + b for a, b in zip(
        (3, *ARCHITECTURES[architecture]), (*ARCHITECTURES[architecture], 1)))
    assert len(parameters) == expected


@pytest.mark.parametrize("policy", experiment.POLICIES)
def test_edit_updates_are_finite_and_never_mutate_pretrained_weights(policy):
    X = np.array([[-1.0], [1.0]] * 4)
    head = SmallHead.initialize(X, "linear", 2026)
    head.weights[0][:] = 0.2
    before = head.pack().copy()
    probabilities = head.probabilities(X, X[:2], np.array([1.0, 0.0]),
                                       policy=policy, lam=0.1)
    assert np.isfinite(probabilities).all()
    assert np.array_equal(head.pack(), before)
    if policy in ("local", "last_layer", "anchored"):
        assert np.mean((probabilities >= 0.5) == (X[:, 0] < 0)) == 1
    elif policy == "bias":
        assert np.allclose(probabilities, head.probabilities(X))


def test_neural_head_can_learn_nonlinear_pattern_from_scratch():
    X = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]] * 8)
    y = (X[:, 0] * X[:, 1] > 0).astype(float)
    head = SmallHead.fit(X, y, "mlp16", lam=0.01, iterations=150)
    assert np.array_equal(head.probabilities(X) >= 0.5, y)


def test_neural_adaptation_line_search_handles_relu_kinks():
    rng = np.random.default_rng(49)
    head = SmallHead.initialize(rng.normal(size=(30, 8)), "mlp16", 49)
    for weight in head.weights:
        weight *= 3
    for bias in head.biases:
        bias[:] = rng.normal(size=bias.shape)
    X = rng.normal(size=(2, 8))
    y = np.array([0., 1.])
    before = head.pack().copy()
    args = (X, y, before, 10., np.ones(2))
    parameters, converged = head.optimize(
        X, y, lam=10., iterations=60, anchor=before)
    assert converged
    assert np.isfinite(parameters).all()
    assert head.objective(parameters, *args)[0] < head.objective(before, *args)[0] - 7
    assert np.array_equal(head.pack(), before)


def test_abnormal_head_optimization_is_still_an_error(monkeypatch):
    from types import SimpleNamespace

    X = np.array([[-1.], [1.]])
    head = SmallHead.initialize(X, "linear", 0)
    before = head.pack().copy()
    monkeypatch.setattr(
        "experimental.benchmark.small_heads.minimize",
        lambda objective, start, **kwargs: SimpleNamespace(
            fun=1., x=start.copy(), success=False, status=2, message="ABNORMAL"),
    )
    with pytest.raises(RuntimeError, match="Head optimization failed: ABNORMAL"):
        head.optimize(X, np.array([0., 1.]), lam=10., iterations=60, anchor=before)
    assert np.array_equal(head.pack(), before)


def test_candidates_are_truth_free_and_labels_match_one_to_one():
    channels = {
        "own:basic_pitch": np.array([[22., 22.2, 60., .8], [22.08, 22.3, 60., .6]]),
        "own:piano_hr": np.array([[22.01, 22.2, 60., .9]]),
    }
    candidates = experiment.candidate_pool(channels)
    assert len(candidates) == 2
    labels = experiment.matched_labels(candidates, [[22.04, 22.4, 60, 80], [25., 25.4, 67, 80]])
    assert labels.sum() == 1
    assert 67 not in candidates[:, 2]


def test_edits_do_not_read_query_labels_and_author_out_of_pool_missing_notes():
    channels = {"own:basic_pitch": np.array([[1., 1.2, 72., .8], [22., 22.2, 72., .8]])}
    candidates = experiment.candidate_pool(channels)
    view = experiment.View({}, "guitar", "single_basic_pitch", channels, candidates,
                           experiment.features(candidates, channels),
                           [[2., 2.2, 60, 80], [22., 22.2, 60, 80]])
    head = SmallHead.initialize(view.X, "linear", 0)
    head.weights[0][:] = 0
    head.biases[0][:] = 1
    stream = experiment.edit_stream(view, head)
    changed = replace(view, reference=[*view.reference[:1], [27., 27.1, 99, 10]])
    other = experiment.edit_stream(changed, head)
    assert [(row.tolist(), y) for row, y in stream] == [(row.tolist(), y) for row, y in other]
    assert all(row[0] < experiment.SUPPORT_SECONDS for row, _ in stream)
    resolved = experiment.apply_overrides(candidates, stream)
    assert (2., 60.) in {(row[0], row[2]) for row in resolved}
    assert (1., 72.) not in {(row[0], row[2]) for row in resolved}
    assert (22., 72.) in {(row[0], row[2]) for row in resolved}


def make_frozen_fixture(output):
    records = []
    reference = [[time, time + .2, 60, 90] for time in (1., 4., 22., 25.)]
    predicted = [[*row[:3], .8] for row in reference]
    predicted.extend([[time + .3, time + .5, 72, .6] for time in (1., 4., 22., 25.)])
    for group, split in (("train-a", "train"), ("train-b", "train"),
                         ("validation", "validation"), ("test", "test")):
        for variant in ("base", "aug"):
            record = {
                "clip": f"{group}-{variant}", "source_group": group, "split": split,
                "composition_id": f"{group}-{variant}", "duration": 30.,
                "notes": {"bass": reference}, "family_destinations": {"bass": "guitar"},
                "flagged": {"bass": False},
            }
            records.append(record)
            for stem in experiment.STEMS:
                for model in ("basic_pitch", "piano_hr"):
                    experiment.write_json(experiment.prediction_path(output, record, stem, model),
                                          {"events": predicted, "runtime_seconds": .01})
    return {"records": records, "models": ["basic_pitch", "piano_hr"],
            "initialization": "fresh"}


def test_three_stage_smoke_is_grouped_and_test_locked(tmp_path):
    data = make_frozen_fixture(tmp_path)
    experiment.baseline_report(tmp_path, data, "train", False)
    assert experiment.read_json(tmp_path / "train_canned_baselines.json")["summary"]
    experiment.train(tmp_path, data, ("linear", "mlp16"), 40, 2, (10.,))
    selected = experiment.read_json(tmp_path / "selection.json")
    assert set(selected["fold_of_source_group"]) == {"train-a", "train-b"}
    assert selected["fold_of_source_group"]["train-a"] != selected["fold_of_source_group"]["train-b"]
    cv = experiment.read_json(tmp_path / "cv_rows.json")
    assert {row["architecture"] for row in cv} == {"linear", "mlp16"}
    assert {row["config"] for row in cv} == {
        "single_basic_pitch", "single_piano_hr", "multi_own", "multi_cross",
    }
    assert {row["policy"] for row in cv} == set(experiment.POLICIES)
    assert all(row["edits_preserved"] for row in cv)
    experiment.evaluate(tmp_path, data, "validation", False)
    result = experiment.read_json(tmp_path / "validation_results.json")
    assert result["baseline_summary"] and result["head_vs_canned"] and result["head_summary"]
    assert result["head_vs_canned"][0]["gain"] > 0
    with pytest.raises(ValueError, match="locked"):
        experiment.evaluate(tmp_path, data, "test", False)
    with pytest.raises(ValueError, match="locked"):
        experiment.baseline_report(tmp_path, data, "test", False)
    assert not (tmp_path / "test_results.json").exists()
    experiment.evaluate(tmp_path, data, "test", True)
    assert (tmp_path / "test_results.json").exists()
    with pytest.raises(FileExistsError, match="already evaluated"):
        experiment.evaluate(tmp_path, data, "test", True)
    with pytest.raises(FileExistsError, match="already completed"):
        experiment.train(tmp_path, data, ("linear",), 5, 2, (10.,))


def test_load_rejects_source_group_leakage(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "code_hash", lambda: "code")
    data = {
        "schema": experiment.SCHEMA, "code_sha256": "code", "dataset": str(tmp_path),
        "dataset_hashes": {}, "records": [
            {"source_group": "same", "split": "train"},
            {"source_group": "same", "split": "test"},
        ],
    }
    experiment.write_json(tmp_path / "experiment.json", data)
    with pytest.raises(ValueError, match="leakage"):
        experiment.load_experiment(tmp_path)


def test_drum_labels_use_canned_model_classes_and_collapse_simultaneous_articulations():
    rows = [[22., 22.2, 35, 90], [23., 23.2, 44, 80], [23., 23.3, 46, 100],
            [24., 24.2, 81, 80]]
    assert experiment.drum_reference(rows) == [
        [22., 22.2, 36, 90], [23., 23.3, 42, 100],
    ]


def add_other_fixture(data):
    for record in data["records"]:
        record["notes"]["other"] = deepcopy(record["notes"]["bass"])
        record["flagged"]["other"] = False
        record["other_families"] = ["guitar", "piano"]


def test_other_is_scored_separately_and_empty_targets_count_false_positives(tmp_path):
    data = make_frozen_fixture(tmp_path)
    add_other_fixture(data)
    positive, empty = data["records"][:2]
    positive["notes"]["other"] = [[22., 22.2, 60, 90], [25., 25.2, 67, 80]]
    empty["notes"]["other"] = []
    empty["other_families"] = []
    prediction = {"events": [[22., 22.2, 60, .8], [25., 25.2, 67, .7]], "runtime_seconds": .01}
    for record in (positive, empty):
        for model in data["models"]:
            experiment.write_json(experiment.prediction_path(tmp_path, record, "other", model), prediction)
    rows = experiment.baselines(tmp_path, data, [positive, empty])
    other = [row for row in rows if row["role"] == "other"]
    assert len(other) == 4
    assert all(row["target_kind"] == "routed_stem" and row["target_stem"] == "other" for row in other)
    assert all(row["f1"] == 1 for row in other if row["clip"] == positive["clip"])
    assert all(row["false_positives"] == 2 for row in other if row["clip"] == empty["clip"])
    summary = experiment.summarize(other, ("role", "system"))
    assert all(row["empty_reference_views"] == 1 and row["empty_reference_false_positives"] == 2
               for row in summary)
    assert all(row["target_kind"] == "instrument_family" for row in rows if row["role"] == "bass")


def test_other_participates_in_head_and_edit_comparisons(tmp_path):
    data = make_frozen_fixture(tmp_path)
    add_other_fixture(data)
    experiment.train(tmp_path, data, ("linear",), 40, 2, (10.,))
    selected = experiment.read_json(tmp_path / "selection.json")
    assert "other" in selected["selection"]
    rows = experiment.read_json(tmp_path / "cv_rows.json")
    other = [row for row in rows if row["role"] == "other"]
    assert {row["config"] for row in other} == {
        "single_basic_pitch", "single_piano_hr", "multi_own", "multi_cross",
    }
    assert {row["policy"] for row in other} == set(experiment.POLICIES)
    assert all(row["edits_preserved"] for row in other)
    experiment.evaluate(tmp_path, data, "validation", False)
    result = experiment.read_json(tmp_path / "validation_results.json")
    assert any(row["role"] == "other" for row in result["head_vs_canned"])
    assert not (tmp_path / "test_results.json").exists()


def test_upgrade_preserves_caches_and_original_family_labels(tmp_path, monkeypatch):
    original = make_frozen_fixture(tmp_path)
    original.update(schema=1, code_sha256=experiment.FAMILY_ONLY_CODE_SHA256,
                    dataset=str(tmp_path), dataset_hashes={})
    for record in original["records"]:
        directory = tmp_path / record["clip"]
        record.update(
            source_composition_id=record["source_group"], audio={"mix": str(directory / "mix.wav")},
            legacy_family_flagged={"bass": False}, routing_policy="whole-family-energy-v1",
        )
        experiment.write_json(directory / "separation.json",
                              {"stem_sha256": {stem: "hash" for stem in experiment.STEMS}})
        for stem in experiment.STEMS:
            for model in original["models"]:
                path = experiment.prediction_path(tmp_path, record, stem, model)
                prediction = experiment.read_json(path)
                prediction.update(model=model, audio_sha256="hash")
                experiment.write_json(path, prediction)
    revised = deepcopy(original)
    revised.update(schema=experiment.SCHEMA, code_sha256=experiment.code_hash())
    add_other_fixture(revised)
    monkeypatch.setattr(experiment, "experiment_manifest", lambda dataset, models: deepcopy(revised))
    experiment.write_json(tmp_path / "experiment.json", original)
    example = experiment.prediction_path(tmp_path, original["records"][0], "other", "basic_pitch")
    frozen = example.read_bytes()
    result = experiment.upgrade_other(tmp_path)
    assert result["target_revision"]["preserved_frozen_predictions"] == 96
    assert not result["target_revision"]["learned_state_reused"]
    assert example.read_bytes() == frozen
    assert experiment.read_json(tmp_path / "experiment.before-other.json") == original
    assert experiment.load_experiment(tmp_path)["records"][0]["notes"]["other"]
    with pytest.raises(ValueError, match="registered"):
        experiment.upgrade_other(tmp_path)


@pytest.mark.parametrize("artifact", ["heads", "selection.json", "cv_rows.json",
                                    "train_canned_baselines.json", "validation_results.json"])
def test_upgrade_refuses_to_change_fitted_or_scored_experiments(tmp_path, artifact):
    experiment.write_json(tmp_path / "experiment.json", {
        "schema": 1, "code_sha256": experiment.FAMILY_ONLY_CODE_SHA256,
    })
    if artifact == "heads":
        (tmp_path / artifact).mkdir()
    else:
        experiment.write_json(tmp_path / artifact, {})
    with pytest.raises(ValueError, match="Cannot revise"):
        experiment.upgrade_other(tmp_path)
    assert not (tmp_path / "experiment.before-other.json").exists()


def test_import_preserves_variants_and_rejects_split_families(tmp_path, monkeypatch):
    from experimental.benchmark import import_atm

    pairs = []
    assignments = {}
    for variant in ("base", "aug"):
        song = f"lakh_12345678901234567890_{variant}"
        directory = tmp_path / song
        (directory / "stems").mkdir(parents=True)
        (directory / "clean").mkdir()
        (directory / "mix.wav").touch()
        for stem in experiment.STEMS:
            (directory / "stems" / f"{stem}.wav").touch()
        (directory / "clean" / "piano.wav").touch()
        assignments[song] = "train"
        pairs.append({
            "composition_id": song, "source_group": "source", "source_composition_id": "original",
            "instrument": "piano", "midi": "fixture.mid",
            "clean_audio": f"{song}/clean/piano.wav", "flagged": False,
        })
        experiment.write_json(directory / "stem-routing.json", {
            "routing_policy": import_atm.FAMILY_ROUTING_POLICY,
            "family_destinations": {"piano": "guitar"},
            "notes": [{"family": "piano", "primary_stem": "guitar"}],
        })
    experiment.write_json(tmp_path / "pairs.json", pairs)
    experiment.write_json(tmp_path / "training_pairs.json", [
        {"composition_id": pair["composition_id"], "stem": "guitar", "flagged": True}
        for pair in pairs
    ])
    experiment.write_json(tmp_path / "splits.json", {"composition_to_split": assignments})
    class Info:
        duration = 30.
    monkeypatch.setattr(import_atm, "load_midi_notes", lambda path: [])
    monkeypatch.setattr(import_atm.sf, "info", lambda path: Info())
    index = import_atm.balanced_index(tmp_path)
    assert len({record["clip"] for record in index}) == 2
    assert {record["source_group"] for record in index} == {"source"}
    assert all(record["flagged"]["piano"] and not record["legacy_family_flagged"]["piano"]
               for record in index)
    assert all(record["notes"]["other"] == [] and record["other_families"] == [] for record in index)
    route_path = tmp_path / pairs[0]["composition_id"] / "stem-routing.json"
    route = experiment.read_json(route_path)
    route["notes"].append({"family": "piano", "primary_stem": "piano"})
    experiment.write_json(route_path, route)
    with pytest.raises(ValueError, match="split across stems"):
        import_atm.balanced_index(tmp_path)
    for pair in pairs:
        experiment.write_json(tmp_path / pair["composition_id"] / "stem-routing.json", {
            "routing_policy": import_atm.FAMILY_ROUTING_POLICY,
            "family_destinations": {"piano": "other"},
            "notes": [{"family": "piano", "primary_stem": "other"}],
        })
    experiment.write_json(tmp_path / "training_pairs.json", [
        {"composition_id": pair["composition_id"], "stem": "other", "flagged": False,
         "midi": "other-target.mid", "note_count": 2, "instrument_families": ["piano"]}
        for pair in pairs
    ])
    notes = [(1., 1.2, 60, 90, 0), (22., 22.2, 64, 80, 0)]
    monkeypatch.setattr(import_atm, "load_midi_notes", lambda path: notes)
    index = import_atm.balanced_index(tmp_path)
    assert all(record["notes"]["other"] == [list(row[:4]) for row in notes] for record in index)
    assert all(record["other_families"] == ["piano"] for record in index)


def test_benchmark_reader_preserves_zero_tick_notes_that_pretty_midi_drops(tmp_path):
    import mido
    import pretty_midi
    from experimental.benchmark.import_atm import load_midi_notes

    midi = mido.MidiFile()
    midi.tracks.append(mido.MidiTrack([
        mido.Message("note_on", note=60, velocity=90),
        mido.Message("note_off", note=60, time=0),
        mido.Message("note_on", note=64, velocity=80, time=480),
        mido.Message("note_off", note=64, time=480),
    ]))
    path = tmp_path / "zero_tick.mid"
    midi.save(path)
    assert sum(len(inst.notes) for inst in pretty_midi.PrettyMIDI(str(path)).instruments) == 1
    rows = load_midi_notes(str(path))
    assert [(row[0], row[2]) for row in rows] == [(0., 60), (.5, 64)]
