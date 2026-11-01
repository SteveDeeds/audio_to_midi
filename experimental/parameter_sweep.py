"""Reproducible sensitivity measurements, not an onset-accuracy benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import html
import json
from pathlib import Path

import numpy as np

from experimental.tuning import Evidence, TuningParameters, load_evidence, retune, tuning_defaults
from experimental.vocal_midi import write_csv

GRIDS = {
    "neural_threshold": [0.20, 0.30, 0.40, 0.50, 0.60, 0.80],
    "voice_db": [-45, -40, -35, -30, -25],
    "onset_wait": [0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.24],
    "pitch_hold": [0.04, 0.06, 0.09, 0.12, 0.16, 0.20, 0.25],
    "min_note": [0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.16],
    "pitch_hysteresis": [0.25, 0.50, 0.75, 1, 1.25, 1.50, 2],
    "periodicity": [0.20, 0.35, 0.50, 0.65, 0.80],
    "energy_valley": [1, 2, 3, 4, 6, 8, 10, 12],
    "release_gap": [0.02, 0.06, 0.10, 0.16],
}
RECOMMENDATIONS = {
    "neural_threshold": {"start": 0.30, "suggested": [0.20, 0.60]},
    "voice_db": {"start": -35, "suggested": [-45, -25]},
    "onset_wait": {"start": 0.12, "suggested": [0.10, 0.16]},
    "pitch_hold": {"start": 0.09, "suggested": [0.06, 0.12]},
    "min_note": {"start": 0.08, "suggested": [0.05, 0.10]},
}


def measure(evidence: Evidence, parameters: TuningParameters) -> dict:
    result, notes, _ = retune(evidence, parameters)
    stats = result["stats"]
    durations = [n.end - n.start for n in notes]
    return {
        "notes": len(notes), "raw_onsets": stats["raw_onsets"],
        "syllable_starts": stats["syllable_starts"], "pitch_only_starts": stats["slur_pitch_starts"],
        "pitch_proposals": stats["pitch_change_proposals"], "extra_syllable_proposals": stats["extra_syllable_proposals"],
        "short_under_100ms": sum(d < 0.1 - 1e-9 for d in durations),
        "long_over_500ms": sum(d > 0.5 + 1e-9 for d in durations),
        "agreement_under_60pct": sum(n.pitch_agreement < 0.6 for n in notes),
        "mean_pitch_agreement": float(np.mean([n.pitch_agreement for n in notes])) if notes else None,
        "minimum_duration_ms": min(durations) * 1000 if notes else None,
        "slurred": stats["links"]["slurred"], "tied": stats["links"]["tied"],
        "articulated": stats["links"]["articulated"], "parameters": asdict(parameters),
    }


def sweep(evidence: Evidence) -> dict:
    rows = []
    for mode, base in (("legacy", TuningParameters(split_pitch=True)), ("syllabic", tuning_defaults())):
        for method in ("madmom_cnn", "madmom_brnn"):
            baseline = replace(base, method=method)
            for name, values in GRIDS.items():
                for value in values:
                    p = replace(baseline, **{name: value})
                    if mode == "syllabic" and name == "energy_valley":
                        p = replace(p, energy_recovery=value)
                    rows.append({"mode": mode, "method": method, "control": name, "value": value,
                                 **measure(evidence, p)})
    profiles = {
        "balanced": tuning_defaults(),
        "more_splits": replace(tuning_defaults(), neural_threshold=0.25, onset_wait=0.10,
                               pitch_hold=0.06, min_note=0.06),
        "fewer_fragments": replace(tuning_defaults(), neural_threshold=0.40, onset_wait=0.16,
                                   pitch_hold=0.12, min_note=0.10),
    }
    # Check the joint effect instead of claiming the one-at-a-time effects add up.
    interaction = []
    for spacing in (0.08, 0.12, 0.16):
        for hold in (0.06, 0.09, 0.16):
            for minimum in (0.05, 0.08, 0.10):
                p = replace(tuning_defaults(), onset_wait=spacing, pitch_hold=hold, min_note=minimum)
                interaction.append(measure(evidence, p))
    return {
        "analysis": evidence.metadata, "recommendations": RECOMMENDATIONS,
        "limitation": "No syllable labels: counts, short fragments and pitch agreement measure sensitivity, not accuracy. More notes are not necessarily better. Long notes can be intentional vowels.",
        "rows": rows, "interaction": interaction,
        "profiles": {name: measure(evidence, p) for name, p in profiles.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).parent / "outputs"
    parser.add_argument("--cache", type=Path, default=root / "tuner_cache_39_to_end")
    parser.add_argument("--output", type=Path, default=root / "syllable_parameter_sweep")
    args = parser.parse_args()
    report = sweep(load_evidence(args.cache))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    rows = [{k: v for k, v in row.items() if k != "parameters"} for row in report["rows"]]
    write_csv(args.output / "sensitivity.csv", list(rows[0]), rows)
    for name, profile in report["profiles"].items():
        (args.output / f"{name}.json").write_text(
            json.dumps(profile["parameters"], indent=2, allow_nan=False), encoding="utf-8",
        )
    columns = list(rows[0])
    table = "<table><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in columns) + "</tr>"
    table += "".join("<tr>" + "".join(f"<td>{html.escape(str(row[c]))}</td>" for c in columns) + "</tr>" for row in rows)
    page = (
        "<!doctype html><meta charset='utf-8'><title>Vocal parameter sensitivity</title>"
        "<style>body{font:14px system-ui;padding:20px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:5px}"
        "th{position:sticky;top:0;background:white}</style><h1>39.0-65.6s vocal parameter sweep</h1>"
        f"<p>{html.escape(report['limitation'])}</p><p>Each row changes one setting from its mode's baseline; "
        "the JSON report also contains 27 joint spacing/hold/minimum-length combinations.</p>"
        "<h2>Suggested starting ranges</h2><pre>" + html.escape(json.dumps(RECOMMENDATIONS, indent=2)) +
        "</pre><h2>Example presets</h2>" +
        "".join(f"<p><a href='{name}.json'>{name}</a>: {p['notes']} notes, {p['syllable_starts']} syllable starts, "
                f"{p['pitch_only_starts']} pitch-only starts, {p['short_under_100ms']} short fragments.</p>"
                for name, p in report["profiles"].items()) + table + "</table>"
    )
    (args.output / "review.html").write_text(page, encoding="utf-8")
    print(json.dumps({"rows": len(rows), "interaction": len(report["interaction"]),
                      "profiles": {n: {k: v for k, v in p.items() if k != "parameters"}
                                   for n, p in report["profiles"].items()},
                      "report": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
