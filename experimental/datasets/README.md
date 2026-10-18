# External transcription datasets

This directory is excluded from Git and is not part of the application. It
contains downloaded audio/MIDI pairs, generated manifests, and benchmark output.

## BabySlakh

- Source: https://zenodo.org/records/4603870
- Version: v2, 16 kHz WAV release
- License: CC BY 4.0
- Archive: `babyslakh/babyslakh_16k.tar.gz`
- Expected bytes: `882818115`
- Expected MD5: `311096dc2bde7d61c97e930edbfc7f78`

The fixed experimental split is by complete composition:

- Train: `Track00001` through `Track00012`
- Validation: `Track00013` through `Track00016`
- Test: `Track00017` through `Track00020`

Run `python experimental/datasets/build_babyslakh_manifest.py` after extraction.
No note, stem, or excerpt from one composition is assigned to another split.

The first baseline reports are stored under `babyslakh/results/`. These reports
use the untrained/bootstrap-only application model and do not modify application
profiles.

## MAESTRO

- Source: https://magenta.tensorflow.org/datasets/maestro
- Version: v3.0.0
- License: CC BY-NC-SA 4.0
- Local subset: shortest composition from each official train/validation/test split

The full MAESTRO archive is approximately 101 GB and does not fit on this
machine. `download_maestro_subset.py` uses HTTP byte ranges to extract only the
selected aligned WAV/MIDI members. It requires the experiment-only `remotezip`
package; this package is intentionally not included in application requirements.

## Baseline results

All values below use the unchanged bootstrap-only model. Note F1 requires exact
MIDI pitch and onset within 50 ms.

| Dataset | Split | Source | Note F1 | Frame F1 | Octave errors | RTF |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| BabySlakh Track 1 S03 | train | electric bass | 0.004 | 0.001 | 118 | 0.086 |
| BabySlakh Track 1 S02 | train | acoustic piano | 0.226 | 0.192 | 76 | 0.085 |
| BabySlakh Track 20 S01 | test | electric bass | 0.000 | 0.000 | 97 | 0.081 |
| BabySlakh Track 20 S03 | test | acoustic piano | 0.483 | 0.293 | 36 | 0.090 |
| MAESTRO Scarlatti K. 525 | test | acoustic piano | 0.145 | 0.127 | 29 | 0.126 |

The external bass results reveal a systematic octave/register failure. The
polyphonic results vary strongly by domain and have poor duration agreement.
These reports are baselines, not trained-model results; training and feature
selection must use only train/validation entries before final test evaluation.

## First-10-note user adaptation

`../user_adaptation_benchmark.py` emulates the intended product workflow while
removing onset and duration detection from the problem:

1. MIDI onsets, offsets, and simultaneous-note count are treated as known.
2. The first 10 chronological notes are recorded as user-confirmed positives.
3. If note 10 cuts through a chord, the remainder of that onset group is not
	evaluated, preventing train/test leakage.
4. Every later onset group is predicted jointly.
5. Zero-shot and adapted predictions are scored on the exact same later notes.

| Dataset | Source | Zero-shot note F1 | Adapted note F1 | Delta |
| --- | --- | ---: | ---: | ---: |
| BabySlakh Track 20 S01 | electric bass | 0.000 | 0.000 | +0.000 |
| BabySlakh Track 20 S03 | acoustic piano | 0.575 | 0.558 | -0.017 |
| MAESTRO Scarlatti K. 525 | acoustic piano | 0.509 | 0.524 | +0.015 |

The current user-training path is not a reliable pitch adapter. Bass remains an
octave away because NNLS supplies an existing candidate and user training only
changes the keep gate or the fallback fingerprint; it cannot replace the chosen
pitch. Ten positive piano examples can either help or hurt the shared gate.
Future mono/poly experiments should therefore learn candidate pitch/rank
corrections directly and compare against these saved reports.