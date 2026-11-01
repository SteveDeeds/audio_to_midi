# Vocal audio-to-MIDI experiment

## Interactive parameter GUI

From the repository root:

```powershell
.\experimental\.venv\Scripts\python.exe -m experimental.tuner_server
Start-Process "http://127.0.0.1:8765"
```

The GUI uses the corrected **39.0-65.6 second** region. It binds only to localhost.
The first launch builds a local analysis cache if needed; subsequent launches
reuse it. Use `--port 8766` if the default port is occupied. Stop a manually
launched server with Ctrl+C.

The GUI defaults to **madmom CNN in syllable mode**, with sustained-pitch
splitting and key correction **Off**. **Original baseline** disables syllable
mode and extra pitch splitting
and restores the original thresholds. It reproduces all five reviewed candidates
on the cached evidence. **Tuning defaults** restores the enhanced starting preset.
These GUI defaults do not change the original command-line experiment's defaults.

There are **five visible numeric controls**, plus model/key selectors:

| Control | Start | Suggested range | Effect |
|---|---:|---:|---|
| Neural onset threshold | 0.30 | 0.20-0.60 | Lower detects weaker syllable attacks; higher rejects weak proposals. |
| Voice energy gate | -35 dB | -45 to -25 dB | Raise to shorten quiet tails/remove quiet regions; -40 to -35 is a plateau on this clip. |
| Minimum syllable spacing | 0.12 s | 0.10-0.16 s | Raise to suppress closely spaced attack duplicates; does not impose this spacing on pitch-only slurs. |
| Slur pitch-change hold | 0.09 s | 0.06-0.12 s | Lower accepts shorter sustained pitch changes; raise toward 0.16 s if brief vibrato/slide excursions split too often. |
| Minimum split length | 0.08 s | 0.05-0.10 s | Lower accepts shorter syllable/pitch segments; raise to remove tiny fragments. It does not invent syllables. |

For librosa/SuperFlux/pitch-aware selections, **Flux onset threshold** (start
0.12) replaces the neural slider, so there are still five visible controls.
Suggested ranges above were measured on the corrected vocal region, not
calibrated across singers. Wider slider bounds remain available and preserve
existing valid presets.

Starting defaults fix pitch hysteresis at 0.75 semitone, CREPE periodicity at
0.35, note voiced-fraction/agreement rejection at zero, phrase gap at 60 ms and
optional key correction at one semitone. Hidden internal settings remain
validated and are preserved by preset loading/saving, tuning, preview and export.
An imported legacy preset retains legacy semantics; **Tuning defaults** restores
syllable mode. No manual split/edit action is added; the production app already
has that workflow.

The underlying neural network weights and spectral preprocessing are not changed
by these controls: this GUI tunes peak picking and note construction.

### Syllables versus slurs

Syllable mode uses neural/spectral attacks as **syllable proxies**, not a phonetic
or lyric transcription. A detected attack retriggers even at the same pitch.
A sustained pitch change away from an attack splits a vowel into exactly
connected/slurred notes. Attacks take priority over nearby pitch-change proposals.
Same-pitch legato splits, including ones collapsed by key correction, are merged
and their pitch/quality evidence is remeasured. Same-pitch syllable attacks are
**not** merged.

Acoustic pre-attack gaps/dips still determine earlier releases. Where an attack
has no measurable gap, a **10 ms rendering gap** encodes articulation in MIDI and
retriggering in the synth; this is explicitly labeled and is **not measured vocal
silence**. Pitch-only changes do not acquire gaps from energy dips. The original
baseline retains the old acoustic-gap-only tie/slur behavior.

Syllable mode also allows energy-backed SuperFlux retriggers if the primary
detector missed them: a 6 dB fall and recovery, voiced evidence and spacing away
from existing attacks are required. These are separate from pitch-only proposals
and appear in blue and the onset CSV with evidence labels. Sweeping this threshold
from 1 to 12 dB added **zero extra boundaries at default CNN settings**, so it was
not chosen as a primary control. At higher neural thresholds or with other
detectors the fallback can add proposals; it never adds regular time-grid splits.
Syllable mode couples recovery to the internal `energy_valley` value; the separate
`energy_recovery` setting is retained for the legacy hybrid only.
In syllable mode SuperFlux and the pitch-aware selection share the same helper
stack; the original baseline retains the distinct original hybrid.

### Measured sensitivity and example presets

Reproduce **224 one-at-a-time runs** on CNN/BRNN in legacy and syllable modes,
plus **27 joint spacing/hold/minimum-length combinations**:

```powershell
.\experimental\.venv\Scripts\python.exe -m experimental.parameter_sweep
Start-Process .\experimental\outputs\syllable_parameter_sweep\review.html
```

The [sweep review](outputs/syllable_parameter_sweep/review.html),
[CSV](outputs/syllable_parameter_sweep/sensitivity.csv) and
[full report](outputs/syllable_parameter_sweep/report.json) retain source/cache
provenance, complete settings and measurements. **Load preset** accepts the
generated examples:

| CNN preset | Notes | Syllable starts | Pitch-only starts | Notes under 100 ms |
|---|---:|---:|---:|---:|
| [Balanced](outputs/syllable_parameter_sweep/balanced.json) | 101 | 91 | 10 | 3 |
| [More splits](outputs/syllable_parameter_sweep/more_splits.json) | 114 | 98 | 16 | 16 |
| [Fewer fragments](outputs/syllable_parameter_sweep/fewer_fragments.json) | 95 | 77 | 18 | 0 |

At the balanced settings, changing only minimum length from 50 to 100 ms moves
CNN output from 107 to 99 notes and short fragments from 10 to zero. Pitch hold
from 60 to 200 ms changes accepted pitch-only starts from 14 to 5 while retaining
91 syllable starts. The 90-120 ms hold interval is a plateau here. Spacing
80-160 ms changes syllable starts from 99 to 86 and short fragments from 11 to 3.
Pitch hysteresis 0.50-1.50 semitones changed only two notes, so it remains fixed
instead of using a control slot. Raising periodicity aggressively removes
voiced/pitch material, so it is not a substitute for fixing syllable boundaries.

**These are sensitivity/fragmentation measurements, not accuracy rankings.**
No syllable/onset ground truth is available. A long vowel or fast ornament can be
intentional; more splits increase short fragments and fewer fragments can merge
real syllables. Review the CQT and vocal-left/synth-right audio to choose.

The CQT updates after a short debounce, with raw onsets, measured pitch, extra
pitch-change proposals, and slur/tie/articulation note boxes. Zoom and pan affect
only the view. The note table includes measured voiced fraction, pitch agreement,
raw MIDI pitch and key-correction indicators.

**Render current A+B** produces a fresh full-excerpt vocal-left/synth-right
preview. Changing parameters stops/removes the old preview so it cannot be
mistaken for the new result. Click the CQT to seek after rendering. **Save/Load
preset** stores/restores parameter JSON. **Export** persists the currently
selected model's MIDI, WAV/FLAC, CQT plot, note/onset CSVs and a settings/provenance
JSON under the cache's `tuning_results` directory. Existing reviews are untouched.

Certainty is not an accuracy claim: CREPE periodicity is measured model evidence,
while note agreement is the fraction of voiced frames within half a semitone of
the measured note before key correction. The pitch tracker's original -45 dB
RMS gate remains in place in addition to the adjustable note-region energy gate.
Real passing/chromatic notes can be intentional, so key awareness is opt-in.

### Cache and reproducibility

The cache retains the original reviewed voiced pitch frames and fresh actual
CREPE periodicity. During verification, torchcrepe's frequency conversion was
found to add random SciPy pitch dither, which can change semitone rounding on
repeated runs. Reusing the reviewed contour avoids that comparison confound.
Freshly decoded pitch for previously unvoiced frames uses a fixed NumPy seed when
rebuilding. Source and reviewed-contour hashes plus library versions are checked;
a stale/mismatched cache fails explicitly.

```powershell
.\experimental\.venv\Scripts\python.exe -m experimental.tuner_server --rebuild --prepare-only
.\experimental\.venv\Scripts\python.exe -m pytest experimental\test_vocal_midi.py experimental\test_tuning.py -q
```

The first GUI load now also warms librosa/SuperFlux peak picking for the
energy-backed fallback. Initial JIT/import warmup can take noticeably longer
than later updates (especially while tests or other analysis run). The loading
status remains visible until cached results arrive.
Subsequent changes reuse neural activations and pitch evidence rather than
rerunning inference. The original baseline review must exist when rebuilding;
`--baseline` can select the matching reviewed report explicitly.

## Batch experiments

This folder is deliberately separate from the production application. No app
behavior or dependencies are changed. Generated audio, figures, models installed
in the experimental environment, and CSV/MIDI results stay local and are ignored
by Git. The runner reuses the app's existing CREPE pitch function read-only.

## Run

From the repository root, in PowerShell:

```powershell
.\experimental\setup.ps1
.\experimental\.venv\Scripts\python.exe -m pytest experimental\test_vocal_midi.py -q
.\experimental\.venv\Scripts\python.exe -m experimental.vocal_midi
.\experimental\.venv\Scripts\python.exe -m experimental.validate_outputs experimental\outputs\vocals_30s
Start-Process .\experimental\outputs\vocals_30s\review.html
```

Setup requires the app's existing `.venv` and Windows MSVC C++ Build Tools for
madmom. The experimental venv gets its own package installations and searches the
app's site-packages afterward through a `.pth` file; it does not install into or
modify the app's venv. Build prerequisites are installed before compiling the
commit-pinned madmom package. Python/library versions and model weight hashes
are recorded in each run's report.

Optional source/output/duration overrides:

```powershell
.\experimental\.venv\Scripts\python.exe -m experimental.vocal_midi `
    --source "C:\Users\steve\Music\How We Go On From Here Project\1_How We Go On From Here 150bpm exact core_(Vocals).mp3" `
    --output ".\experimental\outputs\vocals_30s" --seconds 30 --bpm 150
```

Re-running overwrites the files in the chosen output folder. Use a different
`--output` directory to preserve a previous experiment.

### Corrected working region: 39 seconds to the end

The corrected requested region is **39.0-65.6 seconds of the original MP3**,
or **26.6 seconds**. The decoded source ends at sample 2,892,960 at 44,100 Hz;
this selection includes its exact final sample without truncation or padding.
The same five methods and settings are used, with earlier comparisons preserved.

```powershell
.\experimental\.venv\Scripts\python.exe -m experimental.vocal_midi `
    --start 39 --seconds 26.6 --output ".\experimental\outputs\vocals_39_to_end"
.\experimental\.venv\Scripts\python.exe -m experimental.validate_outputs experimental\outputs\vocals_39_to_end
Start-Process .\experimental\outputs\vocals_39_to_end\review.html
```

Open [the corrected review](outputs/vocals_39_to_end/review.html) for all five
CQT/pitch/onset/note overlays, MIDI exports, and vocals-left/synth-right players.

### Earlier explicit 10-second excerpts

To repeat the same five models/settings on **25.5-35.5 seconds of the original
MP3**, preserving the first loudest-window comparison:

```powershell
.\experimental\.venv\Scripts\python.exe -m experimental.vocal_midi `
    --start 25.5 --seconds 10 --output ".\experimental\outputs\vocals_25p5_10s"
.\experimental\.venv\Scripts\python.exe -m experimental.validate_outputs experimental\outputs\vocals_25p5_10s
Start-Process .\experimental\outputs\vocals_25p5_10s\review.html
```

`--start` is an absolute timestamp in the source, **not** a timestamp relative to
the previous 30-second excerpt. It bypasses the loudest-window search. The entire
requested interval must exist; invalid or out-of-range intervals fail explicitly
rather than truncating. The report and review identify it as a selected excerpt.
Omitting `--start` preserves the original loudest-window selection behavior.

For **25.5 seconds on the previous review's playback clock**, the source start
is instead `14.763038548752834 + 25.5 = 40.26303854875283` seconds:

```powershell
.\experimental\.venv\Scripts\python.exe -m experimental.vocal_midi `
    --start 40.26303854875283 --seconds 10 --output ".\experimental\outputs\vocals_review25p5_10s"
.\experimental\.venv\Scripts\python.exe -m experimental.validate_outputs experimental\outputs\vocals_review25p5_10s
Start-Process .\experimental\outputs\vocals_review25p5_10s\review.html
```

Both interpretations have separate results: [original-source timestamp
review](outputs/vocals_25p5_10s/review.html) and [previous-preview timestamp
review](outputs/vocals_review25p5_10s/review.html). Each contains the same five
models, CQT overlays, MIDI, slur/tie/articulation labels, and stereo comparisons.
The latter extends 5.5 seconds beyond the end of the original 30-second excerpt.

## Selection and controlled comparison

The default source is the requested isolated vocal MP3. The runner decodes the
whole source at its original sample rate and chooses the exact maximum-energy
continuous 30-second window, considering **every original-sample-aligned window**.
Energy is the mean channel power, not downmixed power; stereo phase cancellation
cannot make a loud window look silent. It is not a 30-second grid or a peak sample.

The initial measured window is **14.763039-44.763039 seconds**. This is an
unquantized source position, not a beat-aligned edit.

All five candidates use:

- The same CREPE full contour, with one second of analysis context on either side
  where available. No key cleanup or beat-grid correction is applied.
- The same voicing/energy region mask and evidence-based note release policy.
- The same CQT: 36 bins/octave, C2 through just below C6, 22050 Hz analysis.
- The same median pitch assignment after skipping a short attack section.
- The same additive synthesizer with exactly the exported note times.

Onset comparisons:

| Candidate | What actually runs |
|---|---|
| librosa | Standard mel spectral flux, lag 1, no frequency maximum filter |
| SuperFlux | librosa's documented vibrato-suppression recipe, lag 2 and maximum filter size 3, 138 mel bands |
| madmom CNN | Actual pretrained multi-resolution Schlueter/Boeck onset CNN, not an untrained approximation |
| madmom BRNN | Actual eight-model offline bidirectional recurrent ensemble; this is not a second CNN |
| Pitch-aware | Sustained pitch-change hysteresis, voiced phrase entries, and SuperFlux retriggers gated by vowel-energy valley/recovery |

All parameters and raw onset timestamps are exported. Raw detector proposals are
shown even if voice gating/minimum note length prevents them becoming MIDI notes.
Voiced phrase entries can create notes without a raw detector proposal so that
missed attacks do not silently discard an entire sung phrase.

Neural peak picking uses a 0.30 probability threshold and **no smoothing**.
During validation, a 20 ms madmom smoothing kernel was found to reduce even a
0.988 CNN activation below threshold: its two-frame Hamming kernel is not
normalized. A regression test prevents this zero-onset failure.

The custom method specifically addresses legato pitch movement (which may have no
percussive attack), vibrato chatter, and same-pitch syllable rearticulation.
It is independent of the neural models; it does not train on this clip.

## Slurred, tied, and articulated

Each note's `link_to_next` describes its connection **to the next note**:

- **Slurred (S, green):** different pitches with exactly equal end/start times.
- **Tied (T, yellow):** same pitch with exactly equal end/start times.
- **Articulated (A, pink):** previous note ends before the next begins.
- **Phrase end (white):** final note, with no next note to classify.

This follows the requested WIDI timing convention. MIDI is written at 9600 PPQN
and 150 BPM without rhythmic snapping; connected boundaries share the exact
same tick, and note-off precedes note-on at that tick.

Long voicing gaps (60 ms or more) separate phrases. Before a detected internal
boundary, a 20 ms unvoiced interval or a sustained energy valley supports an
earlier release. The energy valley must be at least 6 dB below the preceding body
and 4 dB below the following recovery for 20 ms. Otherwise notes stay connected.
There is no invented fixed articulation gap.

The synth keeps a continuous oscillator/envelope across connected notes
(same-pitch ties have no amplitude retrigger), but restarts after actual gaps.
CSV note tables include the start/end evidence and measured gap in milliseconds.

These are **acoustic estimates, not musical ground truth**. Breath, consonants,
separation artifacts, reverb and overlapping singers can confuse both voicing
and articulation. A detected same-pitch tie can be an unnecessary onset split.
Review those before deciding to merge ties or integrate any policy into the app.

## Review outputs

Open [the review page](outputs/vocals_30s/review.html) after a run. It contains:

- One all-method CQT overview and one detailed three-panel CQT figure per method.
  The 10-second panels are easier to inspect than squeezing 30 seconds into one
  figure. Open the full-resolution images to zoom in.
- Cyan measured pitch, orange dashed raw onsets, and labeled exported-note boxes.
- RMS and onset-activation plots for each method.
- Stereo `*_AB.wav`: mono vocal **left**, additive synth **right**, with no shift,
  padding, or resampling of the final audio sample grid. Use headphones.
- Lossless `*_AB.flac` versions, embedded directly into the review page so
  browser players also work when the integrated preview blocks file-based audio.
  The page needs no web server and sends no audio to an external service.
- MIDI, note CSVs, raw onset CSVs, pitch frames, and the original stereo excerpt.
- `report.json`: source/window provenance, versions, model hashes, parameters,
  durations, descriptive counts, and all note records.
- `validation.json`: independent recomputation of the loudest window, output
  shapes, MIDI boundary/tick checks, and both audio-channel reconstruction errors.

Audio channel levels are peak-limited to 0.85; synth RMS is matched to the vocal
unless its peak limit intervenes. Original stereo vocals are downmixed only for
the left channel of the comparison, while the separate excerpt preserves them.
The synth is a local additive preview, not a rendered third-party soundfont.

No reference note labels exist yet. Counts and duration distributions do **not**
establish which detector is most accurate. Review onset precision, repeated
same-pitch syllables, slurs, and note releases before choosing a winner.

### Initial comparison

| Method | Raw onsets | MIDI notes | Slurred links | Tied links | Articulated links |
|---|---:|---:|---:|---:|---:|
| librosa | 73 | 76 | 20 | 6 | 49 |
| SuperFlux | 57 | 64 | 12 | 5 | 46 |
| madmom CNN | 105 | 104 | 36 | 15 | 52 |
| madmom BRNN | 89 | 92 | 35 | 8 | 48 |
| Pitch-aware | 92 | 91 | 35 | 4 | 51 |

Each candidate also has one final phrase-end note. These descriptive results
are for the requested 30-second excerpt with four PyTorch CPU threads. The
runner limits PyTorch to four threads in its own process to avoid excessive
CPU contention; it does not change the app's process or global environment.

## Sources and integration restrictions

- [librosa SuperFlux example](https://librosa.org/doc/latest/auto_examples/plot_superflux.html).
  This is the librosa recipe, not a claim of bit-identical original SuperFlux code.
- [madmom onset processors](https://madmom.readthedocs.io/en/latest/modules/features/onsets.html).
  CNN: Jan Schlueter and Sebastian Boeck, *Musical Onset Detection with
  Convolutional Neural Networks* (2013). BRNN: the offline model ensemble supplied
  by madmom; architecture and exact model files are recorded in the report.
- [madmom model licensing](https://github.com/CPJKU/madmom_models):
  **CC BY-NC-SA 4.0**. Do not integrate the pretrained weights into a commercial
  product without appropriate permission/licensing. This experiment does not
  redistribute weights or change the app.
