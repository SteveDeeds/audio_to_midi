# Correction-head benchmark (experimental)

Simulates a user's left/right clicks on rendered, separated songs and scores how much the
small correction heads improve the *held-out* part of each song. Nothing here is used by the app.

| File | Purpose |
| --- | --- |
| `prepare.py` | Render MIDI with FluidSynth, separate with Demucs, write `data/clips/<clip>/truth.json` |
| `import_atm.py` | Index an external `atm_training_set` batch (pilot or expanded) as clips, without copying audio |
| `harness.py` | Today's app behaviour under simulated clicks |
| `ensemble.py` | Onset head on several models' activations (`reg_*`, `pre_*` variants) |
| `pitch_head.py` | Per-note keep/reject head for piano and guitar |
| `metrics.py` | Duration-free musical score: F0.5 with graded wrong notes + salient recall |
| `crosssong.py` | Pre-train head priors across songs; cross-validate and report validation/test |

`data/` and `runs/` are git-ignored. Clip `truth.json` files written by `import_atm.py` hold
absolute paths to the dataset drive, so re-run the import on each machine.

## Running on a GPU machine

```powershell
git checkout experiment/correction-heads
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# CUDA build of torch (pick the index for the installed CUDA driver):
.\.venv\Scripts\python.exe -m pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"

# Point at wherever the dataset drive mounted:
.\.venv\Scripts\python.exe -m experimental.benchmark.import_atm E:\atm_training_set\expanded

# Slow step: run every model once per stem (cached in data/clips/*/cache):
.\.venv\Scripts\python.exe -u -m experimental.benchmark.crosssong extract --workers 3 --threads 4

# Fast steps (CPU only):
.\.venv\Scripts\python.exe -u -m experimental.benchmark.crosssong cv
.\.venv\Scripts\python.exe -u -m experimental.benchmark.crosssong final --lam 1 --implicit 0.25
```

piano_hr and CREPE use CUDA when it is available (`ATM_DEVICE=cpu` forces the CPU).
Keep the machine plugged in and set to never sleep during `extract`; a finished stem is cached,
so an interrupted run resumes where it stopped.

The dataset is Lakh-derived and marked *rights review required*: keep the audio and caches local.

## Fresh, three-part experiment

`fresh_experiment.py` is a separate experiment, not a continuation of the fitted
heads above. It never loads their weights, priors, tuning selections, clip IDs
or prediction caches. All previous experiments remain untouched.

The verified `balanced-200` batch contains **200 variants of 100 original source
groups**, split into 160/20/20 variants (80/10/10 source groups). Start all heads
from scratch using this batch's splits. Do not mix in the old 90-song training
run. The compositions were previously inspected, so this is a reset-model
holdout, **not an independent new-composition evaluation**.

Each song's entire bass, drum, piano or guitar family has one primary separated
stem, even when that is a differently named stem. Multiple families can share a
stem. This is a label/routing invariant: it does not force Demucs to put all
audible content in that stem. Assignment sums solo-calibration energy fractions
over the family's notes; it is calibration-assisted, not autonomous recognition.
The importer rejects per-note routing and source-group leakage. Full composition
IDs are hashed for variant-safe IDs in a new experiment directory.

The combined **other** stem is also an explicit transcription target, separate
from the family-level views. Its MIDI comes from the routed `targets\other.mid`,
not from an assumption about the stem's name. In this batch, 79 variants assign
22,154 guitar/piano notes to other, including 12 variants with both families.
Basic Pitch and piano-HR score that complete pitched target; other receives the
same head/input/edit comparisons as the named families. All 200 variants are
included: an unassigned other stem has an explicitly empty reference, and its
false positives are reported separately. Other results are not pooled with the
overlapping family results. Future batches routing percussion into other fail
explicitly until a separate drum-class target is added.

### Questions and comparisons

1. **Canned baseline:** raw Basic Pitch, piano-HR and ADTOF outputs, with their
   fixed decoders and no fitted correction model. This first screen uses the
   same assigned stem for a family across systems; it is not a rerun of every
   model in the older MT3 bakeoff.
2. **Offline head improvement:** compare single-model input, multi-model input
   on the assigned stem, and multi-model input across all six stems. Compare a
   logistic head, one hidden layer of 16 or 64 units, and a two-hidden-layer
   64/32-unit neural head. Report actual parameter counts. Drum heads initially
   use ADTOF, with same-stem versus cross-stem input.
3. **Learning from edits:** compare no adaptation, a bias update, a local
   kernel-residual update, last-layer updates, and all-layer updates anchored to
   the pretrained head. Test regularization 1/10/100 and budgets
   0/1/2/5/10/20/40. All policies receive the same left-to-right error corrections
   from the frozen head, only within the first 20 seconds. Explicit adds/deletes
   override generated notes and are checked for preservation.

Candidate notes come only from canned predictions; reference notes are never
injected into the query candidate pool. User-authored missing notes are allowed
only in the edit region. Candidate recall exposes this limitation. Features
contain model agreement, velocity/strength, timing, duration, pitch and lower
harmonic evidence. A head selects notes; it does not yet generate new query
onsets, estimate releases or add a new audio encoder.

Training-only source-group CV selects the input/head and adaptation settings.
Every variant of a composition stays in one fold, and feature normalization
uses fitting data only. Validation/test edits adapt only their own song; they
never update a cross-song prior. Test scoring requires an explicit unlock.
The first 100's learned artifacts are never used as initialization.

Primary scoring is exact-pitch/drum-class, one-to-one **raw onset matching within
50 ms**, on audio after 20.1 seconds. No truth-tempo grid is used. Durations are
not scored. GM drums map to ADTOF's five classes; unmapped reference hits are
counted separately. Quality flags come from the assigned routed stem's target;
legacy family/named-stem flags are retained separately in the manifest. All
families remain included, with results also summarized by routed quality flag.
The shared FIFO MIDI reader preserves zero-tick note attacks that PrettyMIDI
drops; the imported note count must match the verified batch exactly. Its
minimum editable note length does not affect the onset-only metric.
Reports include precision/recall/F1, paired head-versus-canned improvements,
adaptation gain, source-group regression fraction, worst/p10 gains, bootstrap
intervals, convergence flags, and update times. Bootstrap intervals are
exploratory, not nested/multiplicity-corrected. A positive mean is not a safety
guarantee; frozen adaptation can legitimately win.

### Running

From the audio-to-MIDI repository root, with the installed experimental
environment and existing canned checkpoints:

```powershell
$python = ".\experimental\.venv\Scripts\python.exe"
$dataset = "C:\Users\steve\OneDrive\Documents\GitHub\atm_training_Set\dataset\balanced-200"
$out = ".\experimental\benchmark\runs\fresh-family-heads"

& $python -m experimental.benchmark.fresh_experiment prepare --dataset $dataset --out $out
& $python -m experimental.benchmark.fresh_experiment extract --out $out
& $python -m experimental.benchmark.fresh_experiment baseline --out $out --split train
& $python -m experimental.benchmark.fresh_experiment train --out $out
& $python -m experimental.benchmark.fresh_experiment evaluate --out $out --split validation

# Only after freezing the experiment:
& $python -m experimental.benchmark.fresh_experiment evaluate --out $out --split test --unlock-test
```

`extract --limit 1` is an inference smoke check, not an experiment result. Missing
frozen predictions fail explicitly rather than silently dropping clips. Extraction
resumes only within this run, using audio hashes; it never consults legacy caches.
Piano-HR uses CUDA when available (`ATM_DEVICE=cpu` overrides it). Neural heads
run on CPU with bounded L-BFGS iterations; budget exhaustion is logged and
reported rather than described as convergence. The optimizer allows up to 100
line-search steps per iteration to handle ReLU kinks during anchored edit
updates; abnormal termination remains a fatal error. Optimizer changes require
a new experiment manifest and fresh fits. Only unchanged canned predictions
may be copied into that run, after checking model/audio provenance; never copy
fold results, selections, or learned heads.

F1 weighs a wrong note the same as a missed note. To favor few wrong notes, run
`python -m experimental.benchmark.threshold_sweep --out $out` after `train`. It
scores `TP - cost * FP` (default cost 2, also 1 and 3), picks each fold's
canned-confidence or head-probability threshold on the other training folds, and
writes `precision_thresholds.json`. With calibrated probabilities the ideal keep
threshold is `cost / (1 + cost)` (0.67 for cost 2). Cached canned output can only
be thinned, not made more permissive, without new inference.

Edit adaptation can over-correct when it trains only on the user's explicit
fixes, a biased sample with no confirmed-correct notes. Run
`python -m experimental.benchmark.adaptation_study --out $out --role <role>` per
role (add `--resume` to keep finished architecture/fold fits from
`adaptation_study/<role>.partial.json`). On the selected input config and the same
source-group folds, it compares linear, MLP 4/16/64x32, a 256x128x64 DNN and
gradient-boosted trees, each with several edit updates. The original update uses
only the explicit fixes. The *reviewed-window* updates assume edits are made in
time order, so every candidate before the k-th edit's onset has been reviewed
and keeps its implicit label. Policies:

- anchored all-layer update
- last-layer update
- Platt recalibration (two parameters)
- gated variants that accept an update only if it lowers log loss on the most
  recent third of the reviewed notes

Outputs are training CV only. The window assumption is optimistic, and any
policy chosen from this study still needs confirmation on the locked validation
split.

The primary metric is now **song-pooled F0.5** (`scoring.py`), because a wrong note
is worse than a missed note:
`F0.5 = 1.25 TP / (1.25 TP + 0.25 FN + FP)`. A wrong note weighs four times a miss,
the score stays in [0, 1], and 1 still means every note found with nothing extra.
TP/FP/FN are pooled over each source song's views before scoring and songs are
averaged equally. A song with no reference notes scores 1 if left silent and 0 if
anything is predicted. F1 is still reported for continuity (`f1_song_pooled`).

- `python -m experimental.benchmark.fbeta_report --out $out` rescores the canned
  models (native output and an F0.5-tuned confidence cut chosen on the other
  folds), every head config/architecture in `cv_rows.json`, and the F1 adaptation
  study from stored counts. It writes `fbeta0.5_report.json` and refits nothing.
- `adaptation_study --beta 0.5 --study adaptation_study_f05` tunes each fold's head
  keep threshold for F0.5 on the other folds' out-of-fold probabilities. The
  threshold controls both what the simulated user sees (so which edits they make)
  and what is scored.

For a cheaper initial screen, use `train --architectures linear,mlp16`; use a new
output directory for a different registered experiment. Preparation refuses to
overwrite an existing experiment, fitted runs cannot be silently retrained,
partition results cannot be overwritten, and code/dataset manifest changes
invalidate the run. Full extraction and CV are long-running jobs; the synthetic
test fixture verifies the three-stage machinery, not real transcription gains.

The original untrained family-only baseline run can be revised once with
`upgrade-other --out $out`. This narrowly versioned migration preserves frozen
canned predictions after checking model/audio hashes and unchanged family labels,
splits and inputs. It backs up the old manifest, records cache provenance, and
refuses any run that has already fitted heads or scored a partition. Resume
`extract`, then `baseline --split train` as before; no old fitted state is reused.

## Actual-mixture note matching pilot

`matching_pilot.py` is a separate, train-only review workflow. It does not change
whole-family routing, the existing importer, cached predictions or fitted runs.
It selects 12 distinct original-pitch training source groups, alternating more
and less flagged examples. Validation/test songs and augmented siblings are not
used. Preparation reads existing audio only: no rendering, separation, checkpoint
download or model training. The optional contextual player renders single-note
diagnostic cues, never new training references or complete compositions.

```powershell
$python = ".\experimental\.venv\Scripts\python.exe"
$dataset = "C:\Users\steve\OneDrive\Documents\GitHub\atm_training_Set\dataset\balanced-200"
$out = ".\experimental\benchmark\runs\matching-pilot"

& $python -m experimental.benchmark.matching_pilot prepare `
  --dataset $dataset --out $out --songs 12 `
  --predictions .\experimental\benchmark\runs\fresh-family-heads
& $python -m experimental.benchmark.matching_pilot serve --out $out --port 8767
# Open http://127.0.0.1:8767 in a browser.
```

For each original MIDI note, a 10 ms preroll / 250 ms post-onset window is
compared against **all six stems in the actual mixture**. Stereo-energy spectra
avoid phase cancellation. Nonnegative spectral fits against all clean family
references estimate local retention; melodic comparisons use eight harmonic
bands around the MIDI pitch, while drum comparisons use the broad attack spectrum.
Scores combine spectral similarity and bounded retention, **not calibrated
probabilities**. Default 0.30 score / 0.15 margin screens only prioritize review;
they are not validated acceptance thresholds.

Optional frozen Basic Pitch, piano-HR and ADTOF predictions corroborate
pitch/class onsets with one-to-one matching within 50 ms. Missing caches are
explicitly logged/reported; mismatched audio or metadata hashes fail. No old
head, learned prior or tuned threshold is reused. ADTOF kick 35/36 aliases
are mapped through the existing five-class helper.

The UI shows the original MIDI piano roll, local evidence and passage playback
of mix/clean/two selectable stems, at original gains. Reviewers must select
audible destinations explicitly: one, several, absent or uncertain. Nothing is
pre-confirmed. Its starting queue offers two harder/clearer notes per family per
song (96 notes for the 12-song pilot), with all notes still available. This is a
convenience sample, not a random accuracy benchmark.
Use **Replay clean reference** to identify the highlighted note, then replay each
candidate stem at the same passage. The reference is a whole clean family, not
an isolated single-note render. Source velocity, MIDI duration and measured
clean-window RMS appear beside the piano roll; very quiet references should be
marked uncertain, not absent. Replay buttons restore a muted/zero-volume player
and always restart the passage. Native play also restarts after the passage end;
passages include the selected note's MIDI duration and a short release tail.
**Restore player sound** resets all browser players without changing stored audio.
For note-level listening, prefer **Click + isolated note**, then **Click + stem A/B**.
The isolated cue contains exactly the selected MIDI attack with the original
program, channel controls, pitch, velocity and MIDI duration, rendered using the
hash-checked original SoundFont and local FluidSynth. It is peak-normalized to
-12 dBFS for audition only, cached under `audition-cues`, and is **not recovered
clean audio or a training reference**. Whole-voice compression and overlapping
notes are not reproduced. Missing tools, silent renders and hash mismatches fail
explicitly rather than producing a substitute tone. `serve --fluidsynth <path>`
overrides the locally recorded executable when necessary.

Context playback schedules audio on a single Web Audio clock: a synthesized
metronome slightly left (-0.25 pan), and the isolated cue or actual stem slightly
right (+0.25). The cue attack is placed at the original MIDI onset within the
same passage. Clicks follow the hash-checked original MIDI tempo and meter:
a higher tick on each downbeat, lower tocs on the other beats (tick-toc-toc-toc
in 4/4). Notes are not snapped to beats. The click-level slider applies to both
comparisons; shared playback attenuation provides sample-peak headroom.
No clean drum layer is added, even when comparing the actual drum stem.
Selection changes/Stop cancel stale
loads and playback. Only four decoded buffers are retained. The source audio
and MIDI are unchanged; these are single-note diagnostic re-renders only, not a
new composition build or model-training run.

Earlier reviews may have judged whole instrument parts. They are preserved but
shown in **Earlier decisions: recheck exact note**, and excluded from training
export until reconfirmed with **Save exact-note review**. That button itself
acknowledges exact-note review; there is no separate checkbox to overlook.
Exact-note decisions carry
`scope: "exact_note"`; export/consumer manifests require that scope. The original
pilot's report/review hashes remain valid only while its pinned matching-function
fingerprint is unchanged. Playback/UI upgrades do not rewrite old evidence or
silently promote earlier decisions.
Each note also has a **Comment for this note** field (up to 4,000 characters).
**Save comment only** persists an annotation without confirming a note or
changing its destination. Saving a review saves the comment too; Undo review
keeps the saved comment. Comments are included in decision downloads and export
provenance, never treated as training labels.

The sticky save banner turns green only after the server acknowledges an atomic
disk save, and shows its timestamp, note identity and revision. A separate status
checks server connectivity every 15 seconds. Requests time out explicitly, and
**Reconnect / refresh saved data** reloads server state without discarding local
drafts. Network failures mean "save not confirmed", not success.

Changed selections/comments remain browser-local drafts keyed by report and
note ID, survive navigation/reload, and appear in **Unsaved drafts**. They never
count as saved reviews. Closing/reloading warns while drafts or an in-flight save
exist; navigation controls lock during a save. Saving a comment alone leaves any
uncommitted stem selections as drafts. Do not clear browser storage before saving
or recovering drafts. Checking boxes alone does not save a review: click a clearly
labelled Save action and wait for its receipt.
Decisions save atomically to the pilot's `reviews.json`, keyed
by stable source-note IDs and the report hash; concurrent edits require reload.
Downloads provide an additional review backup. The loopback-only server streams
only manifest-listed audio and supports byte ranges. Do not expose it externally.

**Limitations:** clean family windows can contain chords, other notes and release
tails. Overlap and similar clean-family spectra trigger warnings. Magnitude
spectra and canned predictions cannot establish note attribution, especially
for unisons/octaves. MIDI offsets and velocities remain source metadata, not
verified separated-audio releases or loudness. This is a review tool, not a new
ground-truth claim or an accuracy result.

After actual human review:

```powershell
& $python -m experimental.benchmark.matching_pilot export `
  --out $out --destination "$out\reviewed-positive-onsets.json"
```

Export refuses empty reviews and overwrites, rechecks source audio/MIDI hashes,
and includes **only human-confirmed positive onsets**. Notes may appear in
multiple stems; ambiguous/absent/unreviewed notes remain unknown. Unlabelled
stems and unreviewed times are **never implicit negatives**. Melodic pitch and
five-class drum targets are separate even in mixed stems. These are positive-only
weak supervision, **not a drop-in dense-label dataset for `fresh_experiment`**.
`load_positive_supervision()` provides the consumer contract and requires explicit
rights clearance before training. Clean-baseline MIDI/audio paths are preserved,
but no training or validation/test evaluation is performed by this pilot.

Validate matching and the review/export contract with:

```powershell
& $python -m pytest -q tests\test_matching_cues.py tests\test_matching_pilot.py tests\test_benchmark_harness.py tests\test_fresh_experiment.py
```
