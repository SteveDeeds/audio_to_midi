# Audio-to-MIDI v2: decision and execution plan

## Implementation status (2026-10-18)

A runnable side-by-side **v2 preview** now exists in [audio_to_midi](../audio_to_midi).
The legacy entry point is unchanged. This is not a cutover or a claim that the
human acceptance/release gates below have passed.

| Plan area | Implemented | Remaining release evidence |
|---|---|---|
| Domain/persistence | Exact quarter-note timing, meter/triplets, identified notes, authored overrides, bounded edits, undo/redo, versioned atomic projects, explicit missing-media/schema errors | Larger real-project recovery/acceptance sessions |
| External-stem editor | Role-assigned multi-track projects, per-track models/training, automatic visible-window inference, progressive whole-track publications, cancellation/stale-result rejection, note/onset editing, audio/MIDI preview | User A/B approval across the reference stems and hardware playback audition |
| Interchange/harmony | Tempo/meter/velocity/GM MIDI, starting MIDI including tempo changes, key overrides, mix/available-stem harmony, chord-note editing/import/export, compatible legacy onset-profile import | MuseScore/Ableton verification; full quality regression corpus |
| Preparation | GUI visual beat editor (CQT/waveform, confidence-weighted candidates, pinned or tapped downbeats with meter guess, phase-locked counting, tempo fit/drift, beat-click audition), persisted beat candidates/manual anchors, forward/inverse maps, explicit lead-in policy, memory-bounded streaming stereo warp, immutable artifact revisions, edited-downstream protection | Learned downbeat-backend evaluation and transient/listening approval |
| Separation | Optional offline Demucs subprocess, consent/local checkpoint requirement, cancellation/output validation/cache, import into empty projects | Actual installed Demucs/checkpoint inference and quality/license review |
| Installation/cutover | Isolated base-environment installer, read-only environment doctor, clean-environment tests, legacy remains usable; no data removal or automatic upgrade | Optional backend clean-machine qualification and human replacement gates |

The domain and application layers do not depend on `AudioMarkers`. Proven
providers remain in the legacy modules and are adapted, not forked. One small
shared extraction, [onset_features.py](../core/onset_features.py), lets both
applications use the same existing candidate-feature formulas. Compatible
legacy onset examples are imported; unsupported profile fields are reported
rather than presented as migrated edits or new-model training.
The shared Basic Pitch adapter now explicitly selects its bundled ONNX checkpoint,
so installing another framework cannot silently switch the inference backend.

Important boundaries:

- Learning is isolated per track/project. Cross-song learning is explicit profile
  import, not automatic.
- Stems with mismatched rates/durations are rejected with instructions to
  explicitly normalize them externally and audition alignment; there is no
  silent resampling, padding, or individual-stem warping.
- Externally warped mixes can be registered verbatim for optional separation.
  They are never re-warped and have no invented original-time map.
- Librosa supplies **beat candidates only**, not inferred downbeats or meter.
  Rendering requires confirmed timing and at least two source beat anchors.
- The experimental renderer streams bounded source windows through a continuous
  shared stereo phase trajectory and rolling overlap-add output. It no longer
  rejects full songs merely for their duration: a 120-second stereo input
  rendered to 130 seconds within a tested 16 MiB working-memory budget.
  Transient/listening acceptance is still required before replacing Ableton.
- Piano-HR's previously documented batch-boundary limitation remains; prefer
  Basic Pitch for sustained-note work until that backend passes continuity
  acceptance. Bass sustained-change splitting has automated coverage but still
  needs the planned listening approval.
- MIDI export refuses unprocessed/stale tracks unless partial export is
  explicitly accepted. Project manifests, not MIDI files, preserve edit state.

Validation includes the unchanged 174-test legacy baseline, domain/recovery
tests, synthetic stereo-warp tests, CLI round trips, real pYIN transcription of a
synthetic A4, and an offscreen Qt editor suite. The GUI suite runs in its own
process when the legacy tests own a non-widget `QCoreApplication`.
Reverified on 2026-10-23: the full runner passes 303 top-level tests plus 24 tests
in that isolated GUI process; Python compilation, CLI help, the environment doctor,
and installer PowerShell syntax checks also pass. Pre-commit regressions cover
original/preparation-media overwrite protection for MIDI and MusicXML exports,
replacement-stem alignment validation before mutation, rejection of chord results
after newer edits, and preservation of revision protection during background stages.
A separate CLI subprocess
round trip verifies create -> external-warp registration -> attachment -> stem
import -> starting MIDI -> export/reopen inspection.
All approved model families were also exercised locally on four-second reference
stem clips using the already-installed checkpoints: CREPE tiny/full bass,
pYIN/CREPE full vocals, Basic Pitch piano/guitar, piano-HR, and ADTOF. An audible
drum clip produced 25 proposals; a quieter comparison clip produced no proposals.
These are actual inference smoke checks, not labeled quality/F1 approval.
No reference audio or pretrained weights were added to the repository.
On this machine, 20 warm edits of a two-second synthetic pYIN vocal clip measured
p95 **8.98 ms edit-to-paint** and **187.93 ms edit-to-cached-retrain-publication**
(including the 150 ms debounce). A separate 2,000-proposal domain case measured
p95 32.89 ms author+resolve; warm logistic retraining measured 6.52 ms. These are
measured synthetic cases, not certification of whole-song/model/hardware latency.

### Run the preview

From the repository root, using the existing environment:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m audio_to_midi
```

The current GUI starts with a full mix: create/open a project, import the mix,
count its bars, warp it (or use it as-is), and confirm timing before importing
or separating stems. Set tempo, meter, and key from the Edit menu. Displayed BPM
is counted in the meter's beat (1/denominator); storage and MIDI use quarter BPM.
The Grid selector uses musical note values; stored subdivisions are in quarter
notes (`1/4` for straight sixteenths, `1/6` for sixteenth triplets).
See [Getting started](getting-started.md) for the current stage-by-stage GUI flow;
the headless external-stem workflow remains available below.

The CQT-backed piano roll adds (click/drag), resizes (drag end), selects, and
removes (right-click) notes. The Info tool opens an exact-value note editor;
right-clicking empty space offers rests. The onset lane adds/deletes onsets.
Save/reopen preserves edits,
rejections and learning examples; undo/redo history is session-local. Preview
renders notes into the stem's audio buffer by default (sample-aligned); an
optional external MIDI output uses a per-machine latency offset set with a
click/wood-block sync test.

Missing models (CREPE, Basic Pitch, piano-HR, ADTOF, Demucs) are pip-installed into
the running Python and their weights downloaded on first use, after one confirmation
(`audio_to_midi/provision.py`); ADTOF and piano-HR prompts include their license notes.
The legacy application still launches with `python app.py`.

### Isolated installation

The lightweight installation is sufficient for the editor, MIDI, pYIN, and
preparation. It deliberately leaves the legacy environment untouched:

```powershell
.\install-v2.ps1
.\.venv-v2\Scripts\python.exe -X utf8 -m audio_to_midi doctor
.\.venv-v2\Scripts\python.exe -X utf8 -m audio_to_midi
```

Add neural transcription backends with `.\install-v2.ps1 -Models`. The optional
manifest constrains resampy to Basic Pitch's supported metadata range, and the
installer installs Basic Pitch separately with `--no-deps` for ONNX inference.
Its upstream mandatory TensorFlow metadata still makes `pip check` report
TensorFlow absent in that optional configuration; do not install incompatible
TensorFlow merely to silence it. Actual ONNX inference is separately verified.

A new isolated Python 3.12.10 environment successfully installed only
[requirements-v2.txt](../requirements-v2.txt), passed all 93 v2 tests (including
actual pYIN, GUI, streaming-renderer and interchange tests), and passed
`pip check` with no broken requirements. It used newer independently resolved
librosa 1.0.0, NumPy 2.5.4, SciPy 1.18.1 and PySide6 6.12.0, in addition to the
legacy environment's verified versions. Neural backends are optional and were
tested in the existing environment, not newly downloaded into the clean one.

Beat editing lives in the main window's **Mix** view (`beats.py`, `timing.py`,
`timeline.py`; see [ui-redesign-plan.md](ui-redesign-plan.md)); it replaced the
former **Prepare mix** dialog described here. It shows a CQT and waveform with
beat candidates. The user pins downbeats (bar/beat) by hand or with **Tap downbeats**:
taps made during playback get a lag correction, snap to the nearest strong
candidate, and set the meter from the pulses per tapped bar. The tool counts the other beats
phase-locked from the pins, fits the tempo, and reports jitter and drift. Steady
mixes are registered verbatim at the exact fitted BPM; drifting mixes are warped
using smoothed per-bar anchors (or every counted beat). The main transport plays
the view with accented beat clicks and auditions output; preparation revisions
are attached within the current project. **Separate stems** asks once, installs
Demucs and downloads `htdemucs_6s` on first use, then imports six stems.
Both run on the cancellable background lane; neither requires command-line use.

### Headless workflow and preparation

Examples below use arbitrary local paths; no reference audio is redistributed.
See `python -m audio_to_midi --help` and each subcommand's `--help`.

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -X utf8 -m audio_to_midi create C:\Music\song --name Song --bpm 75 --meter 4/4 --grid 1/6 --key 'Bb major'
& $py -X utf8 -m audio_to_midi add-stem C:\Music\song bass C:\Audio\bass.wav --model crepe_tiny
& $py -X utf8 -m audio_to_midi transcribe C:\Music\song bass
& $py -X utf8 -m audio_to_midi export C:\Music\song C:\Music\song\song.mid
```

Use `import-midi <project> <role|chords> <file>` for starting notes/chords,
`chords <project>` for baseline harmony, `info <project>` for completion, and
`export --allow-partial` only when partial results are intentional.
`export --role bass` exports one stem; `export --role chords` exports only chords.
The GUI export scope offers the same choices. Starting MIDI also supplies
quantized positive boundary and negative silent-region examples to the onset
learner without turning interior held-note grid points into negative boundaries.

For original-mix preparation, an anchors JSON file is a list of pairs such as
`[[0.32, 0], [1.11, 1], [1.93, 2]]`: source seconds and quarter-note positions
relative to the confirmed downbeat. Inspect candidates first; correct the
anchors in this plain file before explicitly requesting a render:

```powershell
& $py -X utf8 -m audio_to_midi prepare C:\Audio\original.wav C:\Music\prep
& $py -X utf8 -m audio_to_midi prepare C:\Audio\original.wav C:\Music\prep --bpm 75 --meter 4/4 --downbeat 0.32 --anchors C:\Music\anchors.json --lead-in trim
```

Alternatively, **bypass warping entirely** with an externally warped mix:

```powershell
& $py -X utf8 -m audio_to_midi register-warped C:\Audio\warped.wav C:\Music\prep --bpm 75 --meter 4/4
& $py -X utf8 -m audio_to_midi attach-prepared C:\Music\empty-project C:\Music\prep
```

Separation runs Demucs `htdemucs_6s` in a subprocess of the same Python; Demucs and
its weights are installed/downloaded automatically on first use:

```powershell
& $py -X utf8 -m audio_to_midi separate C:\Music\prep --project C:\Music\empty-project
```

Attachment/import requires an empty project to avoid rebasing existing edits.
Explicit `--new-revision` preserves old artifacts while invalidating affected
descendants; attached project edits protect their preparation revisions. Keep
old projects/revisions to retain the original timing/edit relationship.

## Decision

**Build a replacement application core alongside the current app, in this
repository. Do not do a large in-place refactor, and do not start a separate
repository.**

This is the handoff's Option B, delivered incrementally rather than as a
big-bang rewrite. It is a new implementation of the architecture, not a new
product or a fresh investigation of transcription algorithms. Keep the current
application runnable until v2 passes the replacement gates.

The first release target is deliberately narrow: **import externally warped
audio/stems, edit per-stem notes, save/reopen the project, and export MIDI.**
Automatic beat tracking, warping, and separation follow that working slice.

### Why this is the right boundary

| Approach | Assessment |
|---|---|
| Major in-place refactor | Preserves the existing shell, but repeatedly changes seconds-based grids, single-active-stem state, note derivation, workers, and UI together. High coupling makes each intermediate migration expensive. |
| Separate project/repository | Provides isolation, but separates the proven tests, models, history, and comparison application. No verified platform or product requirement justifies that split. |
| Side-by-side replacement in this repository | Creates clean domain boundaries while retaining the working app, regression specifications, reusable algorithms, and an immediate A/B reference. Recommended. |

Evidence from the current code:

- [AudioMarkers](../core/audio_engine.py) combines feature extraction, model
  coordination, note assembly, user state, and persistence. Repackaging it behind
  a new name would preserve the core problem.
- [SongSession](../core/session.py) is an in-memory, bounded cache of analyzed
  stems, not an on-disk multi-track project. It is useful prior art, not the new
  project model.
- [PitchedNoteSpan](../core/audio_engine.py) and authored-span overrides already
  give some notes identities. The gap is not "no note objects at all": generated
  notes still depend on assembly and scattered edit state. Preserve the behavior,
  not that state layout.
- [MonoTranscriber](../core/mono_pitch.py), [PolyTranscriber](../core/poly_pitch.py),
  and [ADTOF](../core/adtof_drums.py) are already separate enough to adapt.
- [LearningRuntimeState](../core/learning_runtime.py) and
  [InferenceCache](../core/inference_cache.py) already separate scheduling and
  stale-result rejection from generated predictions.
- [MIDI import](../core/midi_export.py) reads tempo events and meter; export is
  fixed-tempo and does not write meter. Extend it rather than discarding it.
- The existing [tests](../tests) encode substantial editing behavior, but many
  construct private legacy state. Port their assertions to public v2 APIs rather
  than copying those fixtures.

Reviewed baseline: `56a1d48`, 2026-10-18. All **174 tests pass** using:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p 'test_*.py'
```

Explicit discovery works without a test-package initializer. Without UTF-8 mode,
two tests error when MIDI export prints an arrow to this console's cp1252 stream.
This is a pre-existing output-encoding issue, not evidence of failed MIDI timing.
The unit baseline does not certify real-model quality, GUI usability, or model
installation on a clean machine.

## Architecture and non-negotiable contracts

### Keep, adapt, replace

- **Keep algorithms and their behavior:** approved mono/poly/drum backends,
  key cleaning, onset features and lightweight learning, chord baseline,
  evaluation metrics, and MIDI/playback building blocks.
- **Adapt integration:** scheduling, cancellation, versioned caches, playback,
  model snapshots, and reusable Qt drawing components.
- **Replace ownership/orchestration:** `AudioMarkers`, global mode state,
  note/edit persistence, project lifecycle, and main-window workflow logic.
- Do not port NNLS or its learning machinery into the v2 critical path. Keep it
  available in the legacy app; add a v2 fallback only if an actual use case
  warrants the maintenance cost.
- Keep Python and PySide6 initially. No frontend/framework migration, new
  transcription model research, or pretrained-model retraining in this project.

Use a distinct `audio_to_midi` package with `python -m audio_to_midi` as its
eventual entry point. Leave [app.py](../app.py), [core](../core), [ui](../ui), and
[workers](../workers) intact during migration. Organize the new package into
domain, application, infrastructure, and UI responsibilities, introducing
modules only as the vertical slices need them.

Initially adapt proven legacy modules without modifying their behavior. New
domain/application code must not depend on `AudioMarkers` or legacy UI.
Move reusable implementations and their tests into the new package only after
their adapters pass parity tests; do not maintain two algorithm forks.

### Time model

- Musical positions use **quarter-note units**, with exact grid subdivisions
  suitable for straight and triplet rhythms. Meter is numerator/denominator;
  BPM always means quarter notes per minute. Displayed compound-meter pulse is
  separate, so 12/8 cannot accidentally mean twelve quarter notes per bar.
- Audio providers and playback keep sample/second coordinates. Convert at
  explicit boundaries; do not rewrite pretrained backends into beat arithmetic.
- Distinguish the **source beat/tempo map** from the **fixed-tempo project grid**.
  A monotonic warp map connects original and rendered sample positions.
- Persist source/downbeat offsets and any removed lead-in. State explicitly
  whether pre-bar-one audio is trimmed or retained before rendering; never
  silently discard it. The normalized editing timeline starts at bar 1 beat 1.
- Externally warped material has a declared fixed grid and provenance, not an
  invented original-time warp map. An offset nudge changes shared alignment
  metadata; it must not independently stretch stems.

### Notes, edits, and learning

- Each track owns identified notes, onset events, explicit rests/rejections,
  model settings, and training examples. Separate generated proposals from
  authoritative user edits; resolved notes are the shared editor/export view.
- Edit commands capture their affected interval and support undo/redo. The
  visible-window bound for onset deletion is command input, not hidden UI state
  that changes later during save, export, or background inference.
- Keep onset detection separate from pitch evidence. Both use the selected
  quantization. No pitch-bend MIDI. Sustained mono pitch changes split notes;
  transitional bend-ins and falls do not create fragments.
- Applying the sustained-change rule to bass is an intentional extension of
  current behavior: add dedicated tests and A/B approval, not a silent port.
- Manual notes, releases, rests, and rejected pitches survive inference,
  retraining, navigation, track changes, and save/reopen. A held polyphonic voice
  continues across other voices' onsets; a same-pitch re-strike splits.
- Workers publish immutable results with project/track, asset, settings, edit,
  and model revisions. Stale results never overwrite newer state.

### Project and stage artifacts

A project folder contains a versioned manifest, song/timing metadata, media
references or managed copies, editable chords, per-track state, model snapshots,
and derived outputs. MIDI is an interchange artifact, not the edit database.

Persist input identities, parameters, backend/checkpoint versions, completion
state, and dependencies for each stage. Use atomic state-file replacement and
write derived media to temporary outputs before publishing completion.
An interrupted stage must never appear complete.

Supported entry points: original mix; analyzed song metadata; already-warped
mix; external stems with or without a mix; chord MIDI; per-track MIDI. Missing
stem roles are allowed; `other` is harmony context only and has no MIDI track.
Without a mix, ask for timing and use available pitched stems for harmony.

Validate external stem rate, duration, and shared alignment. Matching lengths
alone do not prove synchronization: include an audition/check step. Resample or
pad only with explicit warning/confirmation; never silently stretch mismatches.
If stems have no mix, use a user-confirmed common duration and origin.

Stage dependencies are explicit, not merely a numbered list:

```text
source + corrected beat anchors -> warped mix -> separated stems
warped mix + key                -> baseline chords
stems + timing + track settings -> per-track notes -> MIDI
other/available pitched stems   -> optional harmony/chord refinement
```

Initial chords must not require stems that do not yet exist. Optional refinement
must preserve manual chord edits. External artifacts can satisfy dependencies
without running upstream stages.

An upstream change marks affected outputs stale and previews affected edits.
Keep old artifacts and edit state until the user explicitly chooses a new
revision/rebase; never silently delete or pretend old edits remain aligned.

## Execution sequence

Each phase ends in a demonstrated outcome and a reviewable set of changes.
Do not begin automatic pipeline work before the external-stem milestone passes.

### Phase 0 - Freeze the behavioral specification

1. Record the legacy revision and reproducible test command; keep legacy tests
   running throughout migration.
2. Build a behavior checklist from note-span, mono/poly/drum, worker/cache,
   profile, playback, and MIDI tests. Distinguish public behavior from obsolete
   implementation details.
3. Create short, locally held reference clips and expected edits/exports for
   bass, vocals, piano/guitar, drums, triplet feel, and compound meter. Include
   sustained notes crossing analysis batches. Use cleared/synthetic fixtures
   for anything committed; do not redistribute the local commercial song.
4. Capture legacy inference quality and edit-loop timings on the same hardware
   and clips. Record model versions and which measurements are cached.
5. Confirm licensing/release intent and per-song versus cross-song learning
   before their affected milestones.

**Exit:** passing baseline, explicit parity checklist, reference cases, and
recorded performance measurements. No architectural code changes required.

### Phase 1 - Domain, timing, and durable state

1. Add project, track, timing/grid, note/onset, and edit-command types with no
   Qt or neural-backend imports.
2. Implement straight/triplet snapping, source/project conversions, meter
   labeling, bounded onset-delete/merge, release edits, rests, and undo/redo.
3. Add versioned project serialization and safe reopening, missing-media errors,
   relative managed paths, provenance, and explicit schema-version rejection.
4. Keep training labels and authored state independently recoverable from
   disposable inference caches. Inference status must distinguish unprocessed,
   processed-empty, and failed intervals.

**Exit:** headless tests prove timing conversions, edit invariants, and exact
save/reopen of notes, overrides, and track settings. An interrupted save leaves
the previous valid project readable.

### Phase 2 - First usable vertical slice: external bass stem

1. Import an externally warped bass stem, optional mix, BPM, meter, and offset.
2. Adapt one approved mono backend and the lightweight onset model. Implement
   visible-window priority, cancellation, cache reuse, and stale-result checks.
3. Add the minimal Qt shell: project open/save, waveform/onset lane, note editor,
   model choice, progress/errors, and audio/MIDI preview.
4. Make edits update authoritative state immediately and queue learning and
   inference off the UI thread.
5. Export fixed-tempo MIDI with meter, velocity, and ordered note-off/re-strike
   events from the same resolved note view shown in the editor.

**Exit:** import -> transcribe -> edit -> preview -> save -> reopen -> export
works on the bass reference clip. No `AudioMarkers` dependency in the new domain
or application layer; cold inference is never disguised as completion.

### Phase 3 - External multi-stem workflow: replacement MVP

1. Add multiple role-assigned tracks, missing-role handling, shared alignment,
   per-track models/edits, and synchronized navigation/playback.
2. Adapt vocals, polyphonic backends, and ADTOF with their existing correction
   behavior. Verify cache/training isolation between tracks.
3. Port mix/all-pitched-stem key detection, manual key override, editable block
   chords, imported chord MIDI, and per-stem starting MIDI.
4. Add per-track/chord exports and, if provided, a combined MIDI with a shared
   conductor tempo/meter track and GM percussion channel.
5. Finish full-track processing/export completeness checks. Export must either
   finish the requested range or explicitly offer a labeled partial export.
6. Add a legacy profile importer for compatible model settings/examples only.
   Report unsupported fields and leave original profiles untouched. Do not
   promise full project recovery from files that never stored full edit state.

**Exit:** the user's current Ableton + external-separation workflow is usable
end to end in v2 and passes parity/A/B checks on all supported roles. This is
the first release candidate; automatic preparation is not a prerequisite.

### Phase 4 - Original-mix analysis and full-mix warping

1. Evaluate beat/downbeat candidates on the reference clips for accuracy,
   correction effort, CPU performance, Windows installation, and licensing.
   Select a backend from evidence, not the handoff's candidate list alone.
2. Provide beat/downbeat editing, meter/BPM confirmation, and lead-in policy.
3. Evaluate a continuous, anchor-driven warp renderer. Avoid independently
   stretching/concatenating beat slices without testing for seams and transients.
4. Render only the full mix, with persisted forward/inverse mapping, corrected
   downbeat at zero, and fixed target tempo. Do not warp individual stems.
5. Add stage revisions, cancellation/resume behavior, and edited-downstream
   invalidation warnings before exposing reruns.

**Exit:** synthetic anchors map within one output sample; representative real
renders pass listening checks for seams, pitch changes, and transient damage.
Compare beat alignment against the existing externally warped references.
External-warp entry remains fully supported.

### Phase 5 - Optional in-app separation

1. Evaluate `htdemucs_6s` installation and quality on the target Windows setup.
   Keep its dependencies optional; use a separate worker environment if the
   measured dependency constraints require it, not as an assumed architecture.
2. Run separation only on the normalized warped mix. Cache by input identity,
   model/checkpoint, and settings; publish stems only after output validation.
3. Integrate progress, cancellation, restart, role mapping, and shared-grid
   checks. Warn about rerunning separation when track edits already exist.
4. Exercise the entire original mix -> analysis -> warp -> chords/stems ->
   per-track MIDI path, including starting again from every external artifact.

**Exit:** a project can run end to end, restart without repeating valid work,
and bypass any preparation stage. All external workflows still pass.

### Phase 6 - Cutover and retirement

1. Reproduce installation in a clean environment, including the Basic Pitch
   ONNX-only dependency path, optional backends, offline/missing-weight errors,
   and explicit checkpoint downloads. Record supported versions.
2. Run the full legacy and v2 suites, reference comparisons, recovery tests,
   and user acceptance sessions. Verify exported files in MuseScore and Ableton.
3. Make v2 the default only after the replacement gates pass. Retain a named
   legacy release/revision and a documented launch/rollback path.
4. Remove legacy orchestration and temporary adapters in a later cleanup after
   acceptance; retain reusable algorithm implementations and relevant tests.

**Exit:** v2 is the supported default with reproducible installation and a
documented recovery path. No deletion of local audio, MIDI, profiles, or projects.

## Replacement gates

These are proposed acceptance targets, not claims about current performance.
Confirm them against Phase 0 measurements before committing to a release date.

| Area | Required evidence |
|---|---|
| Existing behavior | All 174 legacy tests remain green during coexistence; applicable public behavior assertions also pass against v2. |
| User state | No loss of authored notes, rests, releases, or rejections through inference, cancellation, track switching, and save/reopen. |
| Timing | Straight/triplet and 4/4, 6/8, 12/8 cases pass; beat/second round trips are within one sample; MIDI round trips within one tick at documented PPQ. |
| Responsiveness | On the reference machine, cached local edit-to-display p95 <= 100 ms and warm lightweight retrain-to-visible-update p95 <= 500 ms; neural inference is measured separately with progressive results. |
| Quality | Same adapters and inputs introduce no unexplained onset/pitch F1 regression; approve intentional bass slur behavior separately by listening and note inspection. |
| Concurrency | Edits/settings/track changes reject obsolete results; cancellation cannot publish partial artifacts as complete. |
| Export | Tempo, meter, triplets, velocities, held polyphony, same-pitch re-strikes, GM drums, and no pitch bends verified in parsed MIDI and target applications. |
| Recovery | Missing media/models, interrupted saves/stages, corrupt metadata, and upstream reruns produce explicit errors without destroying prior valid work. |
| External entry | Warped-mix, stems-only, incomplete stem sets, imported chords, and per-stem MIDI work without forced analysis, warping, or separation. |

The known piano-HR batch-boundary limitation is a documented baseline limitation,
not permission for new truncation. Test it explicitly; either repair continuity
before relying on that backend for sustained-note work or expose the limitation
and retain an approved alternative.

## Decisions still requiring user input

These do not block writing the architecture or starting domain tests:

- **Distribution/licensing:** ADTOF weights are reported as non-commercial in the
  handoff; piano-HR checkpoint terms also need verification. Before bundling or
  enabling downloads for a release, record the actual code/weight/data licenses
  and intended use. Commercial distribution needs cleared alternatives or
  appropriately licensed training; retraining alone does not guarantee clearance.
- **Learning scope:** propose per-song/per-track learning as the initial isolation
  rule, with legacy profiles preserved and an explicit compatible-profile import.
  Confirm this before Phase 3; automatic cross-song learning is not assumed.
- **Lead-in/pickup policy:** confirm before Phase 4 rendering UI. Store enough
  mapping/provenance to avoid irreversible loss regardless of the chosen policy.

## Immediate work order

1. Approve the same-repository replacement boundary and external-stem MVP.
2. Complete the Phase 0 behavior/performance checklist and reference fixtures.
3. Implement and test timing, note commands, and durable project state.
4. Demonstrate the single-bass vertical slice before expanding models or pipeline.

Do not estimate a full rewrite deadline from line counts. Re-estimate after the
bass slice demonstrates the real costs of parity, persistence, and UI integration.
