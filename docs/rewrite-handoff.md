# Handoff: refactor or rewrite of audio_to_midi

This document is for the agent that picks up the next phase. It summarises what
the project has learned, what is worth keeping, and the decisions to make
before restructuring the app around the new workflow.

Follow-up: [decision and execution plan](execution-plan.md) recommends a
side-by-side replacement core in this repository, keeping the current app
usable until the external-stem workflow passes its replacement gates.

State at handoff: `main` at commit `c508071`. 174 unit tests pass.
Reverified at `56a1d48`: explicit test discovery works without
`tests/__init__.py`. Use UTF-8 output to avoid Windows console encoding errors:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p 'test_*.py'
```

## 1. Target workflow

```
1. original song (mix)
2.   ─► key, time signature, tempo map (beats + downbeats)
3.   ─► WARP the full mix to an exact fixed tempo, bar 1 beat 1 at t = 0
4.   ─► block-chord MIDI track from the warped mix (one chord per beat/bar span)
5.   ─► 6-stem separation of the WARPED mix
6.        ├─ drums   ─► drum MIDI   (user-editable)
          ├─ bass    ─► MIDI        (user-editable)
          ├─ vocals  ─► MIDI        (user-editable)
          ├─ piano   ─► MIDI        (user-editable)
          ├─ guitar  ─► MIDI        (user-editable)
          └─ other   ─► no MIDI (used only as harmony context for key/chords)
```

### Decisions (confirmed by the user)

- **Warp before separation.** Warping is more accurate on the full track, and
  every stem then inherits exactly the same timing. Never warp stems
  individually.
- **The workflow can be picked up at any stage**, including with **external
  warping** (for example, Ableton) and **external stem separation** (for example,
  UVR or EchoGrid). Each stage must therefore read and write plain artefacts on
  disk (see §5, "Stage artefacts and entry points") rather than in-memory-only state.
- **No pitch bends in MIDI.** Every note is one fixed MIDI pitch. Show only the
  primary pitch:
  - A **slur** (a held pitch change between two clearly sustained pitches) becomes
    **two notes**, split at the pitch change.
  - **Bending into a note** becomes just the target note, starting at the attack.
  - A **fall** (or any glide off the end of a note) is absorbed into the note it
    ends; the note keeps its primary pitch.
  - Rule of thumb: short, transitional pitch segments are absorbed into the
    neighbouring stable pitch. Only a stable (held) new pitch starts a new note.

Hard rule learned in this project: **keep onset detection separate from pitch
detection.** The algorithms are good at one or the other, not both (see §3).

Today the first steps happen outside the app. Ableton warps the song to a
fixed tempo; UVR or EchoGrid produce stems. The user then types BPM and beats per
bar, and the app assumes the file starts on a downbeat at t = 0 (`TimeGrid` in
`core/audio_engine.py` has no offset or tempo map). The new workflow adds tempo,
downbeat, warping and separation to the app, while keeping those external tools as
supported entry points.

## 2. What the current app does (and what to keep)

| Area | Where | Status | Keep? |
|---|---|---|---|
| Mono pitch: bass CREPE full/tiny + onset + key; vocals pYIN / CREPE full + key | `core/mono_pitch.py` (`MonoTranscriber`, `track_pitch`, `clean_key_contour`) | User A/B-approved | **Keep** (models and key-cleaning) |
| Poly pitch: piano/guitar Basic Pitch and high-resolution piano model (`piano_hr`) | `core/poly_pitch.py` (`PolyTranscriber`, `transcribe_notes`) | User A/B-approved | **Keep** |
| Drums: ADTOF CRNN, 5 classes, per-class adaptive thresholds and learned relabels | `core/adtof_drums.py` | Best drum method tried | **Keep** |
| Trained onset model (logistic on grid candidates, learns from user edits) | `core/models.py` `CorrectionModel`; features in `AudioMarkers` | Works; owns note boundaries | Keep the idea; rewrite the plumbing |
| NNLS polyphonic pitch | `core/polyphonic_nnls.py`, `core/pitch_adaptation_models.py` | Poor on piano/guitar and bass octaves | Drop, or keep only as the "Standard" fallback |
| Chord detection (chroma templates per beat, collapse) | `core/chord_analysis.py` | Works; exports block-chord MIDI | Keep as a baseline, evaluate upgrades |
| Key detection (Krumhansl–Kessler on chroma) | `core/analysis.py` `detect_key_from_chroma` | Works on the full mix or all stems; wrong on bass alone (said D minor for a Bb major song) | Keep; run on the mix / all pitched stems |
| MIDI import/export | `core/midi_export.py` | Single fixed tempo | Keep; add time signature |
| Note metrics (onset/pitch F1, bipartite matching) | `core/evaluation.py` | Useful for regression tests | Keep |
| Playback (pygame mixer + MIDI) | `core/playback.py` | Works | Keep |
| Worker/learning coordination | `core/learning_runtime.py`, `core/inference_cache.py`, `ui/learning_coordinator.py`, `workers/analysis_worker.py` | Works; visible-window batches with cancel | Keep the pattern |
| Mode registry (`Standard`, `Percussive`, the pitched presets) | `core/mode_registry.py`, `core/pitch_models.py` | Works | Keep the idea; per-stem defaults instead of a global mode |

### Why the structure needs work

- `AudioMarkers` in `core/audio_engine.py` is a god object: 2,783 lines,
  112 methods and about 57 private state fields. It mixes audio loading, CQT/STFT
  features, onset candidates and training, pitch caches, NNLS, mono/poly model
  adapters, note-span assembly, drum features, ADTOF feedback, user edit
  bookkeeping and profile I/O.
- `ui/main_window.py` (1,161 lines) and `ui/cqt_view.py` (1,081 lines) also hold
  workflow logic (export, chord analysis orchestration, stem selection).
- Notes are not a first-class model. They are *derived* each time from
  onset candidates plus per-block pitch rows plus user labels
  (`_assemble_note_spans`), with rows passed around as positional tuples
  `(t0, t1, midi, source, energy[, block_key[, continues]])`.
- Time is in seconds everywhere; the grid is implied by BPM and a
  `"1/16"`-style duration string. There is no tempo map, no downbeat offset and
  no triplet grid, yet the test song has a triplet feel. On "Firm Foundation",
  drum onsets lock to a 16th-triplet grid at 0.72 versus much lower for straight 16ths.
- One stem is loaded at a time, with one global mode. The target workflow is a project
  with N stems, each with its own model and edits.

## 3. What we learned

### Onsets vs pitch

- Pitch is close to solved on clean stems. pYIN, CREPE and even NNLS agree on
  75–86% of bass notes. **Segmentation is the hard part.** Splitting notes where
  f0 changes turns slides into bursts of off-grid fragments.
- Pretrained trackers' own onsets are unreliable: CREPE full produced nothing usable,
  and CREPE tiny produced far too many onsets. The design that works:
  1. The app's trained, **grid-quantised onset model owns note boundaries.**
  2. A pretrained model's onsets (or whole-stem-normalised SuperFlux attacks)
     are *evidence* fed to that onset model, not boundaries.
  3. The pitch model only answers "which pitch(es) sound in this onset-to-onset
     block?"
- Editing rules the user specified (implemented today; keep them):
  - Deleting an onset deletes the note that starts there and extends the previous note to
    the next onset, but not past the edge of the visible window.
  - A release sits on the next onset unless the user adjusts it.
  - Use the same quantisation even when more precise onsets are fed to the models.
  - User-added onsets, pitches and rests are always kept. Rejected pitches stay rejected.
  - Polyphonic held notes carry across new onsets in other voices. A re-struck
    same pitch splits.
- Vocals are rhythmically loose. Onset grid lock is about 0.1 for every method, including raw
  detectors. They need quantisation and user correction, not a better detector.
  Vocals also split on held pitch changes (legato).

### Pitch models (user A/B judgement on "Firm Foundation", htdemucs_6s stems)

| Stem | Use | Rejected |
|---|---|---|
| Bass | CREPE full or CREPE tiny, both with onset and key cleaning | PESTO (voicing about 0 on bass), NNLS (octave errors; BabySlakh bass note F1 about 0), Basic Pitch (misses slides) |
| Vocals | pYIN + key, CREPE full + key | NNLS (weakest pitch support) |
| Piano | Basic Pitch, high-resolution piano model | NNLS ("terrible") |
| Guitar | Basic Pitch, high-resolution piano model | NNLS |
| Other | None. No two methods agree (F1 ≤ 0.33). Use as harmony context only | — |

Key-aware cleaning (`clean_key_contour`): a non-diatonic run held under 0.25 s that is
not in the local harmony takes the neighbouring in-key pitch (it is a slide or bend).
It only relabels pitches, never boundaries. Folding created spurious legato splits when it also
moved boundaries. Caveat: real chromatic passing tones under 0.25 s get folded. This
matches the user's "primary pitch only, no pitch bends" decision (§1). Today only
vocals split on held pitch changes (`mono_legato_split`). The rewrite should apply
the slur rule (a stable new pitch ⇒ a new note; transitional bend-ins and falls are absorbed)
to every mono stem, including bass, and for poly stems the model's notes already behave this way.

### Drums (ADT, 5 classes → GM 36/38/42/47/49)

| Method | Attack recall / precision | Grid lock | CPU speed |
|---|---|---|---|
| **ADTOF** (kept) | 87 / 95 % | 0.75 | 31× realtime |
| YourMT3+ (MT3 family) | 96 / 51 % | 0.53 | 0.8× realtime |
| NMFD (no training) | 81 / 76 % | 0.53 | 1× realtime |

Kick is solved. ADTOF confuses snare with tom (about half of the snares) and misses
quiet hi-hats. Hence the per-class threshold adaptation and learned relabels from
user edits. **ADTOF weights are CC BY-NC-SA (non-commercial).** Retraining is the way
to fix both the licence and the snare/tom confusion.

### Speed on CPU (this machine)

| Model | Speed | Notes |
|---|---|---|
| CREPE full | about 0.16–0.4× realtime | Notes fill in progressively; cache results |
| CREPE tiny | about 8× realtime | |
| pYIN | about 2× realtime | |
| Basic Pitch (ONNX) | about 20× realtime | |
| High-resolution piano model | about 1× realtime | 10 s internal segments; analysed in 9 s batches |
| ADTOF | about 31× realtime | Whole stem once |

Visible-window analysis in cancellable, cached background batches kept the UI
responsive. With the target workflow, consider a whole-stem background pass after
separation, prioritised by what is on screen.

Known limitation: the high-resolution piano model only reports notes it sees start, so a
note held across a batch boundary ends at the boundary. Basic Pitch does not have this
problem.

### Older findings (onset model features, benchmarks)

- `transient` is by far the strongest onset feature for every instrument.
  `chroma_delta` and `energy_db` help moderately. `percussive_spike` was broken
  (raw sums, loudness-dependent) and was normalised (upper 30% of CQT bins,
  centre-to-wings ratio, ^0.7).
- With known onsets, ten user-confirmed notes did not make NNLS a reliable pitch
  adapter. Training only moved a keep gate; it could not fix the chosen pitch.
  This is why pitch now comes from pretrained models and user rejections.
- Datasets used: BabySlakh (CC BY 4.0) and a MAESTRO subset (CC BY-NC-SA). They are
  not in the repo; download scripts are in the archive commit.

## 4. Environment and dependency notes

- App venv: Python 3.12, torch (CPU), torchcrepe, PySide6, pygame.
- **Basic Pitch** pins an old TensorFlow with no 3.12 build. Install it with
  `pip install --no-deps basic-pitch==0.4.0`; it runs the bundled ONNX model via
  `onnxruntime`. It needs `mir_eval resampy scikit-learn` (in `requirements.txt`).
- **High-resolution piano model**: `piano-transcription-inference`. It downloads its checkpoint
  (about 165 MB) to `%USERPROFILE%\piano_transcription_inference_data\` on first use.
  Check the checkpoint licence before shipping.
- **ADTOF**: `adtof-pytorch` from git (pinned in `requirements.txt`).
- **6-stem separation** was done with Demucs `htdemucs_6s` in the experiment
  environment (Python 3.11). The script is `separate_6s.py` in the archive commit. It is not
  yet an app dependency.
- Experiment harnesses are removed from the tree but preserved in commit
  **`d5ff7bb`**:
  - `experimental-pitch/`: the pitch and drum bake-off, proxy metrics, A/B report,
    `run_loudest.py`, `separate_6s.py`, MT3 worker.
  - `experimental/`: onset-feature analyses, BabySlakh/MAESTRO benchmarks and
    dataset download scripts.

  Restore one with `git checkout d5ff7bb -- experimental-pitch`.

## 5. New capabilities the target workflow needs

These are not in the app today. Candidates are starting points to evaluate, not
decisions.

1. **Beat, downbeat and tempo tracking on the mix.** Candidates: CPJKU *Beat This!*
   (ISMIR 2024), madmom's downbeat RNN (check Python 3.12 support), librosa
   (beats only, weak on downbeats). Output a tempo map plus downbeats.
2. **Time signature.** Infer from the downbeat spacing in beats; let the user confirm.
   Remember triplet/compound feels (12/8 versus 4/4 with triplets).
3. **Fixed-tempo rendering of the full mix** (decided: before separation). Time-stretch
   between beat anchors to a constant BPM and trim or pad so bar 1 beat 1 is at
   t = 0. For example, use Rubber Band (pyrubberband) piecewise, or a phase vocoder.
   Let the user correct beat and downbeat anchors before rendering. Save the warp map,
   so MIDI can also be mapped back to original time if ever needed.
4. **Key detection** on the full mix (or all pitched stems), with a user override.
   The existing Krumhansl–Kessler code is fine as a start.
5. **Block-chord MIDI.** `chord_analysis.analyze_chords` and `collapse_chords` already
   work per beat; feed them the warped mix plus the Other stem as harmony context.
   Possibly evaluate a learned chord recogniser later.
6. **Stem separation in-app** (htdemucs_6s) on the warped mix, run once per song and
   cached.
7. **Per-stem tracks in one project**: each with its own onset model, pitch model,
   edits, rejections and export. Other gets no MIDI.

### Stage artefacts and entry points

A project is a folder. Each stage reads the previous stage's artefact and writes
its own, so any stage can be skipped by supplying that artefact from outside.

| Stage | Artefact (suggested) | External alternative |
|---|---|---|
| Original mix | `original.wav` | Any audio file |
| Analysis | `song.json`: key, time signature, BPM, tempo map / beat and downbeat anchors | User enters key, meter and BPM by hand |
| Warp | `warped.wav` (fixed BPM, bar 1 beat 1 at t = 0) + `warp_map.json` | Already-warped mix from Ableton: import as `warped.wav` with user BPM and meter; no warp map |
| Chords | `chords.mid` (block chords) + editable chord list | Import an existing chord MIDI |
| Stems | `stems/{drums,bass,vocals,piano,guitar,other}.wav` | Stems from UVR, EchoGrid, etc. Map files to roles by name or let the user assign them; missing roles are allowed (for example, a 4-stem split has no piano or guitar). |
| Per-stem MIDI | `midi/<stem>.mid` + per-stem edit state | Import an existing MIDI for a stem as a starting point (the app already loads companion MIDI) |

Rules:
- When starting from externally warped audio or stems, the app must not re-warp them.
  It only needs BPM, meter and the downbeat at t = 0 (offer a small offset nudge if
  the first downbeat is not exactly at 0).
- External stems must be validated against the warped mix (same length and sample
  rate, or resample/pad, with a warning) so that all tracks share the grid.
- Re-running an earlier stage must warn before invalidating later artefacts that
  have user edits.

## 6. Refactor or rewrite: options

### Option A: incremental refactor (strangler pattern)

Keep the app running. Extract pieces out of `AudioMarkers` one at a time behind
tests:

1. A `TempoMap` / `Grid` value object in beats (with downbeat offset and
   triplet subdivisions) that replaces `TimeGrid` and `duration_str`.
2. A first-class `NoteStore` per stem: note = (onset_id, pitch, release,
   velocity, source, user_flags). Onset deletion merges notes inside the store,
   not by re-deriving spans from tuples.
3. `OnsetDetector` and `PitchModel` interfaces. Wrap `MonoTranscriber`,
   `PolyTranscriber` and `AdtofDrumTranscriber` behind them.
4. A `SongProject` that owns the mix, key, meter, tempo map, chords and stems.
5. Then add the new front-end pipeline (beats, warp, separation).

Pros: always shippable; reuses tested behaviour. Cons: slow, because every step fights the
seconds-based, single-stem assumptions threaded through `AudioMarkers`, the UI
and the profile format.

### Option B: rewrite the core, port the proven parts (recommended)

Build a new core package around the target workflow and a beat-based time
model. Port the model wrappers and the editing rules as-is. Rebuild the UI on top.

- **Port nearly unchanged:** `mono_pitch.py`, `poly_pitch.py`, `adtof_drums.py`,
  `evaluation.py`, `midi_export.py` (add time signature), `chord_analysis.py`,
  `analysis.detect_key_from_chroma`, `playback.py`, the onset feature set and
  logistic onset model (`models.py`, `logistic_utils.py`), and the
  worker/cancel/cache pattern.
- **Rewrite:** `AudioMarkers`, note-span assembly, the profile format, the main window
  orchestration, and the mode/global-stem model (to become per-stem tracks).
- **Port the tests as specifications.** `tests/test_mono_pitch.py`,
  `tests/test_poly_pitch.py`, `tests/test_adtof_drums.py` and
  `tests/test_note_spans.py` encode the onset/pitch/edit rules above, so rewrite
  them against the new API rather than deleting them.

Why B: the new workflow changes the core invariants. Time becomes a fixed-tempo beat
grid from bar 1, the unit of work becomes a multi-stem project, and notes need to be
first-class editable objects. The current code's main cost is exactly those
assumptions. The valuable, hard-won parts are small, already isolated modules.

### Suggested module layout for a rewrite

```
core/
  project.py        SongProject: mix, stems, key, meter, tempo map, chords, tracks
  timebase.py       TempoMap / Grid in beats (downbeat offset, straight + triplet)
  pipeline/
    beats.py        beat/downbeat/tempo + time signature
    warp.py         fixed-tempo rendering of the full mix (before separation) + warp map
    separate.py     htdemucs_6s on the warped mix, cached; or import external stems
    key.py, chords.py
  onsets/           candidate features, trained grid onset model, evidence fusion
  pitch/            mono (CREPE/pYIN + key cleaning), poly (Basic Pitch, piano_hr)
  drums/            ADTOF + threshold/relabel adaptation
  notes.py          NoteStore + edit commands (delete onset ⇒ merge, etc.), undo
  midi.py           export per stem + chords, with tempo and time signature
ui/                 project view, per-stem editor (onset lane + piano roll), chord lane
workers/            background jobs with cancel + visible-window priority
```

## 7. Open questions for the user

Answered: warp before separation; any stage can be entered, including external
warping and separation; no pitch bends (see §1).

Still open:

1. Rewrite (B) or refactor (A)? This document recommends B.
2. Is non-commercial licensing acceptable for now (ADTOF; check the piano_hr checkpoint),
   or is retraining in scope?
3. Should edits keep training the onset model per song only, or across songs
   (profiles)?
