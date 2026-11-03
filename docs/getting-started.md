# Getting started with Audio to MIDI

Audio to MIDI turns a song into editable MIDI. You work through stages, each
shown in the strip at the top right of the window:

**Mix → Count bars → Stems → Notes → Chords → Export**

1. Import the full mix.
2. Find its bars and beats (detect, then fix with pins or tapped downbeats).
3. Separate or import stems.
4. Transcribe and correct notes per stem.
5. Detect and correct a lead-sheet chord track.
6. Export MIDI and/or a MusicXML lead sheet.

The audio is never time-stretched. Your pinned downbeats define a tempo map
that places every moment of the original recording at an exact bar, beat and
sub-beat, even when the band speeds up or slows down.

Menu items that are not available yet are grayed out; hover over them to see
what is missing.

## 1. Install and launch

Open PowerShell in the repository folder. For a new isolated installation,
use Python 3.12 and run:

```powershell
.\install.ps1
.\.env\Scripts\python.exe -X utf8 -m audio_to_midi
```

The base installation supports the editor, MIDI import/export, the `pyin`
vocal model, and mix preparation. Everything else installs itself the first
time you use it: when you pick a model (CREPE, Basic Pitch, Piano-HR, ADTOF, YourMT3+) or
run **Separate stems**, the app asks once, then pip-installs the packages into
the Python it is running on and downloads the weights to their usual cache. You
never choose a Python or a model folder. This needs an internet connection the
first time; the status bar shows progress, and **Cancel** stops it. ADTOF needs
`git` on PATH. ADTOF drum weights are non-commercial; review the piano-HR
checkpoint terms before relying on it.

YourMT3+ uses a separate, automatically created environment at
`~/.audio_to_midi/yourmt3` with the validated `mt3-infer==0.2.0` adapter and
`transformers==4.45.1` runtime.
Its MoE multi-track no-pitch-shift checkpoint (the bake-off's YourMT3+ variant)
is downloaded once to `~/.cache/mt3-infer/yourmt3`. GPU inference is used when
available, with supported CPU fallback. `ATM_DEVICE=auto|cpu|cuda` controls
device selection; explicitly requesting unavailable CUDA reports an error.
App inference/separation workers coordinate exclusive GPU ownership. Faster
models fall back to CPU while the YourMT3+ worker uses GPU, rather than queueing
behind it. External programs/benchmark processes do not participate in this lock.
Advanced deployments can set `ATM_YOURMT3_PYTHON` and
`ATM_YOURMT3_CHECKPOINT`; checkpoint bytes remain part of cache identity.

The optional `madmom_cnn` and `madmom_brnn` vocal models also install on first
use. Their onset weights are **CC BY-NC-SA 4.0 (non-commercial)**; the app asks
for a separate acknowledgement even if CREPE is already installed. A fresh
madmom build needs `git`, Cython/setuptools (installed by the app), and a C++
compiler; on Windows use Microsoft C++ Build Tools.

To install all the transcription models ahead of time instead (for example
before going offline):

```powershell
.\install.ps1 -Models
```

If your existing environment is already installed, keep using its Python.
For the repository's default environment, either launch command opens the same
project editor:

```powershell
.\.env\Scripts\python.exe -X utf8 -m audio_to_midi
.\.env\Scripts\python.exe -X utf8 app.py
```

The former single-stem application is retired and its unused engine, window,
workers, and chord-analysis pipeline have been removed. Chord symbols, voicings,
and template recognition now live in `audio_to_midi.chords`; the shared
key-to-scale helper remains in `core.chord_analysis`.
This does not delete or migrate any projects,
audio, MIDI, profiles, logs, or model caches. The current project format is
**schema 4**; schema 3 projects still open with unknown note confidence.
Older schemas and unsupported model aliases are rejected. Preferences keep their current settings location. Compatible
onset examples can still be imported from older profiles; legacy session state
is not converted automatically.

Malformed candidate-profile rows, unknown model features, invalid key-analysis
inputs, and invalid tempo curves now raise errors instead of being skipped or
replaced by defaults. Spectrogram-reference failures appear in the status bar
and are not cached as successful measurements. Valid legacy onset-feature
formats remain supported.

The unversioned installer uses `.env` by default (matching `activate.ps1`).
Use `.\install.ps1 -Environment .venv-v2` to update an existing preview
environment, or `-Environment .venv` for another environment. The old
`install-v2.ps1` command is a deprecated forwarding wrapper that keeps its
`.venv-v2` default; the versioned requirements files are compatibility aliases.

For historical access to the retired app, revision
`0a571a3fdcb703465c89b0b266f01daaa12ab07f` is the last pre-cutover snapshot.
Open it in a separate checkout and use that revision's own dependencies and
`python app.py`; do not mix its modules with the current editor.

## 2. Create a project and import the mix

1. **File > New project...** asks for a name, then a parent folder. The project
   is saved in a new folder named after the project; the copied mix and
   separated stems are stored there as well.
2. **File > Import mix...** and choose the full song (WAV or FLAC works best).
   Every project needs a mix; it is the timing reference for everything else.
3. The app immediately runs **Edit > Detect tempo, meter, and key** on the mix.

The imported mix is copied into the project's `audio` folder, so the project
keeps working if you move or delete the original; the whole project folder can
also be moved or renamed. Projects made by earlier versions that still point at
an outside file get a copy the next time you open them (as long as the original
is still there). Save regularly with **File > Save** (Ctrl+S). Save writes directly
to the current project folder without a dialog; it asks for a folder only if
one has not been chosen yet. Closing the window, opening another project, or
creating a new one warns only when there are changes since the last successful
save (including analysis results). A failed or canceled save leaves those
changes unsaved.

## 3. Count the bars (Mix view)

Choose **Mix** in the **Track** selector. The upper lane shows the mix
amplitude with a line for every detected beat; the area below is a constant-Q
spectrogram of the mix. The line under the menus summarizes the timing, for
example `120 BPM · 4/4 · Bar 1: 0.350 s · Key: D major`; click **Edit...** to
type values yourself.

Beat colors:

| Color | Meaning |
|---|---|
| Gray | Detected beat, not yet counted into bars |
| Blue | Counted beat |
| Orange | Downbeat (beat 1 of a bar) |
| Yellow | A downbeat you pinned |

Detection usually guesses the tempo but cannot know where bar 1 is, and it
often locks onto eighth notes or half notes. Fix that with **pins**: tell the
app which beats are downbeats, and it counts every other beat from them.

- **Click** a beat to select it. **D** pins it as a downbeat of the bar the app
  suggests, **U** unpins it, **Delete** removes a false beat.
- **Right-click** a beat for *Downbeat of bar N*, *Downbeat of bar...* (type
  the number), *Unpin* and *Remove this beat*. Right-click between beats to
  *Add a beat here*, or to *Clear all pins*.
- **Edit > Clear all pins** also resets downbeat pins without removing beat
  candidates. It is available when pins exist.
- Pin two downbeats several bars apart; the tempo comes from your pins, not
  from the detector's guess. Add more pins where the count drifts.
- The pins and counted beats form the song's tempo map. Bar lines and the
  "Bar N" labels follow it, so a downbeat pinned to bar N sits exactly on the
  Bar N line. If the music speeds up or slows down, the bars get narrower or
  wider instead of drifting away from the pins. A bar that looks squeezed or
  stretched usually means an extra or missed tap, or a pin set to the wrong
  bar. A steady song gets one constant tempo; a drifting one shows "avg" in
  the timing summary. Typing a tempo or bar 1 in **Edit...** replaces the tempo
  map with one constant tempo.
- Exported MIDI carries the tempo map as tempo changes, so it lines up with
  the original recording in a DAW.

You can also count bars while listening to **any stem** (drums, bass, vocals,
or another part). Choose that track and **Info (I)** to see the same colored
beat markers over its audio and notes. Right-click in the waveform,
spectrogram, or note lanes for the timing menu. A pin uses **exactly where you
right-clicked**, not the nearest detected beat or the note's start; click the
attack you hear or see. If there is no beat there, choosing a downbeat adds
and pins one. Pins are shared by all tracks and the Mix view, not per-stem.
Left-click a beat in the upper lane to select it for **D** / **U**.
Near a beat marker, *Unpin nearby beat* and *Remove nearby beat* let you
manage that marker; downbeat pinning still uses the exact click position.

**Tap downbeats instead.** Check **Edit > Tap downbeats**. The mix plays from
the current window, and you press **Space** on the "one" of every bar. Your
taps appear as markers on the initial keypress; holding Space does not add
repeated taps. Uncheck the item when done (tap at least four bars in
a clear section). The app asks which bar your first tap was, snaps each tap to
the nearest detected beat, pins them, and infers the meter from the beats
between taps. The status line reports how many bars were counted and how late
you tapped on average.

**If the tap arrows consistently land late or early**, use **Edit > Calibrate
tapping...**. It plays a 100 BPM, 4/4 **TICK tock tock tock** pattern. Listen to
one count-in bar, then press **Space on the accented TICK** for eight bars.
Save the measured correction when the run finishes (or use **Finish** after at
least six steady taps). Inconsistent runs ask you to retry.

The correction measures audio-output, keyboard, and your own tapping delay
together; it is not a sound-card-only measurement. It is saved on this machine,
not in the song, and shifts **new** taps and their arrows before snapping.
Existing pins, audio, notes, and the playback cursor are unchanged. Recalibrate
after changing speakers/headphones or audio settings. **Use automatic correction**
in the calibration dialog removes the saved correction and restores the previous
song-based lag estimate. Calibrated taps are not automatically lag-corrected a
second time. This is separate from external MIDI latency in Playback settings.
To check your raw timing without either calibration or the automatic estimate,
choose **Use taps as pressed (no correction)** in that dialog. Automatic correction
runs only after tapping ends; it does not delay live arrow display.

When you are not tapping, **Space** plays and stops. **View > Metronome click**
adds clicks (high on downbeats) so you can hear whether the count lines up.

The stage strip shows **Count bars ✓** once the bars are counted.

## 4. Get stems

Either:

- **Edit > Separate stems...** runs Demucs `htdemucs_6s` on the original mix in
  the background (a few minutes for a full song on a CPU). The first time, the
  app installs Demucs and downloads its model (about 80 MB) automatically.
  Existing tracks of the same role keep their edits and get the new audio. Or:
- **File > Import stems...** and choose stems that were separated from the
  imported mix (same length and sample rate). The role is guessed from the
  file name, and you can change it.

The roles are `bass`, `vocals`, `piano`, `guitar`, `drums`, and `other`; one
track per role. `other` is harmony context only and produces no notes.

## 5. Transcribe and correct notes

Choose a stem in **Track** and its **Model** next to it:

| Role | Models, in ranked order |
|---|---|
| Bass | Piano HR; madmom offline BRNN + CREPE full; madmom CNN + CREPE full |
| Guitar | Basic Pitch; YourMT3+; madmom CNN + CREPE full |
| Piano | Piano HR; YourMT3+; Basic Pitch |
| Drums | ADTOF; YourMT3+ |
| Vocals (unchanged, not bake-off ranked) | `pyin`, `crepe_full`, `crepe_tiny`, `madmom_cnn`, `madmom_brnn` |

MR-MT3 is not offered. Basic Pitch replaces the effectively tied MR-MT3 piano
choice; drums have only two ranked choices. Bass Piano HR is filtered to MIDI
28-67. The instrumental madmom combinations use range-limited CREPE-full pitch
(bass 32.71-420 Hz; guitar 75-1400 Hz) and either CNN onsets or the offline
bidirectional RNN detector. They do **not** use the vocal syllable pipeline.
They retain the existing onset-grid editing, feedback and duration policies.

### YourMT3+ background results and provenance

YourMT3+ inference runs in its own worker/subprocess, independent of the
ordinary analysis lane. Other model results, editing and navigation do not
wait for it. A separate status field reports queueing, model loading, segment
progress, decoding, cancellation and failures. Duplicate requests for the same
track/input are not launched twice. Changing its model or source cancels that
track's job; changing/opening projects or importing a new mix cancels all jobs.
Publication tokens reject results after audio, grid, model, source or edits change.
Esc/Cancel also terminates the inference subprocess, including during loading.

Choose **Stem-derived** or **Full-mix-derived** beside the YourMT3+ selector.
Both outputs are cached separately; no source is silently substituted.
Full-mix evidence retains all decoded instruments; piano/guitar track proposals
select GM program families 0-7/24-31, and drums select drum flags. Stem-derived
pitched proposals keep all non-drum predictions, including instrument
misclassifications. The full-mix option is especially useful for piano:
reviewed mix F1 was 0.810 versus 0.722 on the piano stem. These are historical
scores, not retuned defaults. Visible-bar analysis publishes only its requested
interval, even though the reusable inference covers the whole input.
CLI: `transcribe <project> piano --source mix` (select `yourmt3` on that track first).

### Disposable evidence cache and threshold replay

Inference evidence lives under the app's **`cache/inference/v1`** folder
(`ATM_CACHE_DIR` overrides the cache root), not in project manifests or benchmark
evidence directories. Reopening, switching models and changing decoding
thresholds reuse it. Deleting the cache folder is safe: missing evidence is
regenerated when needed without cache-missing warnings or dialogs.
Actual invalid/corrupt entries report an explicit error naming the entry to
delete; they are not silently discarded or overwritten.

Keys include content hashes, mix/stem role provenance, model/package and
checkpoint identity, inference parameters and evidence schema version.
Writes are atomic, arrays are validated/checksummed, and per-key OS locks avoid
duplicate inference. The two compatible madmom combinations share CREPE-full
evidence. Preserved evidence includes Basic Pitch onset/frame/contour matrices;
Piano HR continuous onset/offset/frame/velocity and pedal planes; untouched
CREPE pitch-bin activations, unmasked F0, periodicity and smoothed periodicity;
pYIN unmasked F0, voiced flags/probabilities; madmom onset curves; ADTOF class
activations/novelty; and YourMT3+ decoded note times, pitches, programs,
velocities and drum flags. Neural vocal windows retain their existing bounds.
Threshold-dependent grid notes and feedback are replayed separately from
these inference archives, not treated as reusable neural evidence.

The **AI notes: None -- All** slider filters existing predictions using their
model-specific evidence scores; it does not change decoder thresholds or rerun
inference. The existing vocal parameter controls still work. The backend
`decoder_parameters` API supports native onset/frame thresholds for Basic
Pitch/Piano HR, onset/voicing thresholds for instrumental madmom, novelty
thresholds for ADTOF, and voicing thresholds for legacy CREPE/pYIN. These are
not a shared calibrated confidence scale. Zero does not create every possible
note: local maxima, frequency ranges, voicing/energy, duration, grid and
structural decoding policies still apply.

**YourMT3+ does not support faithful confidence replay.** Inspection of the
validated adapter's `forward`/`decode`, vendored `inference_file`, and
`t5mod_helper` generation confirms that token logits are reduced to argmax
IDs and not returned/retained; teacher-forced loss is absent without reference
tokens. Decode returns MIDI notes/programs/drum flags/velocities, not note
probabilities. Retaining autoregressive token scores and defining a defensible
token-to-note confidence would require a materially different interface and
semantics. Velocity is never used as confidence. Threshold requests for this
backend are explicitly rejected. The source selector tooltip explains this
limitation. YourMT3+ notes have unknown confidence: they appear at **All** and
are hidden at lower slider positions, without treating velocity as confidence.

Validation covers the app's model choices/project schema, worker cancellation
and stale publication, disk-cache reuse/invalidation/corruption, and native
threshold replay. Opt-in real-checkpoint smoke tests exercise all 19 exposed
role/model/source combinations, reopen the project, and forbid neural inference
while replaying cached evidence (including thresholds 0, 0.8 and 1).
Run these opt-in tests with both GPU/auto and CPU settings; benchmark scores/references are not
changed. To repeat the real-model check in a provisioned environment:

```powershell
$env:ATM_RUN_MODEL_TESTS = "1"
$env:ATM_DEVICE = "auto" # repeat with "cpu"
python -m unittest tests.test_models_live
```

### Choosing how many AI notes to keep

Each instrument track has an **AI notes: None -- All** slider. At the far left,
all AI-generated notes are hidden so you can start from scratch; move slightly
right to keep only notes with the strongest detection evidence. At the far
right, all notes returned by the detector are included, even low-confidence
ones. Manual edits and imported MIDI remain visible at every position.

The filter applies to the whole track, playback, MIDI, and MusicXML export.
It does not delete predictions, train the model, or rerun detection. Moving it
back restores filtered notes, except notes you explicitly deleted. The position
is saved per track and supports Undo/Redo.

Evidence scores are model-specific, not calibrated probabilities: Basic Pitch
uses its note activation, Piano-HR uses pitch-frame activation (not key velocity),
drums use hit strength, and monophonic models use pitch-frame agreement.
Older saved detections have unknown scores and appear only at **All**; analyze
again to obtain scores. **All** includes the detector's returned candidates,
not notes excluded by its internal decoder thresholds or pitch range.
New saves use project schema 4; schema 3 projects still open, but older app
versions cannot read new saves.

### Running detection

With **Model > Reanalyze after edits** checked (the default), the visible bars
are analyzed automatically. **Model > Analyze visible bars** and
**Analyze whole track** run it on demand; **Cancel analysis** (Esc, or the
button in the status bar) stops it.
Background activity messages clear when the task finishes or stops after cancellation.
Errors and result summaries remain visible.
Status messages stay on one line without resizing the piano roll; hover over
the status text to read the full message, including any original line breaks.

**Navigate** with **Previous** / **Next** (Left / Right arrow, or PgUp / PgDn);
each step scrolls half a screen. The label between them
shows the visible bars. **Zoom out (-)** / **Zoom in (+)** next to them (also
Ctrl + mouse wheel, View > Zoom, Ctrl+- / Ctrl+=) show 1, 2, 4, 8 or 16 bars per
screen; the readout beside the buttons shows the current setting. **Up / Down arrows**
shift the piano-roll view up / down one octave without transposing notes, stopping
at MIDI pitches 0 and 127. The plain mouse wheel scrolls pitch; **Ctrl + Shift + wheel**
zooms pitch (not in the drums view). **Grid** sets the editing step (1/4 to 1/32, including
triplets).

The piano roll shows the stem's spectrogram behind the notes: a sustained
pitch is a bright horizontal band. Brightness is measured against the loudest
moment of the whole stem, so bars where the instrument is silent stay dark
(the spectrum appears a moment after a stem is first shown, once that level is
measured). The chords view's pitch-class strip is scaled to the whole song the
same way, on a 40 dB log scale so weaker chord tones still show. Blue notes are model proposals; green ones
are yours. **Hover over a note** to see its name, the bar and beat it starts
on, its length and velocity.

**Left-click a blue note with the Brush tool to confirm it.** It turns green,
keeping its pitch, timing, and velocity, and stays visible even with **AI notes**
set to **None**. Confirmation is saved and supports Undo/Redo; clicking a green
note just selects it. A click on the release edge also confirms; dragging the
edge still resizes. The Info tool continues to open the note dialog without
confirming merely by opening it.

Velocity comes from how loud the note is in its stem: the power at the note's
pitch and its first harmonics over the note's first 200 ms, compared with the
loudest notes of the whole stem (General MIDI curve, so 12 dB quieter is about
half the velocity). This applies to bass, legacy vocals and Basic Pitch; the piano_hr
model predicts piano key velocities itself. Drum velocity is the RMS level at
each hit in that kit piece's band (kick 30-150 Hz, snare 150 Hz-5 kHz, toms
60-600 Hz, hi-hat 6-16 kHz, cymbals 4-16 kHz), so a kick and a hi-hat on the
same beat are measured separately; each piece is scaled to its own loudest
hits. YourMT3+ preserves its decoded velocities, without interpreting them as
confidence. Notes you draw start at 100.

### Syllable-aware vocals

On a **vocals** track, choose **madmom_cnn** or **madmom_brnn**. These combine
the reviewed neural onset models with CREPE full pitch. Existing `pyin` and
CREPE choices are unchanged. Five compact controls sit immediately right of
the model selector; hover over a label or value for its description. They are
enabled for the two new models and saved per track:

| Control | Starting value | Useful tuning range | Effect |
|---|---|---|---|
| Onset | 0.30 | 0.20-0.60 | Lower finds weaker syllable attacks. |
| Voice | -35 dB | -45 to -25 dB | Reject quiet regions relative to this visible crop. |
| Spacing | 0.12 s | 0.10-0.16 s | Suppress closely spaced attack duplicates. |
| Slur hold | 0.09 s | 0.06-0.12 s | Require pitch changes to persist; lower accepts shorter slur notes. |
| Min split | 0.08 s | 0.05-0.10 s | Minimum proposed split interval before articulation shortening. |

Separate syllables, including repeats of the same pitch, leave **half the
selected grid cell** between notes. Pitch changes within one syllable have
**exactly touching** note ends/starts, with continuous built-in synth playback.
For example, Grid **1/4** is a quarter-note cell, so its articulation gap is
an eighth note. These are musical-time gaps that follow tempo pins, not fixed
millisecond silences. Natural phrase rests can be longer. A coarse grid merges
nearby proposals; use a finer grid if syllables or slur notes disappear.

These models analyze and replace notes/onset markers **only in the displayed
bars**. Moving or zooming the view cancels pending work and prevents an old
window's result being published. Changing controls reuses that crop's neural
evidence, without updating hidden notes. Existing notes that cross a window
edge keep their outside fragments; those protected edges are not rewritten to
force an articulation gap. Whole-track analysis is disabled for these models.
Review successive windows to build coverage; export can explicitly export
partial coverage but never silently transcribes the rest.

The project key gently corrects an out-of-key pitch by at most one semitone.
Velocity uses the visible crop's RMS energy rather than whole-stem harmonic
power. Direct onset additions/removals and corrected held notes still guide
boundaries, and authored notes remain protected. Legacy learned onset-feature
profiles are retained for legacy models, but are not used/imported by these
two neural models; their Onset control sets acceptance instead.

There are two tools, with buttons next to **Grid** (or **View** menu):

- **Brush (B)**, the default, adds, resizes and deletes notes:

| Action | Result |
|---|---|
| Click an empty cell | Add a note one beat long |
| Drag from an empty cell | Add a note and set its length, with the end snapping to half-grid intervals |
| Click a note | Select it |
| Drag a note's right edge | Change its length, snapping the end to half-grid intervals |
| Right-click a note, or **Delete** | Reject it (that pitch stays silent there) |
| Right-click empty space | *Rest for this beat* / *Rest for this bar* (silence all pitches) |
| Click the onset lane | Add an onset (note-attack boundary) |
| Right-click the onset lane | Remove the nearest onset |

Note starts stay on the selected grid; dragged note ends can land halfway
between grid lines, with a minimum length of half a grid cell. Extend a note
to the next note's start for connected (legato) playback, or leave a half-grid
gap for a separate articulation. Clicking an empty cell still adds a
one-beat note.

Onset markers follow the resolved note starts: simultaneous notes share one
marker, and a marker disappears when its last note is removed or moved.
Detected candidates with no note are not shown as onsets. Clicking the onset
lane still teaches an attack boundary for reanalysis, but it only shows a
marker once a note actually starts there.

- **Info (I)**: click a note to open a small editor with its bar, beat, length
  in beats, pitch and velocity, plus **Delete note**. Beats accept `2 1/2`,
  `5/2` or `2.5`; pitches accept `Eb4` or `63`. Clicking empty space does
  nothing, so you can inspect notes without adding any. **Right-click** opens
  a context menu instead of immediately deleting: on a note it includes
  **Note details...** and **Delete note**, plus shared beat/downbeat controls
  at the exact clicked audio position. Empty space offers the timing controls
  too. Detect beats first to enable timing actions. The upper lane shows
  counted beats and pins alongside note onsets without editing onset boundaries;
  switch to **Brush (B)** to add or remove onsets. Starts must fit the selected
  grid; note ends can use half-grid intervals, just like dragging.

**Drums** have their own view: the spectrogram fills the top half, and below it
are ADTOF's five lanes - Cymbal, Hi-hat, Tom, Snare and Kick - instead of piano
keys. Clicking a lane adds that drum (GM notes 49, 42, 47, 38 and 36). Imported
drum notes appear in their lane (an open hi-hat, 46, shows as Hi-hat and keeps
its note); kit pieces ADTOF does not model are not drawn, but stay in the MIDI
export, and a note at the top of the view says how many are hidden.

**Edit > Undo / Redo** (Ctrl+Z / Ctrl+Y) work per track. **Edit > Clear
track** and **Clear all** remove notes and corrections (undoable per track).
**File > Import MIDI into track...** loads existing notes into the selected
track.

Your corrections teach lightweight onset logic for the project; they do not
retrain the large pitch models.

For **legacy vocals**, corrected note starts teach where attacks belong, and the
interior of each corrected note teaches where there must not be another
attack. You can lengthen a sustained note without removing all its old onset
markers: the note takes precedence over stale markers inside it when the
region is transcribed again. Separate corrected notes on the same pitch still
teach separate syllables. Note ends are not automatically treated as attacks,
and untouched regions are not assumed to be correct or silent.

This evidence is derived from the current edits, so undo/redo also changes
what the vocal onset model learns. With held-note corrections, vocal onset
acceptance uses a more sensitive threshold (0.25 rather than 0.5) to avoid
losing quieter syllables as the negative examples accumulate. Pitch-change
evidence is retained outside corrected held notes. Other instruments keep
their existing onset policy. Older saved vocal analysis is refreshed with
this policy; corrected notes remain protected.

## 6. Chords

Choose **Chords** in the **Track** selector. From the top down, the view shows
the mix amplitude, the chord lane, and a 12-row chroma (one row per pitch
class, bright where that note sounds). The chord's tones are outlined in the
chroma, and bar/beat lines run through every lane.

- **Edit > Detect chords** proposes chords on the beat grid
  (triads, sevenths, suspended and slash chords). It requires a mix.
  Suspended chords are named as sus4: the notes of Csus2 show as `Gsus4/C`,
  and a typed `Csus2` becomes `Gsus4/C`.
  Separate stems first for the best results: with current stems, chords come
  from piano + guitar + other + bass, the slash bass comes from the bass stem
  alone, drums are ignored, and vocals (slides, vibrato, melody passing tones)
  only help detect the key unless the instruments are silent. Without current
  stems the app offers to separate them first (then detects chords
  automatically) or to use the full mix, where kick drums and vocals can add
  wrong notes. The chroma lane shows the same audio the detector reads.
  The key is judged from piano + guitar + other plus half-level vocals, without
  bass or drums: a loud bass sitting on the IV chord otherwise pulls the key
  there. When stems exist and you have not set the key yourself, **Detect
  chords** re-detects the key from the stems (the first detection used the
  mix) and the status line reports any change.
  Key and chord detection use the same spectral preprocessing. The project key
  gently favors chords in that scale when the audio is ambiguous; clear borrowed
  chords can still win. Set or correct the key with **Edit...**, then rerun
  **Detect chords** to apply it to the chord proposals. The built-in detector
  is a baseline, not a definitive transcription.
- **Model > Chord recognizer** (or the **Model** box when the Chords track is
  selected) chooses the detector used by **Detect chords**:
  **Built-in templates** (default) or **BTC transformer (deep learning)**, the
  MIT-licensed Bi-directional Transformer for Chord recognition (Park et al.,
  ISMIR 2019) trained on lead sheets with a 170-chord vocabulary. BTC is much
  steadier from beat to beat and works well on the full mix, so it does not
  ask to separate stems (it still uses current stems when present). The first
  use installs PyTorch if needed and downloads the ~12 MB model once. Sixth
  chords are shown as their triads, and an inversion slash is added only when
  the bass stem plays a chord tone other than the root. The key still comes
  from the key detector. From the command line: `chords --model btc`.
- Chord, chroma-row and note names are spelled for the project key: `G#m`,
  not `Abm`, in B major; `Bb`, not `A#`, in F major. Common chromatic notes use
  the usual altered degree (`Eb` as the flat 3rd in C, `G#` as the leading tone
  in A minor). Changing the key respells existing chords without redetecting,
  and MusicXML export writes the matching key signature. Detected keys use the
  simpler enharmonic (`Db`, not `C#`).
- **Model > Advanced > Show key CQT...** plots the 12 pitch classes the key
  detector reads, folded over octaves, as overlaid traces: Low (A0-B2), Mid
  (C3-B4), High (C5-C8) and All. All is exactly the key-detection profile and
  the bands add up to it; the legend shows each band's share of the energy and
  the key it alone would give. **Compare shapes** scales each trace to 100% so
  quiet registers can be compared. Choose the audio: the key detection input,
  the full mix, or any stem.
- **Double-click** a chord to type a symbol (`Am7`, `C/E`, `F#m7b5`, `N.C.`).
- **Right-click** for *Alternatives*, *Split at this beat*, *Merge with next
  chord* and *No chord*.
- **Drag** the boundary between two chords to move the change.
- **Click** a note row in the 12-note grid under a chord to add that note;
  **right-click** a row to remove it. The chord is renamed to the symbol with
  exactly those notes (adding A to C gives `Am7/C`, the notes of C6), or to
  the closest symbol in the vocabulary if none matches exactly. The outlined
  rows show the result.
- **File > Import chord MIDI...** turns a MIDI file of block chords into
  symbols.

Undo and redo work in the Chords view too. Playback adds voiced chords.

## 7. Listen

**Play** (or Space) plays from the start of the visible window; **Stop** ends
it, and the view follows the playhead. In a stem view you hear the stem mostly
on the right and the notes mostly on the left; in the Chords view you hear the
mix with the chords.

The **Speed** selector beside **Play** offers **100%**, **90%**, **75%**, and
**50%** playback without changing pitch. Notes, chords, metronome clicks,
external MIDI, and the playhead follow the selected speed. Changing speed while
playing restarts from the current playhead; preparing slowed audio may take a
moment. Source files, the tempo map, and exports are unchanged. Speed starts at
100% when the app opens. Downbeat tapping always uses 100% playback (the speed
selector is disabled while tapping), so tap-latency calibration remains valid.

The **Audio ↔ Synth** slider next to **Play** sets the balance: all the way
left plays only the audio, all the way right only the synth (notes, chords and
metronome clicks), and the center plays both at full level. The other side
fades evenly in loudness: a quarter of the way is about 10 dB quieter, half way
20 dB, and it is silent at the end. Double-click it to
center. Moving it during playback restarts the sound from the playhead with the
new balance. With **MIDI out**, it also sets the synthesizer's channel volume.
The setting is remembered.
**File > Playback settings...** chooses how notes sound:

- **Built-in tones** (default) are rendered into the same buffer as the audio,
  so they are sample-aligned. Use them to judge timing.
- **MIDI out** sends notes to a synthesizer. Windows delays MIDI differently
  from audio; use **MIDI sync test** and adjust **MIDI latency** until the click
  and the wood block sound together. The setting is stored on this computer.

## 8. Export

- **File > Export MIDI...** (Ctrl+E): choose all tracks plus chords, a single
  role, or chords only. If tracks are not fully analyzed, the app asks before
  writing a partial file.
- **File > Export score (MusicXML)...** writes a MuseScore-ready score with
  the transcribed tracks and the chord symbols on the top staff. Bass and drums
  are named **Electric Bass** and **Drum Kit**. Scores and chord-only lead sheets
  have one opening tempo: total quarter-note beats divided by elapsed time
  between the first and last timing anchors (or the set BPM without a timing
  map). No later tempo changes are exported, even for playback. MIDI retains
  the original tempo map to stay aligned with the audio.
  Score export asks for **Quantization**, defaulting to **1/16** notes independently
  of the editor grid. Triplets are only used when explicitly selected (available
  in meters with quarter-note notation beats). Gaps shorter than the selected
  quantization are omitted as ordinary articulation, without a slur. Originally
  touching notes receive a slur; rests at least one quantization step long remain.
  These changes affect only the exported score, not project notes or MIDI.
  For command-line score export, use `--quantization` in quarter-note units:
  `1/4` means sixteenths (default), `1/8` means thirty-seconds, and `1/6`
  means sixteenth triplets.

## Troubleshooting

| Symptom | What to check |
|---|---|
| A menu item is grayed out | Hover over it: the tooltip says which stage is missing (mix or counted bars). |
| The count is at double or half speed | Pin two downbeats several bars apart; the count follows your pins. |
| Tapping did not count bars | Tap the first beat of each bar (not every beat), for at least four bars, in a section with clear drums or bass. Or pin downbeats by hand. |
| Clicks drift late in the song | Pin another downbeat near the drift; the tempo map follows it. |
| No notes appear | Check the model selector and the status bar for errors. `other` never produces notes. |
| Model install fails | First-use installs need an internet connection and write access to the app's Python environment (use a virtual environment rather than a system-wide Python). ADTOF also needs `git`. Or pre-install with `.\install.ps1 -Models`. For a base-only trial, use vocals with `pyin`. |
| Notes sound late with a MIDI device | Run **MIDI sync test** in **File > Playback settings** and raise the latency. |
| Reopening reports missing media | Restore referenced files to their expected locations; projects do not embed the original mix. |

For an environment report, run:

```powershell
.\.env\Scripts\python.exe -X utf8 -m audio_to_midi doctor
```

For architecture and validation notes, see the
[execution plan](execution-plan.md); for the interface design, see the
[UI redesign plan](ui-redesign-plan.md).
