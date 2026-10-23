# UI redesign plan (draft)

Status: **implemented (passes 1–6)**. The chord recognizer is still the
placeholder; evaluating lead-sheet recognizers is a separate track (see the
end of this document). Assumptions made during implementation are listed
under "Implementation notes".

## Goals

1. A proper menu bar; the main window shows only what is used constantly.
2. One window. The separate "Prepare the full mix" dialog is retired; beat
   detection, tapping, warping, and validation happen on the main timeline.
3. A gated workflow: a project moves through stages, and commands that are not
   valid yet are grayed out with a tooltip explaining why.
4. Nothing currently possible becomes impossible (see the control mapping).

## Decisions made

| Topic | Decision |
|---|---|
| Double (x2) / Half (/2) meter buttons | Removed. Meter is typed directly. |
| Note add / reject / rest | Piano-roll interactions, not menu items. |
| Model selection | Visible combo box in the main view. |
| Analyze button | Removed from the main view; visible-window analysis is automatic. |
| Tapping downbeats | Main view, mix only, toggled from the Edit menu; Space records a downbeat. |
| Warping | Required: all downstream work runs on audio at one exact tempo. |
| Demucs separation | Core flow, in the Edit menu, enabled only after timing is confirmed. |
| Stems-only projects | Not allowed. Every project starts from a full mix. |
| Timing validation gate | A simple confirmation checkbox. |
| Re-warp with existing notes | Warn first; notes keep their bar/beat positions. Edit > Clear track / Clear all are available for a clean slate. |
| Clearing notes | Two commands: **Clear track** (selected track) and **Clear all** (every track). Both remove generated notes and user edits. |
| Attach prepared mix | Retired. Preparation always lives inside the project folder. |
| New project | Asks for the name and project folder up front. |
| Backward compatibility | None. Older projects and preparation folders need not open; the project format and stage data can be redesigned cleanly. |
| Chords | A dedicated Chords view (see below), selected as `Chords` in the track selector. |
| Chord recognition | Use a lead-sheet chord recognizer from open source, chosen by evaluating candidates. Until then, a simple placeholder recognizer (below). |
| Chord vocabulary | Triads (maj, min, dim, aug, sus2, sus4), sevenths (7, maj7, m7, m7b5, dim7, mMaj7), and slash chords. |
| Chord export | Voiced MIDI chord track, and a MusicXML lead sheet. |
| Licensing | GPL and non-commercial weights are acceptable for personal use, behind the existing acknowledgement gate. |

## Stage model

```text
① Mix  →  ② Timing  →  ③ Stems  →  ④ Transcribe  →  ⑤ Export
```

| Stage | Done when | Unlocks |
|---|---|---|
| Mix | A full mix is imported | Detect, tap, beat editing, warp |
| Timing | A warp (or "use as-is") revision exists and the user ticked **Timing confirmed** | Separate stems, import stems |
| Stems | At least one stem track exists from the confirmed timing | Model selection, analysis, editing |
| Transcribe | Notes exist on at least one stem track | Export |

A slim **stage strip** under the menu bar shows the stages and the single
next action, e.g. `✔ Mix  ● Timing: drift 42 ms — warp needed  ○ Stems  ○ Transcribe`.

Stage state is stored in `project.stages["timing"]`, holding the warp
revision id and the confirmed flag, alongside the existing
`project.stages["preparation"]`.

## Menus

```text
File
  New project…            (choose name + folder up front)
  Open project…
  Save                    Ctrl+S
  ─────────────
  Import mix…
  Import stems…           [needs confirmed timing]
  Import MIDI…            [needs a stem track]
  Import chord MIDI…
  ─────────────
  Export MIDI…            [needs notes]
  ─────────────
  Exit

Edit
  Undo                    Ctrl+Z
  Redo                    Ctrl+Y
  Clear track…            [stem track selected]
  Clear all…
  ─────────────
  Set tempo, meter, and key…
  Detect tempo, meter, and key
  Tap downbeats           [toggle; mix selected]
  Warp to fixed tempo…    [mix selected, ≥2 downbeats]
  Timing confirmed        [checkbox; needs a warp]
  ─────────────
  Separate stems (Demucs)…  [needs confirmed timing]

View
  Brush tool (B) / Info tool (I)
  Spectrogram             [toggle]
  Beat lane               [toggle]
  ─────────────
  Playback settings…      (note sound, MIDI out, latency, sync test)

Model
  Analyze whole track
  Automatically analyze   [checked]
  Cancel analysis
  ─────────────
  Import onset profile…

Help
  Getting started
  Keyboard shortcuts
  About
```

## Main window

```text
File   Edit   View   Model   Help
──────────────────────────────────────────────────────────────
✔ Mix   ✔ Timing   ● Stems: separate or import   ○ Transcribe
Track: [Bass ▾]    Model: [CREPE tiny ▾]
120 BPM · 4/4 · Bar 1: 0.350 s · Key: D major       (click to edit)
[Previous]   Bars 9–10   [Next]                     Grid: [1/16 ▾]
──────────────────────────────────────────────────────────────
 beat lane: detected beats · downbeats · pins · ▼ taps
──────────────────────────────────────────────────────────────
                         PIANO ROLL
                 (spectrogram, onsets, notes)
──────────────────────────────────────────────────────────────
        (no note table: hover a note for its stats)
──────────────────────────────────────────────────────────────
[▶ Play] [■ Stop] [♪ Click]          Analysis: partial  [Cancel]
```

- **Track selector:** `Mix` first, then stem tracks, then `Chords`.
  - Selecting `Mix` shows the mix CQT with the beat lane and disables the model selector.
  - Selecting `Chords` opens the Chords view; the model combo then lists chord recognizers.
- **Grid:** musical labels (1/16, 1/8 triplet, …); the stored value stays in quarter notes.
- **Click (metronome):** a toggle next to Play; high click on downbeats.
- **Cancel:** visible only while work is running.
- **Space:** play/pause. While tapping is on, Space records a downbeat instead.

### Piano-roll interactions (core editing)

| Action | Gesture |
|---|---|
| Add note | Click empty grid cell (snaps to the grid) |
| Resize note | Drag note end |
| Select note | Click note |
| Reject note | Right-click note, or select and press Delete |
| Rest in range | Right-click > Rest here (covers the note or grid cell) |
| Edit exact values | Info tool (I): click a note, edit in a small dialog (replaces the note table) |

## Stage 2 — Timing, in the main view

With `Mix` selected:

1. **Detect tempo, meter, and key** runs beat detection and key detection on
   the mix. The beat lane shows candidates, faded by confidence; counted beats;
   and downbeats. The summary line updates.
2. **Tap downbeats** plays the mix from the left edge of the view; Space marks
   each bar's beat 1. Turning it off fits the taps: latency correction, snapping
   to strong beats, and a meter guess, reusing `beats.py`.
3. **Beat-lane context menu** (replaces the preparation dialog buttons):
   Set downbeat (D), Set bar/beat (Enter), Unpin (U), Remove beat (Delete),
   Add beat here, Clear pins.
4. **Drift** appears in the stage strip in ms.
   - Steady: "use as-is". The audio is copied verbatim (register).
   - Drifting: "warp needed".
5. **Warp to fixed tempo…** opens a small dialog with the target BPM, warp
   anchors (smoothed / every beat), lead-in policy (trim / pad), and a
   **Use as-is (no stretching)** option for steady audio. It creates a warp
   revision, and the view switches to the warped mix.
6. **Validation view.** After a warp, beats are re-detected on the warped audio
   and drawn against the grid, so misalignment is visible. Listen with the click.
7. **Timing confirmed** checkbox (Edit menu, also clickable in the stage strip)
   unlocks stage 3.

Preparation artifacts are written to `<project folder>/preparation/`. This
is why **New project** asks for the folder up front.

## Stage 3 — Stems

- **Separate stems (Demucs)…** runs on the confirmed warp revision. It asks once,
  then installs Demucs and downloads its weights automatically on first use; the
  user never picks a Python or model folder.
- **Import stems…** is for external separations (UVR, etc.). The dialog states
  that stems must come from the confirmed warped mix. Existing duration and
  sample-rate checks apply.

## Re-warping after stems or notes exist

Unchecking **Timing confirmed**, or warping again, shows a warning that lists:

- the number of notes and edits on each track: they keep their bar/beat
  positions but may no longer sit on the audio;
- Demucs stems: stale; re-separate;
- imported stems: must be re-imported from the new warp;
- the generated-analysis cache: discarded and re-run;
- onset training examples (stored in seconds): dropped.

The old warp/separation revisions remain on disk (immutable revisions).

**Edit > Clear track…** (selected track) and **Edit > Clear all…** (every
track) remove generated notes and user edits so transcription can restart on
the new audio. Tempo, meter, key, and bar-1 timing are kept. Both ask for
confirmation and can be undone during the session.

## Control mapping (nothing lost)

### Main window

| Current control | New home |
|---|---|
| New / Open project / Save | File |
| Import warped mix | File > Import mix (then the Timing stage) |
| Import stem | File > Import stems (gated) |
| Import MIDI / Export MIDI | File |
| Prepare mix | Retired: main-view Timing stage |
| Separate mix | Edit > Separate stems (Demucs) |
| Attach prepared mix | Retired: preparation is in the project folder |
| Import onset profile | Model > Import onset profile |
| Import chord MIDI | File > Import chord MIDI |
| Edit chord notes | Chords view (chord symbols, not individual notes) |
| BPM / meter / offset / key fields + Apply | Edit > Set tempo, meter, and key… (or click the summary line) |
| Grid | Main view, Grid combo |
| Track / model selectors | Main view |
| Previous / Next | Main view |
| Analyze visible | Automatic |
| Analyze whole track / Cancel | Model menu; Cancel also in the status bar |
| Reanalyze after edits | Model > Automatically analyze |
| Add note / Apply selected row / Reject selected / Rest in selection | Piano roll (Brush) + Info tool |
| Undo / Redo | Edit menu + shortcuts |
| Analyze key/chords | Key: Edit > Detect tempo, meter, and key. Chords: automatic in the Chords view (Model > Analyze whole track) |
| Preview audio + notes / Stop | Transport bar |
| Note sound / MIDI latency / MIDI sync test | View > Playback settings… |

### Preparation dialog

| Current control | New home |
|---|---|
| Mix path / artifact folder | Implicit: the project mix, `<project>/preparation/` |
| Meter field | Edit > Set tempo, meter, and key… |
| Double / Half | Removed |
| Detect beats | Edit > Detect tempo, meter, and key |
| Tap downbeats | Edit > Tap downbeats (toggle) |
| Set downbeat / bar-beat / Unpin / Remove / Clear pins | Beat-lane context menu + D / Enter / U / Delete |
| Play view with beat clicks | Transport Play + Click toggle |
| Lead-in policy / Warp anchors / BPM / bar-1 time | Warp to fixed tempo… dialog |
| Register without warping | Warp dialog > Use as-is |
| Warp original mix | Warp dialog > Warp |
| Confirm checkbox | Edit > Timing confirmed |
| New revision checkbox | Implicit: re-warp warning |
| Attach to empty project | Automatic |
| Audition prepared mix | Normal Play on the Mix track |

## Implementation passes

1. **Menus and cleanup.** Remove Double/Half; add the menu bar; timing dialog;
   model combo stays in the main view; playback settings dialog; collapsible
   note table; piano-roll context menu for rest/reject. Behavior unchanged.
2. **Mix as a track view.** Add `Mix` to the track selector and give the timeline
   a mix mode (CQT, no notes, model disabled).
3. **Beat lane and detection.** Draw beats, pins, and taps on the timeline;
   Detect tempo, meter, and key; metronome click; beat-lane context menu.
4. **Tapping, warping, and gating.** Tap toggle with Space; warp dialog;
   validation re-detect; stage strip; `stages["timing"]`; gray-out rules;
   re-warp warning; Clear track / Clear all. Retire the preparation dialog and
   Attach prepared mix.
5. **Chords view with the placeholder recognizer.**
   - Chord-segment data model.
   - Amplitude, chord, chroma, and bar/beat lanes, and chord-lane editing.
   - Chord playback, and voiced-MIDI and MusicXML export.
   - Retire the chord-notes dialog, and replace the template matcher in
     `core/chord_analysis.py` with the placeholder.
6. **Docs and tests.** Rewrite `getting-started.md` around the stages; update
   the UI tests (`tests/test_v2_ui.py`, `tests/test_v2_workflow.py`).

Each pass leaves the app working and tested.

**Later, separate track — chord recognizer evaluation.** Once the
song-generator test set exists:
- build the harness;
- score the placeholder and the candidates, with and without grid smoothing;
- choose a default recognizer and add it as a model backend.

## Chords view

Selected as `Chords` in the track selector, like `Mix` is for timing. It needs
confirmed timing, and it always analyzes the **warped full mix**, not the stems.

```text
Track: [Chords ▾]    Model: [<chord recognizer> ▾]
120 BPM · 4/4 · Bar 1: 0.350 s · Key: D major
[Previous]   Bars 9–12   [Next]
──────────────────────────────────────────────────────────────
 amplitude  ▁▃▆█▆▃▂▁▃▆█▇▅▃▁▂▄▆█▆▄▂▁▃▅▇█▆▄▂ (mix waveform/RMS)
──────────────────────────────────────────────────────────────
 chords     | D          | Bm7    A/C# | G          | A7sus4 A7 |
──────────────────────────────────────────────────────────────
 B  ░░░░▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 A# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    12-row chroma
 A  ▓▓▓▓▓▓░░░░▓▓▓▓▓▓░░░░░░░░░░▓▓▓▓▓▓▓▓▓    (CQT folded to
 …                                         pitch classes)
 C# ░░░░░░░░░░▓▓▓▓░░░░░░░░░░░░░░▓▓▓▓░░░
 C  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
──────────────────────────────────────────────────────────────
 9    ·    ·    ·  | 10   ·    ·    · | 11 …   (bars and beats)
```

### Lanes

- **Amplitude:** mix waveform/RMS envelope, which helps locate section changes
  and hits.
- **Chord lane:** lead-sheet symbols (`Bm7`, `A/C#`, `A7sus4`) spanning their
  bar/beat range.
  - Generated chords and user-edited chords have distinct colors.
  - Low-confidence chords are shown faded.
- **Chroma:** 12 rows, C to B. This is the CQT folded to pitch classes, normalized
  per frame so harmony stays visible in quiet passages. The chord tones of the
  current chord are outlined on the chroma, so a mismatch is visible.
- **Bar/beat grid:** vertical lines through every lane: strong at bars, weak at
  beats, with bar numbers.
- **Key:** shown in the summary line. A key-change lane is an option for later.

### Editing

| Action | Gesture |
|---|---|
| Change chord | Double-click a segment, type a symbol (validated, e.g. `F#m7b5`) |
| Pick an alternative | Right-click > the recognizer's next-best candidates |
| Split | Right-click > Split at beat (or click a beat line in the chord lane) |
| Merge | Right-click > Merge with next |
| Move boundary | Drag; snaps to beats |
| No chord | Right-click > N.C. |

Undo and redo work as in the other tracks. User-edited segments are kept when
analysis runs again.

### Data model

Chords become their own type rather than piano-roll notes:

```text
ChordSegment(start: Fraction, end: Fraction,   # quarter-note positions
             symbol: str,                        # e.g. "A7sus4", "D/F#", "N"
             source: generated|authored,
             confidence: float | None,
             alternatives: list[(symbol, score)])
```

`project.chords: list[Note]` is replaced (no backward compatibility needed).
Voicings are derived from the symbol for playback and export.

### Playback and export

- **Play:** the mix plus a simple voicing of the chords, plus the optional click.
- **Export:**
  - a MIDI track of voiced chords, written by MIDI export alongside the other tracks;
  - a MusicXML lead sheet: chord symbols per bar, in the project meter and key. It
    can later include a melody line from the vocal track.

### Placeholder recognizer (first implementation)

A simple grid-aware chroma threshold, until the evaluation is done:

1. Compute the 12-bin chroma of the warped mix and average it per beat.
2. Normalize each beat. Pitch classes above a threshold, relative to the
   strongest, are the chord notes.
3. Name the pitch-class set against the vocabulary templates, choosing the best
   root. The bass note comes from the lowest strong CQT bin; if it is not the
   root, the chord becomes a slash chord.
4. Merge equal adjacent beats into segments, and suppress changes shorter than a
   beat.

The threshold can be adjusted. This is enough to build and use the whole view,
and it becomes the baseline in the evaluation.

### Recognizer: evaluate before choosing

The recognizer sits behind the same backend interface as the pitch models and
appears in the model combo when `Chords` is selected. Several recognizers can
stay installed and available.

**Use the known grid.** After the timing stage, beats and bars are exact. Any
recognizer that outputs frame-level probabilities can therefore:

1. average the probabilities per beat;
2. run a Viterbi pass over beats that discourages chord changes off strong
   beats and penalizes very short chords.

This lead-sheet-style smoothing is ours to build and works on top of any
backend. It is likely to matter as much as the choice of model.

**Candidates to evaluate.** All details need verification during the spike;
web summaries of these projects were unreliable.

| Candidate | Notes to verify |
|---|---|
| Chordino / NNLS Chroma (via `chord-extractor` or a Vamp host) | Lead-sheet oriented: sevenths, slash chords; widely used baseline; GPL; Windows install path |
| BTC-ISMIR19 (jayg996/BTC-ISMIR19) | Transformer; large vocabulary (~170 classes incl. sevenths); PyTorch; weights licensing |
| crema (bmcfee/crema) | Structured large vocabulary, including inversions; old TensorFlow/Keras dependency |
| madmom chord models | Deep chroma / CNN-CRF; major/minor only (baseline); Python version compatibility |
| autochord | Simple, major/minor only; low priority |
| Newer research models (e.g. ChordFormer, 2025) | Check whether code and weights are released |
| Current template matcher (`core/chord_analysis.py`) | Baseline to beat |

**Evaluation method:**

1. **Test set (separate exploration; placeholder).** Synthetic songs from
   the user's MIDI-style song generator. Each song has a known chord
   progression (exact ground truth), rendered audio, and the same audio
   separated with Demucs. This tests recognizers on the full mix and on
   Demucs output, and allows tuning (thresholds, smoothing) without manual
   annotation. Real songs, hand-checked in the Chords view, can be added later.
2. **Harness:** a script that runs each backend, with and without the
   grid-aware smoothing, and scores it with `mir_eval.chord`. Scores are
   duration-weighted at three levels: root, majmin, and sevenths. Segmentation
   quality (over/under-segmentation) and the number of edits needed per song
   are also measured.
3. **Practical criteria:** CPU speed on a full song, Windows installability with
   Python 3.12, and license.
4. **Output:** a short results table in `docs/`. Choose a default recognizer,
   and keep the others as selectable models if they are useful.

## Implementation notes

Choices made without a decision; revisit any of them.

- Importing a mix runs **Detect tempo, meter, and key** automatically.
- The meter is inferred only from tapped downbeats (beats between taps);
  otherwise it comes from detection or Edit > Set tempo, meter, and key.
- In the Mix view, **D** pins the selected beat as a downbeat, **U** unpins it,
  **Delete** removes it. **Space** taps while tapping, otherwise play/stop.
- The warp dialog defaults to **keep audio before bar 1** (pickup) and to
  **Use as-is** when the detected tempo is steady.
- After warping, beats within ±30 ms of the grid are green, others red.
- Chord detection requires confirmed timing; chord MIDI import needs a mix.
- Chord playback and voiced-MIDI export: the bass (or slash) note in octave 2
  (MIDI 36–47), chord tones in close position from middle C.
- Placeholder recognizer: the CQT is raised to the 4th power and folded into a
  chroma; per beat, pitch classes at ≥ 0.7 of the strongest are matched against
  the chord templates (F-measure), and equal neighbors are merged.
- Playback starts at the window start and runs to the end of the file; the
  view pages along with the playhead.
- Changing the Grid subdivision invalidates analysis, as before.
- `preparation_ui.py` and `beat_view.py` were removed; their behavior moved to
  the Mix view, `timing.py` and `dialogs.py`.
- Models install themselves (`provision.py`): on first use of a model or of
  Separate stems, after one confirmation, missing packages are pip-installed
  into the app's own Python and weights downloaded (piano-HR's checkpoint by the
  app, since the library's `wget` call fails on Windows). Demucs runs in a
  subprocess of the same Python (`separation_worker.py`) so Cancel can stop it.
  The Demucs Python/model-folder fields and its consent checkbox are gone
  (Demucs code and weights are MIT); the ADTOF/piano-HR license notes remain.

- The note table was removed. Hovering a note shows name, start bar/beat,
  length and velocity; View > Brush (B) / Info (I) switch tools, and Info opens
  a small per-note editor with Delete.
- Drums get their own view: the spectrogram in the top half, then ADTOF's five
  lanes (Cymbal, Hi-hat, Tom, Snare, Kick; GM 49, 42, 47, 38, 36). Nothing
  beyond what ADTOF detects. Other GM drum notes map to their ADTOF lane and keep
  their note; unmodeled kit pieces are hidden but exported, with a count shown.

## Open questions

1. **Chord test set.** Design the interface with the song generator: the file
   format for chord ground truth, and the tempo and meter conventions. This is a
   separate exploration.
