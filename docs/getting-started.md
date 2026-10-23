# Getting started with Audio to MIDI

Audio to MIDI turns a song into editable MIDI. You work through stages, each
shown in the strip at the top right of the window:

**Mix → Count bars → Warp → Stems → Notes → Chords → Export**

1. Import the full mix.
2. Find its bars and beats (detect, then fix with pins or tapped downbeats).
3. Warp the mix to one steady tempo.
4. Separate or import stems.
5. Transcribe and correct notes per stem.
6. Detect and correct a lead-sheet chord track.
7. Export MIDI and/or a MusicXML lead sheet.

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
time you use it: when you pick a model (CREPE, Basic Pitch, Piano-HR, ADTOF) or
run **Separate stems**, the app asks once, then pip-installs the packages into
the Python it is running on and downloads the weights to their usual cache. You
never choose a Python or a model folder. This needs an internet connection the
first time; the status bar shows progress, and **Cancel** stops it. ADTOF needs
`git` on PATH. ADTOF drum weights are non-commercial; review the piano-HR
checkpoint terms before relying on it.

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
and workers have been removed. This does not delete or migrate any projects,
audio, MIDI, profiles, logs, or model caches. Existing project-editor projects
and preferences keep their current formats and settings location. Compatible
onset examples can still be imported from older profiles; legacy session state
is not converted automatically.

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
   is saved in a new folder named after the project; the warped mix and
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
- Pin two downbeats several bars apart; the tempo comes from your pins, not
  from the detector's guess. Add more pins where the count drifts.

**Tap downbeats instead.** Check **Edit > Tap downbeats**. The mix plays from
the current window, and you press **Space** on the "one" of every bar. Your
taps appear as markers. Uncheck the item when done (tap at least four bars in
a clear section). The app asks which bar your first tap was, snaps each tap to
the nearest detected beat, pins them, and infers the meter from the beats
between taps. The status line reports how many bars were counted and how late
you tapped on average.

When you are not tapping, **Space** plays and stops. **View > Metronome click**
adds clicks (high on downbeats) so you can hear whether the count lines up.

The stage strip shows **Count bars ✓** once the bars are counted.

## 4. Warp the mix

**Edit > Warp mix to tempo...** writes a constant-tempo copy of the mix into
the project folder, so the bar grid lines up with the music everywhere.

- **Use as-is** skips stretching. It is preselected when the mix already has a
  steady tempo (for example, it was produced to a click or warped in a DAW).
- Otherwise choose the **Target tempo**, how the **Anchors** follow your beats,
  and whether to keep or trim the audio before bar 1.

After warping, the app re-detects beats on the warped mix as a check. Green
beat lines fall on the grid (within 30 ms); red ones do not. Click any beat
line to see the summary. Listen with the metronome; if it drifts, return to
the original mix and recount (below). Stems, chord detection and stem import
unlock as soon as the mix is warped.

To start over, **Edit > Return to original mix...** discards the warped copy.
Bar and beat positions of existing notes are kept, but analysis of stems that
came from the old warp becomes stale (marked **(stale)** in the track list), so
re-separate or re-import them.

## 5. Get stems

Either:

- **Edit > Separate stems...** runs Demucs `htdemucs_6s` on the warped mix in
  the background (a few minutes for a full song on a CPU). The first time, the
  app installs Demucs and downloads its model (about 80 MB) automatically.
  Existing tracks of the same role keep their edits and get the new audio. Or:
- **File > Import stems...** and choose stems that were separated from the
  **warped** mix (same length and sample rate). The role is guessed from the
  file name, and you can change it.

The roles are `bass`, `vocals`, `piano`, `guitar`, `drums`, and `other`; one
track per role. `other` is harmony context only and produces no notes.

## 6. Transcribe and correct notes

Choose a stem in **Track** and its **Model** next to it:

| Role | Models |
|---|---|
| Bass | `crepe_tiny`, `crepe_full` |
| Vocals | `pyin`, `crepe_full`, `crepe_tiny` |
| Piano / guitar | `basic_pitch`, `piano_hr` |
| Drums | `adtof` |

With **Model > Reanalyze after edits** checked (the default), the visible bars
are analyzed automatically. **Model > Analyze visible bars** and
**Analyze whole track** run it on demand; **Cancel analysis** (Esc, or the
button in the status bar) stops it.

**Navigate** with **Previous** / **Next** (PgUp / PgDn). The label between them
shows the visible bars. **Zoom out (-)** / **Zoom in (+)** next to them (also
Ctrl + mouse wheel, View > Zoom, Ctrl+- / Ctrl+=) show 1, 2, 4, 8 or 16 bars per
screen; the readout beside the buttons shows the current setting. The plain mouse wheel scrolls pitch; **Ctrl + Shift + wheel**
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

Velocity comes from how loud the note is in its stem: the power at the note's
pitch and its first harmonics over the note's first 200 ms, compared with the
loudest notes of the whole stem (General MIDI curve, so 12 dB quieter is about
half the velocity). This applies to bass, vocals and Basic Pitch; the piano_hr
model predicts piano key velocities itself. Drum velocity is the RMS level at
each hit in that kit piece's band (kick 30-150 Hz, snare 150 Hz-5 kHz, toms
60-600 Hz, hi-hat 6-16 kHz, cymbals 4-16 kHz), so a kick and a hi-hat on the
same beat are measured separately; each piece is scaled to its own loudest
hits. Notes you draw start at 100.

There are two tools, with buttons next to **Grid** (or **View** menu):

- **Brush (B)**, the default, adds, resizes and deletes notes:

| Action | Result |
|---|---|
| Click an empty cell | Add a note one beat long |
| Drag from an empty cell | Add a note and set its length |
| Click a note | Select it |
| Drag a note's right edge | Change its length |
| Right-click a note, or **Delete** | Reject it (that pitch stays silent there) |
| Right-click empty space | *Rest for this beat* / *Rest for this bar* (silence all pitches) |
| Click the onset lane | Add an onset (note-attack boundary) |
| Right-click the onset lane | Remove the nearest onset |

- **Info (I)**: click a note to open a small editor with its bar, beat, length
  in beats, pitch and velocity, plus **Delete note**. Beats accept `2 1/2`,
  `5/2` or `2.5`; pitches accept `Eb4` or `63`. Clicking empty space does
  nothing, so you can inspect notes without adding any. Right-click still
  deletes.

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

## 7. Chords

Choose **Chords** in the **Track** selector. From the top down, the view shows
the mix amplitude, the chord lane, and a 12-row chroma (one row per pitch
class, bright where that note sounds). The chord's tones are outlined in the
chroma, and bar/beat lines run through every lane.

- **Edit > Detect chords** proposes chords on the beat grid
  (triads, sevenths, suspended and slash chords). It requires a warped mix.
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
- **File > Import chord MIDI...** turns a MIDI file of block chords into
  symbols.

Undo and redo work in the Chords view too. Playback adds voiced chords.

## 8. Listen

**Play** (or Space) plays from the start of the visible window; **Stop** ends
it, and the view follows the playhead. In a stem view you hear the stem mostly
on the right and the notes mostly on the left; in the Chords view you hear the
mix with the chords.

**File > Playback settings...** chooses how notes sound:

- **Built-in tones** (default) are rendered into the same buffer as the audio,
  so they are sample-aligned. Use them to judge timing.
- **MIDI out** sends notes to a synthesizer. Windows delays MIDI differently
  from audio; use **MIDI sync test** and adjust **MIDI latency** until the click
  and the wood block sound together. The setting is stored on this computer.

## 9. Export

- **File > Export MIDI...** (Ctrl+E): choose all tracks plus chords, a single
  role, or chords only. If tracks are not fully analyzed, the app asks before
  writing a partial file.
- **File > Export lead sheet (MusicXML)...** writes the chord symbols on bars,
  ready for MuseScore.

## Troubleshooting

| Symptom | What to check |
|---|---|
| A menu item is grayed out | Hover over it: the tooltip says which stage is missing (mix, counted bars, or warp). |
| The count is at double or half speed | Pin two downbeats several bars apart; the count follows your pins. |
| Tapping did not count bars | Tap the first beat of each bar (not every beat), for at least four bars, in a section with clear drums or bass. Or pin downbeats by hand. |
| Clicks drift late in the song | Pin another downbeat near the drift, then warp with stretching instead of as-is. |
| Many red beats after warping | Return to the original mix, add pins where it drifts, and warp again. |
| A track is marked (stale) | Its audio came from a different warp of the mix. Re-separate or re-import it. |
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
