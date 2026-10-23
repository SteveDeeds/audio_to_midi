# Getting started with Audio to MIDI v2

Audio to MIDI turns a song into editable MIDI. You work through stages, each
shown in the strip at the top right of the window:

**Mix → Count bars → Warp → Confirm → Stems → Notes → Chords → Export**

1. Import the full mix.
2. Find its bars and beats (detect, then fix with pins or tapped downbeats).
3. Warp the mix to one steady tempo and confirm the timing.
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
.\install-v2.ps1
.\.venv-v2\Scripts\python.exe -X utf8 -m audio_to_midi
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
.\install-v2.ps1 -Models
```

If the existing development environment is already installed, launch with:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m audio_to_midi
```

Do not use `python app.py` for this guide: that opens the legacy application.

## 2. Create a project and import the mix

1. **File > New project...** asks for a name, then a parent folder. The project
   is saved in a new folder named after the project; the warped mix and
   separated stems are stored there as well.
2. **File > Import mix...** and choose the full song (WAV or FLAC works best).
   Every project needs a mix; it is the timing reference for everything else.
3. The app immediately runs **Edit > Detect tempo, meter, and key** on the mix.

Projects reference audio files rather than copying the original mix, so keep
it where it is. Save regularly with **File > Save** (Ctrl+S); closing the
window asks before discarding unsaved changes.

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

## 4. Warp and confirm the timing

**Edit > Warp mix to tempo...** writes a constant-tempo copy of the mix into
the project folder, so the bar grid lines up with the music everywhere.

- **Use as-is** skips stretching. It is preselected when the mix already has a
  steady tempo (for example, it was produced to a click or warped in a DAW).
- Otherwise choose the **Target tempo**, how the **Anchors** follow your beats,
  and whether to keep or trim the audio before bar 1.

After warping, the app re-detects beats on the warped mix as a check. Green
beat lines fall on the grid (within 30 ms); red ones do not. Click any beat
line to see the summary. Listen with the metronome, then check
**Edit > Timing confirmed**. Stems, chord detection and stem import stay
grayed out until the timing is confirmed.

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
shows the visible bars. **Ctrl + mouse wheel** (or View > Zoom) shows more or
fewer bars; the plain mouse wheel scrolls pitch; **Ctrl + Shift + wheel**
zooms pitch (not in the drums view). **Grid** sets the editing step (1/4 to 1/32, including
triplets).

The piano roll shows the stem's spectrogram behind the notes: a sustained
pitch is a bright horizontal band. Blue notes are model proposals; green ones
are yours. **Hover over a note** to see its name, the bar and beat it starts
on, its length and velocity.

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

- **Edit > Detect chords** proposes chords on the beat grid from the warped mix
  (triads, sevenths, suspended and slash chords). It requires confirmed timing.
  The current recognizer is a simple placeholder; a better lead-sheet
  recognizer is being evaluated.
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
| A menu item is grayed out | Hover over it: the tooltip says which stage is missing (mix, counted bars, warp, or confirmed timing). |
| The count is at double or half speed | Pin two downbeats several bars apart; the count follows your pins. |
| Tapping did not count bars | Tap the first beat of each bar (not every beat), for at least four bars, in a section with clear drums or bass. Or pin downbeats by hand. |
| Clicks drift late in the song | Pin another downbeat near the drift, then warp with stretching instead of as-is. |
| Many red beats after warping | Return to the original mix, add pins where it drifts, and warp again. |
| A track is marked (stale) | Its audio came from a different warp of the mix. Re-separate or re-import it. |
| No notes appear | Check the model selector and the status bar for errors. `other` never produces notes. |
| Model install fails | First-use installs need an internet connection and write access to the app's Python environment (use a virtual environment rather than a system-wide Python). ADTOF also needs `git`. Or pre-install with `.\install-v2.ps1 -Models`. For a base-only trial, use vocals with `pyin`. |
| Notes sound late with a MIDI device | Run **MIDI sync test** in **File > Playback settings** and raise the latency. |
| Reopening reports missing media | Restore referenced files to their expected locations; projects do not embed the original mix. |

For an environment report, run:

```powershell
.\.venv-v2\Scripts\python.exe -X utf8 -m audio_to_midi doctor
```

For architecture and validation notes, see the
[execution plan](execution-plan.md); for the interface design, see the
[UI redesign plan](ui-redesign-plan.md).
