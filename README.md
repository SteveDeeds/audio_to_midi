This readme file serves only as high level guideance.
There will be no implementation details in this file.
When editing this file, keep it short.

App Goal:
A machine learning assisted audio to MIDI converter.
It focus on getting human input specific to the song being worked on to train a simple model to reconize chord changes, note onsets, and pitches.

A side-by-side v2 project editor is available as a preview; the existing app
remains supported. See [Getting started with the v2 GUI](docs/getting-started.md)
for installation and a step-by-step user manual, or the
[execution plan](docs/execution-plan.md) for scope and remaining acceptance gates.

Fundamental constraint:
Models must remain lightweight enough to learn from each user edit and rerun inference on the visible notes in real time. Analysis and inference for the rest of the file can run in the background, but the edit-to-training-to-screen-update loop must stay interactive.

The Mode selector also offers Bass (CREPE full/tiny + onset + key) and Vocals
(pYIN/CREPE full + key). Install requirements.txt in the app's environment for
CREPE support. Selecting a model applies it automatically to the visible region.
The trained onset model still owns note boundaries on the usual grid; precise
attacks from the stem only help it decide which grid onsets to keep, and the
pretrained tracker only supplies one pitch per onset-to-onset block (vocals also
split on held pitch changes). Deleting an onset merges the notes, a note ends
early on the grid when the pitch fades, and releases stop at the view's right
edge. User onsets, pitches and rests are kept. Analysis runs in the background
in 2 s batches and revisited audio is cached. Edits do not retrain the neural
tracker. CREPE full is slow on CPU (about 0.16x realtime), so notes fill in
progressively. Key context uses recognised sibling stems, with Other used only
as harmony context. Standard and Percussive remain available; there is no new
Other transcription option.

Piano and Guitar modes offer Basic Pitch and the high-resolution piano model.
They follow the same onset grid but keep up to the polyphony slider's number of
notes per block; a held note carries across new onsets until the model ends it.
Basic Pitch is fast; the piano model runs about realtime on CPU. Install the
extra packages listed in requirements.txt (Basic Pitch needs `--no-deps`).

Percussive: ADTOF runs the ADTOF drum CRNN (Zehren et al.; weights are
CC BY-NC-SA, non-commercial) once per drum stem in the background. Edits adapt
per-class hit thresholds and teach relabels (e.g. tom → snare) for similar hits.

This app acts as a bridge between Ableton live (with it's profesional warping capabilites to create exact on beat tracks) and Unreal Vocal Remover to produce stems of the parts.  This is the input to the App.  The app then produces midi that is imported into Muse Score