# Audio to MIDI

A machine-learning-assisted audio-to-MIDI project editor. Import a mix, count
bars with pinned downbeats (the audio is never stretched), separate stems,
correct notes and chords, then export MIDI or a MuseScore-ready MusicXML score.

The refactored project editor is now the default and only supported app.
Both `python app.py` and `python -m audio_to_midi` open it.
See [Getting started](docs/getting-started.md) for installation and the user
manual. The former single-stem app has been retired; shared transcription
providers and compatible onset-profile import remain.

Fundamental constraint: models must remain lightweight enough to learn from
each user edit and update the visible notes interactively. Whole-file analysis
can run in the background.

The reviewed per-stem rankings, independent cancellable YourMT3+ worker,
explicit mix/stem sources and disposable inference-evidence cache are described
in the [model workflow](docs/getting-started.md#5-transcribe-and-correct-notes).
Projects save in schema 4 and can still open schema 3 projects.
The AI-notes slider filters saved predictions without rerunning inference;
native backend threshold replay remains separate from this display filter.
