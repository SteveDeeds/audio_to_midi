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
