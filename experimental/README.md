# Experimental Scripts

This folder contains analysis and diagnostics utilities used to investigate model behavior.

These scripts are not imported by the production application and are not required for normal app use.

## Purpose

- Analyze profile quality and feature importance.
- Visualize training distributions.
- Diagnose specific issues (for example, percussive spike behavior).

## Usage

Run scripts directly from the repository root, for example:

```powershell
c:/Users/steve/OneDrive/Documents/GitHub/audio_to_midi/.venv/Scripts/python.exe experimental/analyze_model_profiles.py --help
```

## Notes

- Script outputs and reports are intended for manual inspection.
- Keep production code changes in core/, ui/, and workers/.
