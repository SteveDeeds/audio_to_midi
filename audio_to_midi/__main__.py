"""Run the v2 GUI, or use its headless project commands."""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Audio to MIDI v2 preview (legacy: python app.py)")
    commands = root.add_subparsers(dest="command")
    commands.add_parser("doctor", help="Inspect environment without installing or downloading models")
    create = commands.add_parser("create", help="Create an externally warped project")
    create.add_argument("folder", type=Path)
    create.add_argument("--name", default="Untitled")
    create.add_argument("--bpm", type=float, required=True, help="Quarter notes per minute")
    create.add_argument("--meter", default="4/4")
    create.add_argument("--grid", default="1/4", help="Quarter-note units; 1/6 = sixteenth triplets")
    create.add_argument("--offset", type=float, default=0, help="Shared downbeat in audio seconds")
    create.add_argument("--key", default="")
    create.add_argument("--mix", type=Path)
    stem = commands.add_parser("add-stem", help="Import a role-assigned, already-warped stem")
    stem.add_argument("folder", type=Path)
    stem.add_argument("role", choices=["bass", "vocals", "piano", "guitar", "drums", "other"])
    stem.add_argument("audio", type=Path)
    stem.add_argument("--model")
    midi = commands.add_parser("import-midi", help="Import starting MIDI into a stem or chords")
    midi.add_argument("folder", type=Path)
    midi.add_argument("role")
    midi.add_argument("midi", type=Path)
    analyze = commands.add_parser("transcribe", help="Transcribe a whole stem with an approved backend")
    analyze.add_argument("folder", type=Path)
    analyze.add_argument("role")
    analyze.add_argument("--start", type=Fraction, default=Fraction(0))
    analyze.add_argument("--end", type=Fraction)
    analyze.add_argument("--accept-noncommercial", action="store_true", help="Acknowledge ADTOF non-commercial weight restriction")
    analyze.add_argument("--accept-checkpoint-terms", action="store_true", help="Acknowledge piano-HR terms and possible checkpoint download")
    commands.add_parser("gui", help="Open the project editor").add_argument("folder", type=Path, nargs="?")
    commands.add_parser("info").add_argument("folder", type=Path)
    chords = commands.add_parser("chords", help="Detect chords from the stems or the mix")
    chords.add_argument("folder", type=Path)
    chords.add_argument("--model", choices=["template", "btc"], default="template",
                        help="template: built-in chord templates; btc: BTC transformer (downloads ~12 MB once)")
    export = commands.add_parser("export", help="Export resolved tracks plus chords with tempo and meter")
    export.add_argument("folder", type=Path)
    export.add_argument("output", type=Path)
    export.add_argument("--allow-partial", action="store_true")
    export.add_argument("--role", choices=["bass", "vocals", "piano", "guitar", "drums", "chords"], help="Export one stem or just the chord track; default is combined MIDI")
    prepare = commands.add_parser("prepare", help="Inspect/experimentally warp an ORIGINAL full mix")
    prepare.add_argument("mix", type=Path)
    prepare.add_argument("artifacts", type=Path)
    prepare.add_argument("--bpm", type=float)
    prepare.add_argument("--meter")
    prepare.add_argument("--downbeat", type=float)
    prepare.add_argument("--anchors", type=Path, help="JSON pairs [source_seconds, quarter_position]")
    prepare.add_argument("--lead-in", choices=["trim", "pad"], help="Required to render; otherwise inspect only")
    prepare.add_argument("--new-revision", action="store_true", help="Preserve old outputs and explicitly invalidate descendants")
    register = commands.add_parser("register-warped", help="Register an EXTERNAL warped mix verbatim, without re-warping")
    register.add_argument("mix", type=Path)
    register.add_argument("artifacts", type=Path)
    register.add_argument("--bpm", type=float, required=True)
    register.add_argument("--meter", default="4/4")
    register.add_argument("--offset", type=float, default=0)
    register.add_argument("--new-revision", action="store_true")
    attach = commands.add_parser("attach-prepared", help="Attach a prepared mix to an EMPTY project")
    attach.add_argument("folder", type=Path)
    attach.add_argument("artifacts", type=Path)
    separate = commands.add_parser(
        "separate", help="Demucs htdemucs_6s separation of a prepared full mix (installs/downloads on first use)",
    )
    separate.add_argument("artifacts", type=Path)
    separate.add_argument("--new-revision", action="store_true")
    separate.add_argument("--project", type=Path, help="Import all six outputs into an empty project")
    return root


def _track_for_role(project, role):
    track = next((t for t in project.tracks if t.role == role), None)
    if track is None:
        raise ValueError(f"Project has no {role} stem.")
    return track


def main(argv: list[str] | None = None) -> int:
    root = parser()
    args = root.parse_args(argv)
    try:
        if args.command == "doctor":
            import json
            from audio_to_midi.environment import inspect_environment

            report = inspect_environment()
            print(json.dumps(report, indent=2))
            return 0 if report["base_ready"] else 1
        if args.command in (None, "gui"):
            import sys
            from PySide6.QtWidgets import QApplication
            from audio_to_midi.ui import ProjectWindow
            from audio_to_midi.spectrogram import warm_up
            from audio_to_midi.storage import load_project

            warm_up()
            app = QApplication(sys.argv[:1])
            app.setApplicationName("Audio to MIDI v2 preview")
            window = ProjectWindow()
            if getattr(args, "folder", None):
                window.project = load_project(args.folder)
                window.folder = args.folder
                window._reset_state()
            window.show()
            return app.exec()
        from audio_to_midi.domain import Project
        from audio_to_midi.timebase import Grid
        from audio_to_midi.storage import load_project, save_project
        from audio_to_midi.application import (
            add_stem, analyze_harmony, attach_prepared, complete, publish, token, track_end,
            TranscriptionService,
        )
        from audio_to_midi.adapters import MODELS, normalize_key
        from audio_to_midi.midi import export_midi, import_notes

        if args.command in ("prepare", "register-warped", "separate"):
            import json
            import sys
            from audio_to_midi.pipeline import ArtifactStore, prepare_mix, register_warped_mix, separate_mix

            if args.command == "register-warped":
                registered = register_warped_mix(
                    args.mix, args.artifacts, bpm=args.bpm,
                    meter=tuple(int(x) for x in args.meter.split("/")),
                    offset_seconds=args.offset, force_new_revision=args.new_revision,
                )
                print(f"External mix registered verbatim: {registered['warp_revision']['id']}")
            elif args.command == "prepare":
                anchors = json.loads(args.anchors.read_text(encoding="utf-8")) if args.anchors else []
                meter = tuple(int(x) for x in args.meter.split("/")) if args.meter else None
                prepared = prepare_mix(
                    args.mix, args.artifacts, bpm=args.bpm, meter=meter,
                    downbeat_seconds=args.downbeat, beat_anchors=anchors, lead_in=args.lead_in,
                    force_new_revision=args.new_revision,
                )
                print(f"Analysis revision: {prepared['analysis_revision']['id']}")
                if prepared["warp_revision"]:
                    print(f"EXPERIMENTAL warp revision: {prepared['warp_revision']['id']}; audition before use")
            else:
                if args.project:
                    project = load_project(args.project)
                    if project.tracks or project.chords:
                        raise ValueError("Import separation into an empty project to preserve existing edits.")
                store = ArtifactStore(args.artifacts)
                warp = store.manifest["active"].get("warp")
                if not warp:
                    raise ValueError("No prepared warp revision. Run prepare first.")
                revision = separate_mix(
                    store, warp, force_new_revision=args.new_revision, progress=print,
                )
                if args.project:
                    attach_prepared(project, args.artifacts)
                    for role in MODELS:
                        add_stem(project, store.output(revision, role), role)
                    project.stages["preparation"]["separation_revision"] = revision["id"]
                    save_project(project, args.project)
                print(f"Separation revision: {revision['id']}")
            return 0
        if args.command == "create":
            if (args.folder / "project.json").exists():
                raise ValueError("A project already exists in that folder.")
            numerator, denominator = [int(value) for value in args.meter.split("/")]
            project = Project.create(args.name, Grid(
                args.bpm, numerator, denominator, Fraction(args.grid), args.offset,
            ))
            project.key = normalize_key(args.key)
            if args.mix:
                import soundfile as sf
                sf.info(args.mix)
                project.mix_path = str(args.mix.resolve(strict=True))
            save_project(project, args.folder)
            print(f"Created {project.name}")
            return 0
        project = load_project(args.folder)
        if args.command == "attach-prepared":
            attach_prepared(project, args.artifacts)
            save_project(project, args.folder)
        elif args.command == "add-stem":
            track = add_stem(project, args.audio, args.role)
            if args.model:
                if args.model not in MODELS[args.role]:
                    raise ValueError(f"Invalid {args.role} model: {args.model}")
                track.model = args.model
            save_project(project, args.folder)
        elif args.command == "import-midi":
            from audio_to_midi.application import mark_project_edited
            mark_project_edited(project)
            if args.role == "chords":
                from core.midi_export import load_midi_notes
                from audio_to_midi.chords import notes_to_segments
                from audio_to_midi.domain import Note

                notes = [
                    Note(f"chord-import:{i}", Fraction(str(a * project.grid.bpm / 60)).limit_denominator(1000000),
                         Fraction(str(b * project.grid.bpm / 60)).limit_denominator(1000000), pitch, velocity, "imported")
                    for i, (a, b, pitch, velocity, _) in enumerate(load_midi_notes(str(args.midi)))
                ]
                project.chords = notes_to_segments(notes, project.grid)
            else:
                track = _track_for_role(project, args.role)
                import_notes(args.midi, project, track)
            save_project(project, args.folder)
        elif args.command == "transcribe":
            track = _track_for_role(project, args.role)
            if track.role == "drums" and not args.accept_noncommercial:
                raise ValueError("ADTOF weights are non-commercial. Pass --accept-noncommercial to acknowledge.")
            if track.model == "piano_hr" and not args.accept_checkpoint_terms:
                raise ValueError("Review piano-HR checkpoint terms, then use --accept-checkpoint-terms to allow loading/downloading.")
            if args.accept_noncommercial or args.accept_checkpoint_terms:
                track.settings["license_acknowledged"] = True
            from audio_to_midi import provision

            provision.ensure(track.model, progress=print)
            end = args.end if args.end is not None else track_end(project, track)
            result_token = token(project, track)
            result = TranscriptionService().predict(project, track, args.start, end)
            if not publish(project, result_token, result):
                raise RuntimeError("Prediction became stale before publication.")
            save_project(project, args.folder)
            print(f"Analyzed {track.role}: {len(result.notes)} proposals; edits preserved")
        elif args.command == "chords":
            if project.chords:
                raise ValueError("Chord data already exists. Preserve it before requesting a replacement.")
            if args.model == "btc":
                from audio_to_midi import provision

                provision.ensure("btc", print)
            project.key, project.chords = analyze_harmony(project, model=args.model)
            save_project(project, args.folder)
        elif args.command == "export":
            selected = project.tracks if args.role is None else (
                [] if args.role == "chords" else [_track_for_role(project, args.role)]
            )
            if args.role == "chords" and not project.chords:
                raise ValueError("Project has no chords.")
            if args.output.suffix.lower() in (".musicxml", ".xml"):
                if args.role not in (None, "chords"):
                    raise ValueError("MusicXML export is a chord lead sheet; use --role chords or omit --role.")
                from audio_to_midi.midi import export_musicxml
                export_musicxml(project, args.output)
                print(f"Lead sheet: {args.output}")
                return 0
            incomplete = [t.role for t in selected if not complete(project, t)]
            if incomplete and not args.allow_partial:
                raise ValueError(f"Unprocessed or stale tracks: {', '.join(incomplete)}. Transcribe or use --allow-partial.")
            export_midi(project, args.output, selected, include_chords=args.role in (None, "chords"))
            print(f"{'PARTIAL' if incomplete else 'Complete'} MIDI: {args.output}")
        elif args.command == "info":
            print(f"{project.name}: {project.grid.bpm} quarter BPM, {project.grid.numerator}/{project.grid.denominator}")
            for track in project.tracks:
                print(f"  {track.role}: {track.model or 'harmony only'}, {len(track.notes.resolved())} notes, "
                      f"{'complete' if complete(project, track) else 'partial/unprocessed'}")
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, StopIteration) as exc:
        root.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
