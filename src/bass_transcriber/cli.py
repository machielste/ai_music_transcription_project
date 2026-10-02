"""Command-line interface for bass-transcriber."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from bass_transcriber import __version__
from bass_transcriber.diagnostics import collect_diagnostics, has_errors
from bass_transcriber.export.gp5 import write_gp5
from bass_transcriber.export.json import read_notes_json, write_notes_json
from bass_transcriber.export.midi import write_midi
from bass_transcriber.export.rhythm_json import read_rhythm_json, write_rhythm_json
from bass_transcriber.fingering_debug import (
    DEFAULT_DEBUG_PROFILES,
    serve_fingering_debugger,
)
from bass_transcriber.models import BassInstrument
from bass_transcriber.rhythm import detect_rhythm
from bass_transcriber.separation import separate_bass
from bass_transcriber.tab import FINGERING_PROFILES
from bass_transcriber.transcription.muscriptor import (
    MODEL_SIZES,
    ModelFetchError,
    ModelSize,
    fetch_model,
    transcribe_bass,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level command-line parser."""
    parser = argparse.ArgumentParser(
        prog="bass-transcriber",
        description="Transcribe songs into synchronized bass tablature for ToneLib Jam.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "doctor",
        help="check local audio, CUDA, and Hugging Face prerequisites",
    )

    model_parser = subparsers.add_parser("model", help="manage transcription models")
    model_subparsers = model_parser.add_subparsers(dest="model_command", required=True)
    fetch_parser = model_subparsers.add_parser(
        "fetch",
        help="download a MuScriptor checkpoint into the shared Hugging Face cache",
    )
    fetch_parser.add_argument("size", choices=MODEL_SIZES)

    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="run a bass-conditioned MuScriptor transcription",
    )
    transcribe_parser.add_argument("audio", type=Path)
    transcribe_parser.add_argument("--output", "-o", type=Path)
    transcribe_parser.add_argument("--model", choices=MODEL_SIZES, default="small")
    transcribe_parser.add_argument(
        "--instrument",
        choices=("auto", "electric_bass", "acoustic_bass"),
        default="electric_bass",
        help="use 'auto' to classify all instruments, then retain labelled bass events",
    )
    transcribe_parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    transcribe_parser.add_argument(
        "--separate-bass", action="store_true",
        help="separate and retain a bass WAV before electric-bass-conditioned transcription",
    )

    export_parser = subparsers.add_parser("export", help="export a transcription artifact")
    export_subparsers = export_parser.add_subparsers(dest="export_format", required=True)
    midi_parser = export_subparsers.add_parser(
        "midi",
        help="convert a raw notes JSON sidecar to a Standard MIDI File",
    )
    midi_parser.add_argument("notes", type=Path)
    midi_parser.add_argument("--output", "-o", type=Path)

    gp5_parser = export_subparsers.add_parser(
        "gp5",
        help="write a four- or five-string bass Guitar Pro 5 score",
    )
    gp5_parser.add_argument("notes", type=Path)
    gp5_parser.add_argument("--rhythm", type=Path, required=True)
    gp5_parser.add_argument("--output", "-o", type=Path)
    gp5_parser.add_argument("--title")
    gp5_parser.add_argument(
        "--strings",
        type=int,
        choices=(4, 5),
        default=5,
        help="bass string count and tuning: 4=EADG, 5=BEADG",
    )
    gp5_parser.add_argument(
        "--fingering-profile",
        choices=(*FINGERING_PROFILES, "legacy"),
        default="balanced",
        help="hand-position/open-string policy, or 'legacy' to disable optimization",
    )

    debug_parser = subparsers.add_parser("debug", help="open interactive debugging tools")
    debug_subparsers = debug_parser.add_subparsers(dest="debug_command", required=True)
    fingering_debug_parser = debug_subparsers.add_parser(
        "fingerings",
        help="compare fingering strategies in synchronized perspective note highways",
    )
    fingering_debug_parser.add_argument("notes", type=Path)
    fingering_debug_parser.add_argument("--rhythm", type=Path, required=True)
    fingering_debug_parser.add_argument(
        "--audio",
        type=Path,
        help="source audio; defaults to the source recorded in the notes JSON",
    )
    fingering_debug_parser.add_argument(
        "--profiles",
        nargs="+",
        choices=(*FINGERING_PROFILES, "legacy"),
        default=DEFAULT_DEBUG_PROFILES,
        help="strategies to show, in reference/comparison order",
    )
    fingering_debug_parser.add_argument(
        "--strings",
        type=int,
        choices=(4, 5),
        default=5,
        help="bass string count and tuning: 4=EADG, 5=BEADG",
    )
    fingering_debug_parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="localhost port; zero selects an available port",
    )
    fingering_debug_parser.add_argument(
        "--no-open",
        action="store_true",
        help="serve the debugger without opening the default browser",
    )

    rhythm_parser = subparsers.add_parser("rhythm", help="analyze musical timing")
    rhythm_subparsers = rhythm_parser.add_subparsers(dest="rhythm_command", required=True)
    detect_parser = rhythm_subparsers.add_parser(
        "detect",
        help="detect tempo, meter, downbeats, and transcription onset lag",
    )
    detect_parser.add_argument("notes", type=Path)
    detect_parser.add_argument("--output", "-o", type=Path)
    detect_parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    detect_parser.add_argument(
        "--detector",
        choices=("auto", "beat_this", "librosa"),
        default="auto",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        checks = collect_diagnostics()
        for check in checks:
            print(f"[{check.status.upper():7}] {check.name}: {check.detail}")
        return 1 if has_errors(checks) else 0

    if args.command == "model" and args.model_command == "fetch":
        try:
            files = fetch_model(cast(ModelSize, args.size))
        except ModelFetchError as error:
            parser.error(str(error))
        print(f"MuScriptor {files.size} is cached and ready:")
        print(f"  weights: {files.weights}")
        print(f"  config:  {files.config}")
        return 0

    if args.command == "transcribe":
        audio = cast(Path, args.audio)
        if not audio.is_file():
            parser.error(f"audio file does not exist: {audio}")
        output = cast(Path | None, args.output) or audio.with_suffix(".notes.json")

        def show_progress(completed: int, total: int) -> None:
            print(f"Transcribing chunk {completed}/{total}", file=sys.stderr)

        instrument_mode = cast(str, args.instrument)
        transcription_audio = audio
        separation_metadata = None
        if args.separate_bass:
            print("Separating bass with BS-RoFormer SW", file=sys.stderr)
            transcription_audio = output.with_name(
                output.name.removesuffix(".notes.json") + ".bass.stem.wav"
            )
            try:
                separation_metadata = separate_bass(
                    audio, transcription_audio, device=cast(str, args.device),
                )
            except (RuntimeError, OSError) as error:
                parser.error(str(error))
            instrument_mode = "electric_bass"
        instrument = None if instrument_mode == "auto" else cast(BassInstrument, instrument_mode)
        notes = transcribe_bass(
            transcription_audio,
            size=cast(ModelSize, args.model),
            instrument=instrument,
            device=cast(str, args.device),
            progress=show_progress,
        )
        write_notes_json(
            output,
            notes,
            source=audio,
            model_size=cast(str, args.model),
            instrument_mode=instrument_mode,
            transcription_audio=transcription_audio if args.separate_bass else None,
            separation=separation_metadata,
        )
        print(f"Wrote {len(notes)} notes to {output}")
        return 0

    if args.command == "export" and args.export_format == "midi":
        notes_path = cast(Path, args.notes)
        if not notes_path.is_file():
            parser.error(f"notes JSON does not exist: {notes_path}")
        midi_output = cast(Path | None, args.output)
        if midi_output is None:
            base_name = notes_path.name.removesuffix(".notes.json")
            midi_output = notes_path.with_name(f"{base_name}.mid")
        try:
            document = read_notes_json(notes_path)
            write_midi(midi_output, document.notes)
        except ValueError as error:
            parser.error(str(error))
        print(f"Wrote {len(document.notes)} notes to {midi_output}")
        return 0

    if args.command == "export" and args.export_format == "gp5":
        notes_path = cast(Path, args.notes)
        rhythm_path = cast(Path, args.rhythm)
        if not notes_path.is_file():
            parser.error(f"notes JSON does not exist: {notes_path}")
        if not rhythm_path.is_file():
            parser.error(f"rhythm JSON does not exist: {rhythm_path}")
        gp5_output = cast(Path | None, args.output)
        if gp5_output is None:
            base_name = notes_path.name.removesuffix(".notes.json")
            gp5_output = notes_path.with_name(f"{base_name}.gp5")
        notes_document = read_notes_json(notes_path)
        rhythm_grid = read_rhythm_json(rhythm_path)
        title = cast(str | None, args.title) or f"{notes_document.source.stem} - Bass"
        try:
            export_result = write_gp5(
                gp5_output,
                notes_document.notes,
                rhythm_grid,
                title=title,
                fingering_profile=(
                    None if args.fingering_profile == "legacy" else args.fingering_profile
                ),
                five_string=args.strings == 5,
            )
        except ValueError as error:
            parser.error(str(error))
        print(
            f"Wrote {args.strings}-string {'BEADG' if args.strings == 5 else 'EADG'} "
            f"GP5 using an integer tempo schedule "
            f"for {rhythm_grid.bpm:.3f} BPM to {gp5_output}"
        )
        if export_result.dropped_pitches:
            print(
                f"Warning: dropped {len(export_result.dropped_pitches)} notes outside "
                "the BEADG range",
                file=sys.stderr,
            )
        return 0

    if args.command == "debug" and args.debug_command == "fingerings":
        notes_path = cast(Path, args.notes)
        rhythm_path = cast(Path, args.rhythm)
        if not notes_path.is_file():
            parser.error(f"notes JSON does not exist: {notes_path}")
        if not rhythm_path.is_file():
            parser.error(f"rhythm JSON does not exist: {rhythm_path}")
        if not 0 <= args.port <= 65535:
            parser.error("port must be between 0 and 65535")
        try:
            notes_document = read_notes_json(notes_path)
            rhythm_grid = read_rhythm_json(rhythm_path)
            audio = cast(Path | None, args.audio) or notes_document.source
            if not audio.is_file():
                parser.error(f"source audio does not exist: {audio}")
            serve_fingering_debugger(
                audio,
                notes_document.notes,
                rhythm_grid,
                profiles=cast(Sequence[str], args.profiles),
                five_string=args.strings == 5,
                port=cast(int, args.port),
                open_browser=not cast(bool, args.no_open),
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
        return 0

    if args.command == "rhythm" and args.rhythm_command == "detect":
        notes_path = cast(Path, args.notes)
        if not notes_path.is_file():
            parser.error(f"notes JSON does not exist: {notes_path}")
        document = read_notes_json(notes_path)
        if not document.source.is_file():
            parser.error(f"source audio does not exist: {document.source}")
        rhythm_output = cast(Path | None, args.output)
        if rhythm_output is None:
            base_name = notes_path.name.removesuffix(".notes.json")
            rhythm_output = notes_path.with_name(f"{base_name}.rhythm.json")
        print(f"Detecting rhythm from {document.source} ...", file=sys.stderr)
        grid = detect_rhythm(
            document.source,
            note_onsets=[note.start_seconds for note in document.notes],
            device=cast(str, args.device),
            detector=cast(str, args.detector),  # type: ignore[arg-type]
        )
        write_rhythm_json(rhythm_output, grid, source=document.source)
        meter = f"{grid.beats_per_bar}/4" if grid.beats_per_bar else "unknown"
        first_downbeat = (
            f"{grid.first_downbeat_seconds:.3f}s"
            if grid.first_downbeat_seconds is not None
            else "unknown"
        )
        print(
            f"Detected {grid.bpm:.3f} BPM with {grid.detector}, meter {meter}, "
            f"first downbeat {first_downbeat}, "
            f"onset delay {grid.onset_delay_seconds * 1000:+.1f}ms"
        )
        print(f"Wrote {len(grid.beat_times_seconds)} beats to {rhythm_output}")
        return 0

    parser.print_help()
    return 0
