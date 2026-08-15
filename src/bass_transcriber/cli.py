"""Command-line interface for bass-transcriber."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from bass_transcriber import __version__
from bass_transcriber.diagnostics import collect_diagnostics, has_errors
from bass_transcriber.export.json import read_notes_json, write_notes_json
from bass_transcriber.export.midi import write_midi
from bass_transcriber.models import BassInstrument
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

    export_parser = subparsers.add_parser("export", help="export a transcription artifact")
    export_subparsers = export_parser.add_subparsers(dest="export_format", required=True)
    midi_parser = export_subparsers.add_parser(
        "midi",
        help="convert a raw notes JSON sidecar to a Standard MIDI File",
    )
    midi_parser.add_argument("notes", type=Path)
    midi_parser.add_argument("--output", "-o", type=Path)
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
        instrument = None if instrument_mode == "auto" else cast(BassInstrument, instrument_mode)
        notes = transcribe_bass(
            audio,
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

    parser.print_help()
    return 0
