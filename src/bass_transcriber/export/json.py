"""Lossless JSON sidecar export for raw transcription results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from bass_transcriber.models import BassInstrument, BassNote


@dataclass(frozen=True, slots=True)
class NotesDocument:
    """Raw transcription loaded from a JSON sidecar."""

    source: Path
    model_size: str
    notes: list[BassNote]


def write_notes_json(
    output: Path,
    notes: list[BassNote],
    *,
    source: Path,
    model_size: str,
    instrument_mode: str = "unspecified",
) -> None:
    """Write raw notes and provenance in a stable, inspectable format."""
    document = {
        "schema_version": 1,
        "source": str(source.resolve()),
        "transcriber": {
            "name": "muscriptor",
            "model_size": model_size,
            "instrument_mode": instrument_mode,
        },
        "notes": [asdict(note) for note in notes],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def read_notes_json(input_path: Path) -> NotesDocument:
    """Load and validate a schema-version-1 note sidecar."""
    payload = cast(dict[str, object], json.loads(input_path.read_text(encoding="utf-8")))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported or missing notes JSON schema_version")

    source = payload.get("source")
    transcriber = payload.get("transcriber")
    raw_notes = payload.get("notes")
    if not isinstance(source, str):
        raise ValueError("notes JSON source must be a string")
    if not isinstance(transcriber, dict) or not isinstance(transcriber.get("model_size"), str):
        raise ValueError("notes JSON transcriber.model_size must be a string")
    if not isinstance(raw_notes, list):
        raise ValueError("notes JSON notes must be a list")

    notes: list[BassNote] = []
    for index, raw_note in enumerate(raw_notes):
        if not isinstance(raw_note, dict):
            raise ValueError(f"note {index} must be an object")
        try:
            instrument = raw_note["instrument"]
            if instrument not in ("electric_bass", "acoustic_bass"):
                raise ValueError(f"note {index} has unsupported instrument {instrument!r}")
            confidence = raw_note.get("confidence")
            notes.append(
                BassNote(
                    pitch=int(cast(int, raw_note["pitch"])),
                    start_seconds=float(cast(float, raw_note["start_seconds"])),
                    end_seconds=float(cast(float, raw_note["end_seconds"])),
                    instrument=cast(BassInstrument, instrument),
                    confidence=None if confidence is None else float(cast(float, confidence)),
                )
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"note {index} has invalid or missing fields") from error

    return NotesDocument(
        source=Path(source),
        model_size=cast(str, transcriber["model_size"]),
        notes=notes,
    )
