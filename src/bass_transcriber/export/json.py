"""Lossless JSON sidecar export for raw transcription results."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from bass_transcriber.models import BassNote


def write_notes_json(
    output: Path,
    notes: list[BassNote],
    *,
    source: Path,
    model_size: str,
) -> None:
    """Write raw notes and provenance in a stable, inspectable format."""
    document = {
        "schema_version": 1,
        "source": str(source.resolve()),
        "transcriber": {"name": "muscriptor", "model_size": model_size},
        "notes": [asdict(note) for note in notes],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
