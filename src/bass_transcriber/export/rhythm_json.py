"""JSON export for detected musical timing."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

from bass_transcriber.models import RhythmGrid


def write_rhythm_json(output: Path, grid: RhythmGrid, *, source: Path) -> None:
    """Write a detected rhythm grid without modifying note timestamps."""
    document = {
        "schema_version": 1,
        "source": str(source.resolve()),
        "detector": grid.detector,
        "grid": asdict(grid),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def read_rhythm_json(input_path: Path) -> RhythmGrid:
    """Load a schema-version-1 rhythm sidecar."""
    payload = cast(dict[str, object], json.loads(input_path.read_text(encoding="utf-8")))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported or missing rhythm JSON schema_version")
    raw_grid = payload.get("grid")
    if not isinstance(raw_grid, dict):
        raise ValueError("rhythm JSON grid must be an object")
    beats = raw_grid.get("beat_times_seconds")
    if not isinstance(beats, list):
        raise ValueError("rhythm JSON beat_times_seconds must be a list")
    first_downbeat = raw_grid.get("first_downbeat_seconds")
    return RhythmGrid(
        detector=str(raw_grid["detector"]),
        bpm=float(cast(float, raw_grid["bpm"])),
        beats_per_bar=(
            None
            if raw_grid.get("beats_per_bar") is None
            else int(cast(int, raw_grid["beats_per_bar"]))
        ),
        first_downbeat_seconds=(
            None if first_downbeat is None else float(cast(float, first_downbeat))
        ),
        beat_times_seconds=tuple(float(cast(float, beat)) for beat in beats),
        onset_delay_seconds=float(cast(float, raw_grid["onset_delay_seconds"])),
    )
