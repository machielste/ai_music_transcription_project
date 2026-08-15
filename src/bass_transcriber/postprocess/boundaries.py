"""Audio-boundary cleanup for note events."""

from __future__ import annotations

from dataclasses import replace

from bass_transcriber.models import BassNote


def clip_notes_to_duration(notes: list[BassNote], duration_seconds: float) -> list[BassNote]:
    """Remove notes after the source ends and clip notes crossing its boundary."""
    if duration_seconds < 0:
        raise ValueError("audio duration must not be negative")

    clipped: list[BassNote] = []
    for note in notes:
        if note.start_seconds >= duration_seconds:
            continue
        clipped.append(replace(note, end_seconds=min(note.end_seconds, duration_seconds)))
    return clipped
