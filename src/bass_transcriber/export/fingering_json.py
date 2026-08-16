"""Editable JSON artifact for resolved bass tablature fingerings."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from itertools import groupby
from pathlib import Path
from typing import TypeGuard, cast

from bass_transcriber.models import RhythmGrid
from bass_transcriber.tab import FingeringTimeline, PrunedChord, ResolvedTabNote


@dataclass(frozen=True, slots=True)
class FingeringDocument:
    """A validated, editable fingering draft ready for GP5 export."""

    source: Path
    title: str
    profile: str | None
    strings: tuple[tuple[int, int], ...]
    fret_count: int
    rhythm: RhythmGrid
    notes: tuple[ResolvedTabNote, ...]
    dropped_pitches: tuple[int, ...] = ()
    pruned_chords: tuple[PrunedChord, ...] = ()


def write_fingering_json(
    output: Path,
    timeline: FingeringTimeline,
    *,
    source: Path,
    title: str,
    profile: str | None,
    strings: tuple[tuple[int, int], ...],
    rhythm: RhythmGrid,
    fret_count: int = 24,
) -> None:
    """Write a human-editable resolved fingering draft."""
    draft = FingeringDocument(
        source=source.resolve(),
        title=title,
        profile=profile,
        strings=strings,
        fret_count=fret_count,
        rhythm=rhythm,
        notes=timeline.notes,
        dropped_pitches=timeline.dropped_pitches,
        pruned_chords=timeline.pruned_chords,
    )
    document = {
        "schema_version": 1,
        "document_type": "bass_fingering_draft",
        "immutable_sha256": _immutable_sha256(draft),
        "source": str(draft.source),
        "title": title,
        "fingering_profile": profile,
        "tuning": {
            "name": "BEADG" if len(strings) == 5 else "EADG",
            "fret_count": fret_count,
            "strings": [
                {"number": number, "open_pitch": open_pitch}
                for number, open_pitch in strings
            ],
        },
        "rhythm": asdict(rhythm),
        "notes": [
            {
                "id": index,
                "pitch": note.pitch,
                "start_slot": note.start_slot,
                "end_slot": note.end_slot,
                "string": note.string,
                "fret": note.fret,
            }
            for index, note in enumerate(timeline.notes, start=1)
        ],
        "diagnostics": {
            "dropped_pitches": list(timeline.dropped_pitches),
            "pruned_chords": [asdict(chord) for chord in timeline.pruned_chords],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def read_fingering_json(input_path: Path) -> FingeringDocument:
    """Load and strictly validate a manually editable fingering draft."""
    try:
        payload = cast(dict[str, object], json.loads(input_path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid fingering JSON at line {error.lineno}, column {error.colno}: {error.msg}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("fingering JSON root must be an object")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported or missing fingering JSON schema_version")
    if payload.get("document_type") != "bass_fingering_draft":
        raise ValueError("unsupported or missing fingering JSON document_type")
    immutable_sha256 = _required_string(payload, "immutable_sha256", "fingering JSON")

    source = _required_string(payload, "source", "fingering JSON")
    title = _required_string(payload, "title", "fingering JSON")
    profile_value = payload.get("fingering_profile")
    if profile_value is not None and not isinstance(profile_value, str):
        raise ValueError("fingering JSON fingering_profile must be a string or null")

    tuning = _required_object(payload, "tuning", "fingering JSON")
    fret_count = _required_int(tuning, "fret_count", "fingering JSON tuning")
    if fret_count < 0:
        raise ValueError("fingering JSON tuning.fret_count must not be negative")
    raw_strings = tuning.get("strings")
    if not isinstance(raw_strings, list) or not raw_strings:
        raise ValueError("fingering JSON tuning.strings must be a non-empty array")
    strings: list[tuple[int, int]] = []
    for index, raw_string in enumerate(raw_strings, start=1):
        if not isinstance(raw_string, dict):
            raise ValueError(f"fingering JSON tuning string {index} must be an object")
        number = _required_int(raw_string, "number", f"fingering JSON tuning string {index}")
        open_pitch = _required_int(
            raw_string,
            "open_pitch",
            f"fingering JSON tuning string {index}",
        )
        if not 0 <= open_pitch <= 127:
            raise ValueError(f"fingering JSON tuning string {index} has invalid open_pitch")
        strings.append((number, open_pitch))
    if len({number for number, _ in strings}) != len(strings):
        raise ValueError("fingering JSON tuning contains duplicate string numbers")

    rhythm = _read_rhythm(_required_object(payload, "rhythm", "fingering JSON"))
    raw_notes = payload.get("notes")
    if not isinstance(raw_notes, list) or not raw_notes:
        raise ValueError("fingering JSON notes must be a non-empty array")
    open_pitch_by_string = dict(strings)
    notes: list[ResolvedTabNote] = []
    previous_start = -1
    for index, raw_note in enumerate(raw_notes, start=1):
        context = f"fingering JSON note {index}"
        if not isinstance(raw_note, dict):
            raise ValueError(f"{context} must be an object")
        note_id = _required_int(raw_note, "id", context)
        if note_id != index:
            raise ValueError(f"{context} must retain id {index}, got {note_id}")
        pitch = _required_int(raw_note, "pitch", context)
        start_slot = _required_int(raw_note, "start_slot", context)
        end_slot = _required_int(raw_note, "end_slot", context)
        string = _required_int(raw_note, "string", context)
        fret = _required_int(raw_note, "fret", context)
        if not 0 <= pitch <= 127:
            raise ValueError(f"{context} pitch must be between 0 and 127")
        if start_slot < 0 or end_slot <= start_slot:
            raise ValueError(f"{context} must have non-negative timing and positive duration")
        if start_slot < previous_start:
            raise ValueError(f"{context} is out of start-time order")
        previous_start = start_slot
        if string not in open_pitch_by_string:
            available = ", ".join(str(number) for number in open_pitch_by_string)
            raise ValueError(f"{context} uses string {string}; available strings: {available}")
        if not 0 <= fret <= fret_count:
            raise ValueError(f"{context} fret must be between 0 and {fret_count}")
        expected_pitch = open_pitch_by_string[string] + fret
        if pitch != expected_pitch:
            expected_fret = pitch - open_pitch_by_string[string]
            raise ValueError(
                f"{context} MIDI pitch {pitch} cannot be played on string {string} "
                f"at fret {fret}; expected fret {expected_fret}"
            )
        notes.append(ResolvedTabNote(pitch, start_slot, end_slot, string, fret))
    _validate_string_occupancy(notes)

    diagnostics_value = payload.get("diagnostics", {})
    if not isinstance(diagnostics_value, dict):
        raise ValueError("fingering JSON diagnostics must be an object")
    dropped_pitches = _read_int_array(
        diagnostics_value.get("dropped_pitches", []),
        "fingering JSON diagnostics.dropped_pitches",
    )
    pruned_chords = _read_pruned_chords(diagnostics_value.get("pruned_chords", []))
    document = FingeringDocument(
        source=Path(source),
        title=title,
        profile=profile_value,
        strings=tuple(strings),
        fret_count=fret_count,
        rhythm=rhythm,
        notes=tuple(notes),
        dropped_pitches=dropped_pitches,
        pruned_chords=pruned_chords,
    )
    if immutable_sha256 != _immutable_sha256(document):
        raise ValueError(
            "fingering JSON immutable content changed; edit only note string and fret values"
        )
    return document


def _read_rhythm(payload: dict[str, object]) -> RhythmGrid:
    detector = _required_string(payload, "detector", "fingering JSON rhythm")
    bpm = _required_number(payload, "bpm", "fingering JSON rhythm")
    beats_per_bar_value = payload.get("beats_per_bar")
    if beats_per_bar_value is not None and not _is_int(beats_per_bar_value):
        raise ValueError("fingering JSON rhythm.beats_per_bar must be an integer or null")
    first_downbeat_value = payload.get("first_downbeat_seconds")
    if first_downbeat_value is not None and not _is_number(first_downbeat_value):
        raise ValueError(
            "fingering JSON rhythm.first_downbeat_seconds must be a number or null"
        )
    raw_beats = payload.get("beat_times_seconds")
    if not isinstance(raw_beats, list) or not all(_is_number(value) for value in raw_beats):
        raise ValueError("fingering JSON rhythm.beat_times_seconds must be a number array")
    onset_delay = _required_number(payload, "onset_delay_seconds", "fingering JSON rhythm")
    return RhythmGrid(
        detector=detector,
        bpm=bpm,
        beats_per_bar=beats_per_bar_value,
        first_downbeat_seconds=(
            None if first_downbeat_value is None else float(first_downbeat_value)
        ),
        beat_times_seconds=tuple(float(value) for value in raw_beats),
        onset_delay_seconds=onset_delay,
    )


def _read_pruned_chords(value: object) -> tuple[PrunedChord, ...]:
    if not isinstance(value, list):
        raise ValueError("fingering JSON diagnostics.pruned_chords must be an array")
    chords: list[PrunedChord] = []
    for index, raw_chord in enumerate(value, start=1):
        context = f"fingering JSON pruned chord {index}"
        if not isinstance(raw_chord, dict):
            raise ValueError(f"{context} must be an object")
        chords.append(
            PrunedChord(
                start_slot=_required_int(raw_chord, "start_slot", context),
                available_strings=_required_int(raw_chord, "available_strings", context),
                original_pitches=_read_int_array(
                    raw_chord.get("original_pitches"),
                    f"{context}.original_pitches",
                ),
                kept_pitch=_required_int(raw_chord, "kept_pitch", context),
                removed_pitches=_read_int_array(
                    raw_chord.get("removed_pitches"),
                    f"{context}.removed_pitches",
                ),
                reason=_required_string(raw_chord, "reason", context),
            )
        )
    return tuple(chords)


def _validate_string_occupancy(notes: list[ResolvedTabNote]) -> None:
    ordered = sorted(notes, key=lambda note: (note.string, note.start_slot, note.end_slot))
    for string, group_iterator in groupby(ordered, key=lambda note: note.string):
        previous: ResolvedTabNote | None = None
        for note in group_iterator:
            if previous is not None and note.start_slot < previous.end_slot:
                raise ValueError(
                    f"fingering JSON notes {previous.start_slot}-{previous.end_slot} and "
                    f"{note.start_slot}-{note.end_slot} overlap on string {string}"
                )
            previous = note


def _immutable_sha256(document: FingeringDocument) -> str:
    payload = {
        "source": str(document.source),
        "title": document.title,
        "profile": document.profile,
        "strings": document.strings,
        "fret_count": document.fret_count,
        "rhythm": asdict(document.rhythm),
        "notes": [
            (index, note.pitch, note.start_slot, note.end_slot)
            for index, note in enumerate(document.notes, start=1)
        ],
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _required_object(
    payload: dict[str, object],
    key: str,
    context: str,
) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{context} {key} must be an object")
    return cast(dict[str, object], value)


def _required_string(payload: dict[str, object], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} {key} must be a non-empty string")
    return value


def _required_int(payload: dict[str, object], key: str, context: str) -> int:
    value = payload.get(key)
    if not _is_int(value):
        raise ValueError(f"{context} {key} must be an integer")
    return value


def _required_number(payload: dict[str, object], key: str, context: str) -> float:
    value = payload.get(key)
    if not _is_number(value):
        raise ValueError(f"{context} {key} must be a number")
    return float(value)


def _read_int_array(value: object, context: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(_is_int(item) for item in value):
        raise ValueError(f"{context} must be an integer array")
    return tuple(cast(int, item) for item in value)


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )
