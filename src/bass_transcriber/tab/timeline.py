"""Quantized, fully resolved bass tablature timelines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import groupby, product

from bass_transcriber.models import BassNote, RhythmGrid
from bass_transcriber.tab.fingering import (
    BEADG_STRINGS,
    FingeringCandidate,
    FingeringEvent,
    generate_candidates,
    optimize_fingering,
)

SUBDIVISIONS_PER_BEAT = 8


@dataclass(frozen=True, slots=True)
class ResolvedTabNote:
    """One quantized note with its final playable string and fret."""

    pitch: int
    start_slot: int
    end_slot: int
    string: int
    fret: int


@dataclass(frozen=True, slots=True)
class FingeringTimeline:
    """Resolved notes and pitches rejected by the selected tuning."""

    notes: tuple[ResolvedTabNote, ...]
    dropped_pitches: tuple[int, ...]


def build_fingering_timeline(
    notes: Sequence[BassNote],
    grid: RhythmGrid,
    profile_name: str | None = "balanced",
    *,
    strings: Sequence[tuple[int, int]] = BEADG_STRINGS,
) -> FingeringTimeline:
    """Quantize notes and resolve their final string/fret assignments.

    A ``None`` profile retains the legacy lowest-fret-per-note behavior. Chord
    collisions are repaired after phrase optimization so consumers see the
    same final placements that are written to Guitar Pro.
    """
    period_seconds = 60.0 / grid.bpm
    first_beat = grid.beat_times_seconds[0]
    offset_slots = round(first_beat / period_seconds * SUBDIVISIONS_PER_BEAT)
    events: list[FingeringEvent] = []
    dropped_pitches: list[int] = []

    for raw_note in sorted(notes, key=lambda item: item.start_seconds):
        if not generate_candidates(raw_note.pitch, strings=strings):
            dropped_pitches.append(raw_note.pitch)
            continue
        corrected_start = raw_note.start_seconds - grid.onset_delay_seconds
        corrected_end = raw_note.end_seconds - grid.onset_delay_seconds
        start_slot = max(
            0,
            round((corrected_start - first_beat) / period_seconds * SUBDIVISIONS_PER_BEAT)
            + offset_slots,
        )
        end_slot = max(
            start_slot + 1,
            round((corrected_end - first_beat) / period_seconds * SUBDIVISIONS_PER_BEAT)
            + offset_slots,
        )
        events.append(FingeringEvent(raw_note.pitch, start_slot, end_slot))

    if profile_name is None:
        fingerings = [_choose_fingering(event.pitch, strings) for event in events]
    else:
        optimized = optimize_fingering(events, profile_name, strings=strings)
        fingerings = [(choice.string, choice.fret) for choice in optimized]
    resolved = [
        ResolvedTabNote(event.pitch, event.start_slot, event.end_slot, string, fret)
        for event, (string, fret) in zip(events, fingerings, strict=True)
    ]
    resolved = _assign_distinct_chord_strings(resolved, strings)

    # Prevent independently rounded offsets from extending through the next
    # attack while retaining every note in an explicit simultaneous chord.
    cleaned: list[ResolvedTabNote] = []
    onset_groups = [
        list(group)
        for _, group in groupby(resolved, key=lambda tab_note: tab_note.start_slot)
    ]
    for group_index, chord in enumerate(onset_groups):
        next_start = (
            onset_groups[group_index + 1][0].start_slot
            if group_index + 1 < len(onset_groups)
            else None
        )
        for tab_note in chord:
            end_slot = (
                min(tab_note.end_slot, next_start)
                if next_start is not None
                else tab_note.end_slot
            )
            cleaned.append(
                ResolvedTabNote(
                    tab_note.pitch,
                    tab_note.start_slot,
                    max(tab_note.start_slot + 1, end_slot),
                    tab_note.string,
                    tab_note.fret,
                )
            )
    return FingeringTimeline(tuple(cleaned), tuple(dropped_pitches))


def slot_duration_seconds(grid: RhythmGrid) -> float:
    """Return the duration of one tablature slot for a rhythm grid."""
    return 60.0 / grid.bpm / SUBDIVISIONS_PER_BEAT


def source_aligned_seconds(slot: int, grid: RhythmGrid) -> float:
    """Map an onset-corrected score slot back onto the source-audio clock."""
    return max(0.0, slot * slot_duration_seconds(grid) + grid.onset_delay_seconds)


def _assign_distinct_chord_strings(
    notes: list[ResolvedTabNote],
    strings: Sequence[tuple[int, int]],
) -> list[ResolvedTabNote]:
    """Keep optimized fingerings when possible and resolve chord collisions."""
    resolved: list[ResolvedTabNote] = []
    for _, group_iterator in groupby(notes, key=lambda note: note.start_slot):
        chord = list(group_iterator)
        if len(chord) == 1 or len({note.string for note in chord}) == len(chord):
            resolved.extend(chord)
            continue
        if len(chord) > len(strings):
            raise ValueError(
                f"cannot place {len(chord)} simultaneous notes on {len(strings)} strings"
            )

        preferred = [FingeringCandidate(note.string, note.fret) for note in chord]
        candidates = [generate_candidates(note.pitch, strings=strings) for note in chord]
        combinations = (
            combination
            for combination in product(*candidates)
            if len({choice.string for choice in combination}) == len(combination)
        )
        try:
            selected = min(
                combinations,
                key=lambda combination: _chord_fingering_cost(combination, preferred),
            )
        except ValueError as error:
            pitches = ", ".join(str(note.pitch) for note in chord)
            raise ValueError(
                f"cannot place simultaneous MIDI pitches {pitches} on distinct strings"
            ) from error
        resolved.extend(
            ResolvedTabNote(
                note.pitch,
                note.start_slot,
                note.end_slot,
                choice.string,
                choice.fret,
            )
            for note, choice in zip(chord, selected, strict=True)
        )
    return resolved


def _chord_fingering_cost(
    choices: tuple[FingeringCandidate, ...],
    preferred: list[FingeringCandidate],
) -> tuple[int, int, int]:
    changed = sum(choice != original for choice, original in zip(choices, preferred, strict=True))
    displacement = sum(
        abs(choice.fret - original.fret) + abs(choice.string - original.string)
        for choice, original in zip(choices, preferred, strict=True)
    )
    return changed, displacement, sum(choice.fret for choice in choices)


def _choose_fingering(
    pitch: int,
    strings: Sequence[tuple[int, int]] = BEADG_STRINGS,
) -> tuple[int, int]:
    candidates = [
        (candidate.fret, candidate.string)
        for candidate in generate_candidates(pitch, strings=strings)
    ]
    if not candidates:
        tuning_name = "BEADG" if len(strings) == 5 else "EADG"
        raise ValueError(f"MIDI pitch {pitch} is outside {tuning_name} range")
    fret, string = min(candidates)
    return string, fret
