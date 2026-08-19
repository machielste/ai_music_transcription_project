"""Minimal four- and five-string bass Guitar Pro 5 export."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import guitarpro  # type: ignore[import-untyped]
from guitarpro.models import (  # type: ignore[import-untyped]
    Beat,
    BeatStatus,
    Duration,
    GuitarString,
    Measure,
    MeasureHeader,
    MixTableChange,
    MixTableItem,
    Note,
    NoteType,
    Song,
    TimeSignature,
)

from bass_transcriber.models import BassNote, RhythmGrid
from bass_transcriber.tab import (
    BEADG_STRINGS,
    EADG_STRINGS,
    PrunedChord,
    ResolvedTabNote,
    build_fingering_timeline,
)

SUBDIVISIONS_PER_BEAT = 8
SLOTS_PER_BAR = 4 * SUBDIVISIONS_PER_BEAT
TICKS_PER_SLOT = Duration.quarterTime // SUBDIVISIONS_PER_BEAT

_DURATION_VALUE_BY_SLOTS = {
    32: Duration.whole,
    16: Duration.half,
    8: Duration.quarter,
    4: Duration.eighth,
    2: Duration.sixteenth,
    1: Duration.thirtySecond,
}

_GP5_TEXT_ENCODING = "cp1252"
logger = logging.getLogger(__name__)


def _gp5_compatible_text(value: str) -> str:
    """Return text that can be stored in GP5's legacy 8-bit text fields.

    GP5 does not have Unicode metadata fields. PyGuitarPro therefore writes
    them using cp1252 and raises ``UnicodeEncodeError`` for characters such
    as Chinese text or emoji. Keep all representable characters and replace
    only the characters the file format cannot store.
    """
    compatible = value.encode(_GP5_TEXT_ENCODING, errors="replace").decode(
        _GP5_TEXT_ENCODING
    )
    if compatible != value:
        logger.warning(
            "GP5 metadata cannot represent some characters; replaced them with '?': %r",
            value,
        )
    return compatible


@dataclass(frozen=True, slots=True)
class GP5ExportResult:
    """Details about notes accepted or dropped during GP5 export."""

    exported_note_count: int
    dropped_pitches: tuple[int, ...]
    pruned_chords: tuple[PrunedChord, ...] = ()


def write_gp5(
    output: Path,
    notes: list[BassNote],
    grid: RhythmGrid,
    *,
    title: str,
    fingering_profile: str | None = "balanced",
    five_string: bool = True,
) -> GP5ExportResult:
    """Write a first-pass 4/4 GP5 score for a four- or five-string bass."""
    strings = BEADG_STRINGS if five_string else EADG_STRINGS
    tuning_name = "BEADG" if five_string else "EADG"
    tab_notes, dropped_pitches, pruned_chords = _quantize_and_finger(
        notes,
        grid,
        fingering_profile,
        strings,
    )
    if not tab_notes:
        if dropped_pitches:
            raise ValueError(
                f"cannot export: every transcribed note is outside {tuning_name} range"
            )
        raise ValueError("cannot export an empty transcription")

    write_resolved_gp5(
        output,
        tab_notes,
        grid,
        title=title,
        strings=strings,
    )
    return GP5ExportResult(
        len(tab_notes),
        tuple(dropped_pitches),
        tuple(pruned_chords),
    )


def write_resolved_gp5(
    output: Path,
    tab_notes: Sequence[ResolvedTabNote],
    grid: RhythmGrid,
    *,
    title: str,
    strings: Sequence[tuple[int, int]] = BEADG_STRINGS,
) -> None:
    """Write GP5 from already quantized and fingered tablature notes."""
    if not tab_notes:
        raise ValueError("cannot export an empty fingering draft")
    resolved_strings = tuple(strings)

    measure_count = math.ceil(max(note.end_slot for note in tab_notes) / SLOTS_PER_BAR)
    tempo_schedule = _tempo_schedule(grid.bpm, measure_count)
    song = _new_song(
        title=title,
        initial_bpm=tempo_schedule[0],
        measure_count=measure_count,
        strings=resolved_strings,
    )
    track = song.tracks[0]

    for measure_index, measure in enumerate(track.measures):
        measure_start = measure_index * SLOTS_PER_BAR
        measure_end = measure_start + SLOTS_PER_BAR
        relevant = [
            note
            for note in tab_notes
            if note.end_slot > measure_start and note.start_slot < measure_end
        ]
        boundaries = sorted(
            {
                measure_start,
                measure_end,
                *(
                    max(note.start_slot, measure_start)
                    for note in relevant
                ),
                *(
                    min(note.end_slot, measure_end)
                    for note in relevant
                ),
            }
        )
        for span_start, span_end in pairwise(boundaries):
            active = tuple(
                sorted(
                    (
                        note
                        for note in relevant
                        if note.start_slot <= span_start < note.end_slot
                    ),
                    key=lambda note: note.string,
                )
            )
            _append_span(measure, span_start, span_end, active)

    _attach_tempo_schedule(track.measures, tempo_schedule)

    output.parent.mkdir(parents=True, exist_ok=True)
    guitarpro.write(song, str(output), version=(5, 1, 0))
    _validate_round_trip(output, resolved_strings)


def _quantize_and_finger(
    notes: list[BassNote],
    grid: RhythmGrid,
    fingering_profile: str | None,
    strings: tuple[tuple[int, int], ...],
) -> tuple[list[ResolvedTabNote], list[int], list[PrunedChord]]:
    timeline = build_fingering_timeline(
        notes,
        grid,
        fingering_profile,
        strings=strings,
    )
    return (
        list(timeline.notes),
        list(timeline.dropped_pitches),
        list(timeline.pruned_chords),
    )


def _new_song(
    *,
    title: str,
    initial_bpm: int,
    measure_count: int,
    strings: tuple[tuple[int, int], ...],
) -> Song:
    song = Song(
        title=_gp5_compatible_text(title),
        tempo=initial_bpm,
        tempoName="Detected tempo",
    )
    track = song.tracks[0]
    tuning_name = "BEADG" if len(strings) == 5 else "EADG"
    track.name = f"Bass ({len(strings)}-string {tuning_name})"
    track.fretCount = 24
    track.indicateTuning = True
    track.strings = [GuitarString(number, pitch) for number, pitch in strings]
    track.channel.instrument = 33  # General MIDI electric bass (finger), zero-based.

    first_header = song.measureHeaders[0]
    first_header.timeSignature = TimeSignature(4, Duration(Duration.quarter))
    while len(song.measureHeaders) < measure_count:
        previous = song.measureHeaders[-1]
        header = MeasureHeader(
            number=len(song.measureHeaders) + 1,
            start=previous.end,
            timeSignature=TimeSignature(4, Duration(Duration.quarter)),
        )
        song.addMeasureHeader(header)
        track.measures.append(Measure(track, header))
    return song


def _tempo_schedule(target_bpm: float, measure_count: int) -> list[int]:
    """Approximate fractional BPM with integer tempos using error diffusion."""
    if measure_count < 1:
        raise ValueError("tempo schedule requires at least one measure")
    lower = math.floor(target_bpm)
    upper = math.ceil(target_bpm)
    if lower == upper:
        return [lower] * measure_count

    target_bar_seconds = 240.0 / target_bpm
    elapsed = 0.0
    schedule: list[int] = []
    for measure_index in range(measure_count):
        desired_end = (measure_index + 1) * target_bar_seconds
        tempo = min(
            (lower, upper),
            key=lambda candidate: abs(elapsed + 240.0 / candidate - desired_end),
        )
        schedule.append(tempo)
        elapsed += 240.0 / tempo
    return schedule


def _attach_tempo_schedule(measures: list[Measure], schedule: list[int]) -> None:
    previous = schedule[0]
    for measure, tempo in zip(measures[1:], schedule[1:], strict=True):
        if tempo != previous:
            first_beat = measure.voices[0].beats[0]
            first_beat.effect.mixTableChange = MixTableChange(
                tempo=MixTableItem(value=tempo, duration=0, allTracks=True),
                tempoName="",
                hideTempo=True,
            )
        previous = tempo


def _append_span(
    measure: Measure,
    start_slot: int,
    end_slot: int,
    notes: tuple[ResolvedTabNote, ...],
) -> None:
    voice = measure.voices[0]
    cursor = start_slot
    while cursor < end_slot:
        local_slot = cursor % SLOTS_PER_BAR
        remaining = end_slot - cursor
        size = _largest_aligned_duration(local_slot, remaining)
        beat = Beat(
            voice,
            duration=Duration(_DURATION_VALUE_BY_SLOTS[size]),
            start=measure.header.start + local_slot * TICKS_PER_SLOT,
            status=BeatStatus.normal if notes else BeatStatus.rest,
        )
        for note in notes:
            beat.notes.append(
                Note(
                    beat,
                    value=note.fret,
                    string=note.string,
                    type=NoteType.normal if cursor == note.start_slot else NoteType.tie,
                )
            )
        voice.beats.append(beat)
        cursor += size


def _largest_aligned_duration(local_slot: int, remaining: int) -> int:
    for size in _DURATION_VALUE_BY_SLOTS:
        if size <= remaining and local_slot % size == 0:
            return size
    return 1


def _validate_round_trip(output: Path, expected_strings: tuple[tuple[int, int], ...]) -> None:
    parsed = guitarpro.parse(str(output))
    expected_tuning = [pitch for _, pitch in expected_strings]
    actual_tuning = [string.value for string in parsed.tracks[0].strings]
    if actual_tuning != expected_tuning:
        raise RuntimeError(f"GP5 tuning round-trip failed: {actual_tuning}")
    if not any(measure.voices[0].beats for measure in parsed.tracks[0].measures):
        raise RuntimeError("GP5 round-trip produced an empty score")
