"""Minimal five-string Guitar Pro 5 export."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import guitarpro  # type: ignore[import-untyped]
from guitarpro.models import (  # type: ignore[import-untyped]
    Beat,
    BeatStatus,
    Duration,
    GuitarString,
    Measure,
    MeasureHeader,
    Note,
    NoteType,
    Song,
    TimeSignature,
)

from bass_transcriber.models import BassNote, RhythmGrid

SUBDIVISIONS_PER_BEAT = 8
SLOTS_PER_BAR = 4 * SUBDIVISIONS_PER_BEAT
TICKS_PER_SLOT = Duration.quarterTime // SUBDIVISIONS_PER_BEAT

# Guitar Pro numbers strings from highest to lowest.
BEADG_STRINGS: tuple[tuple[int, int], ...] = (
    (1, 43),  # G2
    (2, 38),  # D2
    (3, 33),  # A1
    (4, 28),  # E1
    (5, 23),  # B0
)

_DURATION_VALUE_BY_SLOTS = {
    32: Duration.whole,
    16: Duration.half,
    8: Duration.quarter,
    4: Duration.eighth,
    2: Duration.sixteenth,
    1: Duration.thirtySecond,
}


@dataclass(frozen=True, slots=True)
class _TabNote:
    pitch: int
    start_slot: int
    end_slot: int
    string: int
    fret: int


def write_gp5(
    output: Path,
    notes: list[BassNote],
    grid: RhythmGrid,
    *,
    title: str,
) -> None:
    """Write a first-pass, 4/4, five-string BEADG GP5 score."""
    tab_notes = _quantize_and_finger(notes, grid)
    if not tab_notes:
        raise ValueError("cannot export an empty transcription")

    measure_count = math.ceil(tab_notes[-1].end_slot / SLOTS_PER_BAR)
    song = _new_song(title=title, bpm=round(grid.bpm), measure_count=measure_count)
    track = song.tracks[0]

    for measure_index, measure in enumerate(track.measures):
        measure_start = measure_index * SLOTS_PER_BAR
        measure_end = measure_start + SLOTS_PER_BAR
        cursor = measure_start
        relevant = [
            note
            for note in tab_notes
            if note.end_slot > measure_start and note.start_slot < measure_end
        ]
        for note in relevant:
            note_start = max(note.start_slot, measure_start)
            note_end = min(note.end_slot, measure_end)
            if note_start > cursor:
                _append_span(measure, cursor, note_start, None)
            _append_span(measure, note_start, note_end, note)
            cursor = note_end
        if cursor < measure_end:
            _append_span(measure, cursor, measure_end, None)

    output.parent.mkdir(parents=True, exist_ok=True)
    guitarpro.write(song, str(output), version=(5, 1, 0))
    _validate_round_trip(output)


def _quantize_and_finger(notes: list[BassNote], grid: RhythmGrid) -> list[_TabNote]:
    period_seconds = 60.0 / grid.bpm
    first_beat = grid.beat_times_seconds[0]
    offset_slots = round(first_beat / period_seconds * SUBDIVISIONS_PER_BEAT)
    result: list[_TabNote] = []

    for raw_note in sorted(notes, key=lambda item: item.start_seconds):
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
        string, fret = _choose_fingering(raw_note.pitch)
        result.append(_TabNote(raw_note.pitch, start_slot, end_slot, string, fret))

    # The source is monophonic. Prevent independently rounded offsets from
    # extending through the next attack while preserving explicit rests.
    cleaned: list[_TabNote] = []
    for index, tab_note in enumerate(result):
        next_start = result[index + 1].start_slot if index + 1 < len(result) else None
        end_slot = (
            min(tab_note.end_slot, next_start) if next_start is not None else tab_note.end_slot
        )
        cleaned.append(
            _TabNote(
                tab_note.pitch,
                tab_note.start_slot,
                max(tab_note.start_slot + 1, end_slot),
                tab_note.string,
                tab_note.fret,
            )
        )
    return cleaned


def _choose_fingering(pitch: int) -> tuple[int, int]:
    candidates = [
        (fret, string)
        for string, open_pitch in BEADG_STRINGS
        if 0 <= (fret := pitch - open_pitch) <= 24
    ]
    if not candidates:
        raise ValueError(f"MIDI pitch {pitch} is outside five-string BEADG range")
    fret, string = min(candidates)
    return string, fret


def _new_song(*, title: str, bpm: int, measure_count: int) -> Song:
    song = Song(title=title, tempo=bpm, tempoName="Detected tempo")
    track = song.tracks[0]
    track.name = "Bass (5-string BEADG)"
    track.fretCount = 24
    track.indicateTuning = True
    track.strings = [GuitarString(number, pitch) for number, pitch in BEADG_STRINGS]
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


def _append_span(
    measure: Measure,
    start_slot: int,
    end_slot: int,
    note: _TabNote | None,
) -> None:
    voice = measure.voices[0]
    cursor = start_slot
    first_piece = True
    while cursor < end_slot:
        local_slot = cursor % SLOTS_PER_BAR
        remaining = end_slot - cursor
        size = _largest_aligned_duration(local_slot, remaining)
        beat = Beat(
            voice,
            duration=Duration(_DURATION_VALUE_BY_SLOTS[size]),
            start=measure.header.start + local_slot * TICKS_PER_SLOT,
            status=BeatStatus.rest if note is None else BeatStatus.normal,
        )
        if note is not None:
            note_type = (
                NoteType.normal if first_piece and cursor == note.start_slot else NoteType.tie
            )
            beat.notes.append(
                Note(
                    beat,
                    value=note.fret,
                    string=note.string,
                    type=note_type,
                )
            )
        voice.beats.append(beat)
        cursor += size
        first_piece = False


def _largest_aligned_duration(local_slot: int, remaining: int) -> int:
    for size in _DURATION_VALUE_BY_SLOTS:
        if size <= remaining and local_slot % size == 0:
            return size
    return 1


def _validate_round_trip(output: Path) -> None:
    parsed = guitarpro.parse(str(output))
    expected_tuning = [pitch for _, pitch in BEADG_STRINGS]
    actual_tuning = [string.value for string in parsed.tracks[0].strings]
    if actual_tuning != expected_tuning:
        raise RuntimeError(f"GP5 tuning round-trip failed: {actual_tuning}")
    if not any(measure.voices[0].beats for measure in parsed.tracks[0].measures):
        raise RuntimeError("GP5 round-trip produced an empty score")
