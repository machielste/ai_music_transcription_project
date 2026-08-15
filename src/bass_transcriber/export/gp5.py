"""Minimal four- and five-string bass Guitar Pro 5 export."""

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
    FingeringEvent,
    generate_candidates,
    optimize_fingering,
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


@dataclass(frozen=True, slots=True)
class _TabNote:
    pitch: int
    start_slot: int
    end_slot: int
    string: int
    fret: int


@dataclass(frozen=True, slots=True)
class GP5ExportResult:
    """Details about notes accepted or dropped during GP5 export."""

    exported_note_count: int
    dropped_pitches: tuple[int, ...]


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
    tab_notes, dropped_pitches = _quantize_and_finger(notes, grid, fingering_profile, strings)
    if not tab_notes:
        if dropped_pitches:
            raise ValueError(
                f"cannot export: every transcribed note is outside {tuning_name} range"
            )
        raise ValueError("cannot export an empty transcription")

    measure_count = math.ceil(tab_notes[-1].end_slot / SLOTS_PER_BAR)
    tempo_schedule = _tempo_schedule(grid.bpm, measure_count)
    song = _new_song(
        title=title,
        initial_bpm=tempo_schedule[0],
        measure_count=measure_count,
        strings=strings,
    )
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

    _attach_tempo_schedule(track.measures, tempo_schedule)

    output.parent.mkdir(parents=True, exist_ok=True)
    guitarpro.write(song, str(output), version=(5, 1, 0))
    _validate_round_trip(output, strings)
    return GP5ExportResult(len(tab_notes), tuple(dropped_pitches))


def _quantize_and_finger(
    notes: list[BassNote],
    grid: RhythmGrid,
    fingering_profile: str | None,
    strings: tuple[tuple[int, int], ...],
) -> tuple[list[_TabNote], list[int]]:
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

    if fingering_profile is None:
        fingerings = [_choose_fingering(event.pitch, strings) for event in events]
    else:
        optimized = optimize_fingering(events, fingering_profile, strings=strings)
        fingerings = [(choice.string, choice.fret) for choice in optimized]
    result = [
        _TabNote(event.pitch, event.start_slot, event.end_slot, string, fret)
        for event, (string, fret) in zip(events, fingerings, strict=True)
    ]

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
    return cleaned, dropped_pitches


def _choose_fingering(
    pitch: int, strings: tuple[tuple[int, int], ...] = BEADG_STRINGS
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


def _new_song(
    *,
    title: str,
    initial_bpm: int,
    measure_count: int,
    strings: tuple[tuple[int, int], ...],
) -> Song:
    song = Song(title=title, tempo=initial_bpm, tempoName="Detected tempo")
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


def _validate_round_trip(output: Path, expected_strings: tuple[tuple[int, int], ...]) -> None:
    parsed = guitarpro.parse(str(output))
    expected_tuning = [pitch for _, pitch in expected_strings]
    actual_tuning = [string.value for string in parsed.tracks[0].strings]
    if actual_tuning != expected_tuning:
        raise RuntimeError(f"GP5 tuning round-trip failed: {actual_tuning}")
    if not any(measure.voices[0].beats for measure in parsed.tracks[0].measures):
        raise RuntimeError("GP5 round-trip produced an empty score")
