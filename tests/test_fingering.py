import pytest

from bass_transcriber.models import BassNote, RhythmGrid
from bass_transcriber.tab import (
    EADG_STRINGS,
    FINGERING_PROFILES,
    FingeringEvent,
    build_fingering_timeline,
    generate_candidates,
    optimize_fingering,
)


def test_candidate_generation_returns_every_playable_position() -> None:
    candidates = generate_candidates(38)  # D2

    assert [(choice.string, choice.fret) for choice in candidates] == [
        (2, 0),
        (3, 5),
        (4, 10),
        (5, 15),
    ]


def test_balanced_profile_keeps_descending_run_on_a_string() -> None:
    events = [
        FingeringEvent(38, 0, 2),
        FingeringEvent(37, 2, 4),
        FingeringEvent(35, 4, 6),
    ]

    result = optimize_fingering(events, "balanced")

    assert [(choice.string, choice.fret) for choice in result] == [
        (3, 5),
        (3, 4),
        (3, 2),
    ]


def test_only_balanced_and_avoid_open_are_offered_for_new_fingerings() -> None:
    assert tuple(FINGERING_PROFILES) == ("balanced", "avoid_open")


def test_balanced_open_string_does_not_force_the_hand_to_position_zero() -> None:
    events = [
        FingeringEvent(33, 0, 2),
        FingeringEvent(33, 2, 4),
        FingeringEvent(37, 4, 6),
    ]

    result = optimize_fingering(events, "balanced", strings=EADG_STRINGS)

    assert [(choice.string, choice.fret) for choice in result] == [
        (3, 0),
        (3, 0),
        (3, 4),
    ]


def test_avoid_open_uses_a_coherent_fretted_alternative() -> None:
    events = [
        FingeringEvent(33, 0, 2),
        FingeringEvent(33, 2, 4),
        FingeringEvent(37, 4, 6),
    ]

    result = optimize_fingering(events, "avoid_open", strings=EADG_STRINGS)

    assert [(choice.string, choice.fret) for choice in result] == [
        (4, 5),
        (4, 5),
        (3, 4),
    ]


def test_real_timing_changes_the_best_way_to_prepare_a_large_jump() -> None:
    events = [
        FingeringEvent(33, 0, 2),
        FingeringEvent(33, 2, 4),
        FingeringEvent(48, 4, 6),
    ]

    fast = optimize_fingering(
        events,
        "balanced",
        strings=EADG_STRINGS,
        seconds_per_slot=0.015,
    )
    slow = optimize_fingering(
        events,
        "balanced",
        strings=EADG_STRINGS,
        seconds_per_slot=0.15,
    )

    assert [(choice.string, choice.fret) for choice in fast] == [
        (3, 0),
        (3, 0),
        (3, 15),
    ]
    assert [(choice.string, choice.fret) for choice in slow] == [
        (3, 0),
        (3, 0),
        (2, 10),
    ]


def test_a_long_rest_makes_repositioning_on_the_same_string_reasonable() -> None:
    continuous = [
        FingeringEvent(33, 0, 2),
        FingeringEvent(34, 2, 4),
        FingeringEvent(38, 4, 6),
    ]
    after_rest = [
        FingeringEvent(33, 0, 2),
        FingeringEvent(34, 2, 4),
        FingeringEvent(38, 20, 22),
    ]

    continuous_result = optimize_fingering(
        continuous, "balanced", strings=EADG_STRINGS
    )
    rest_result = optimize_fingering(after_rest, "balanced", strings=EADG_STRINGS)

    assert (continuous_result[-1].string, continuous_result[-1].fret) == (3, 5)
    assert (rest_result[-1].string, rest_result[-1].fret) == (4, 10)


def test_exact_repeated_riffs_are_locked_to_one_fingering() -> None:
    riff = [38, 40, 42, 40, 38, 35]
    pitches = [33, *riff, 33, 33, *riff, 52]
    events = [
        FingeringEvent(pitch, index * 2, index * 2 + 2)
        for index, pitch in enumerate(pitches)
    ]

    unlocked = optimize_fingering(
        events,
        "balanced",
        strings=EADG_STRINGS,
        enforce_repeated_phrases=False,
    )
    locked = optimize_fingering(events, "balanced", strings=EADG_STRINGS)

    assert unlocked[1:7] != unlocked[9:15]
    assert locked[1:7] == locked[9:15]


def test_saved_legacy_profile_names_remain_reexportable() -> None:
    events = [FingeringEvent(38, 0, 2), FingeringEvent(37, 2, 4)]

    assert optimize_fingering(events, "compact") == optimize_fingering(
        events, "balanced"
    )


def test_a_rare_upper_double_stop_does_not_distort_the_bass_backbone() -> None:
    grid = RhythmGrid(
        detector="test",
        bpm=120.0,
        beats_per_bar=4,
        first_downbeat_seconds=0.0,
        beat_times_seconds=(0.0, 0.5, 1.0),
        onset_delay_seconds=0.0,
    )
    bassline = [
        BassNote(40, 0.0, 0.25, "electric_bass"),
        BassNote(38, 0.25, 0.5, "electric_bass"),
        BassNote(40, 0.5, 0.75, "electric_bass"),
    ]
    with_double_stop = [
        bassline[0],
        BassNote(50, 0.25, 0.5, "electric_bass"),
        bassline[1],
        bassline[2],
    ]

    baseline = build_fingering_timeline(
        bassline, grid, "balanced", strings=EADG_STRINGS
    )
    chord_result = build_fingering_timeline(
        with_double_stop, grid, "balanced", strings=EADG_STRINGS
    )

    assert [
        (note.pitch, note.string, note.fret)
        for note in chord_result.notes
        if note.pitch != 50
    ] == [(note.pitch, note.string, note.fret) for note in baseline.notes]
    middle = [note for note in chord_result.notes if note.start_slot == 4]
    assert len({note.string for note in middle}) == 2


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown fingering profile"):
        optimize_fingering([FingeringEvent(28, 0, 2)], "unknown")
