from dataclasses import replace
from itertools import product

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
from bass_transcriber.tab.fingering import (
    FingeringCandidate,
    _placement_cost,
    _root_fifth_octave_cost,
    _search,
    _SearchState,
    _transition_cost,
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


def test_supported_strategies_include_combined_and_preserve_legacy_aliases() -> None:
    assert tuple(FINGERING_PROFILES) == ("balanced", "avoid_open", "combined")
    events = [FingeringEvent(34, 0, 4), FingeringEvent(41, 4, 8), FingeringEvent(46, 8, 12)]
    assert optimize_fingering(events, "compact") == optimize_fingering(events, "balanced")


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


def test_soft_position_preference_can_move_a_riff_without_changing_defaults() -> None:
    events = [
        FingeringEvent(pitch, i * 4, i * 4 + 4)
        for i, pitch in enumerate([34, 41, 46, 41] * 3)
    ]
    profile = FINGERING_PROFILES["avoid_open"]
    original = optimize_fingering(events, "avoid_open", strings=EADG_STRINGS)
    experimental = optimize_fingering(
        events, "avoid_open", strings=EADG_STRINGS,
        profile_override=replace(profile, hand_position_weight=0.15),
    )

    assert [(n.string, n.fret) for n in original[:3]] == [(3, 1), (2, 3), (1, 3)]
    assert [(n.string, n.fret) for n in experimental[:3]] == [(4, 6), (3, 8), (2, 8)]
    assert FINGERING_PROFILES["avoid_open"] is profile
    assert optimize_fingering(events, "avoid_open", strings=EADG_STRINGS) == original


def test_position_preference_still_allows_required_low_notes() -> None:
    events = [FingeringEvent(pitch, i * 4, i * 4 + 4) for i, pitch in enumerate([28, 29, 30])]
    result = optimize_fingering(
        events, "avoid_open", strings=EADG_STRINGS,
        profile_override=replace(FINGERING_PROFILES["avoid_open"], hand_position_weight=10),
    )

    assert [(n.string, n.fret) for n in result] == [(4, 0), (4, 1), (4, 2)]


def test_shape_search_matches_exhaustive_paths_and_reconstructs_their_cost() -> None:
    events = [FingeringEvent(p, i * 4, i * 4 + 4)
              for i, p in enumerate([34, 41, 46, 41, 34])]
    choices = {
        34: ((4, 6, 5), (3, 1, 1)),
        41: ((3, 8, 5), (2, 3, 3)),
        46: ((2, 8, 5), (1, 3, 3)),
    }
    states = [tuple(_SearchState(FingeringCandidate(s, f), h) for s, f, h in choices[e.pitch])
              for e in events]
    profile = replace(FINGERING_PROFILES["avoid_open"], hand_position_weight=0.4,
                      same_string_jump_weight=0.5, root_fifth_octave_reward=4)

    def cost(path: tuple[_SearchState, ...]) -> float:
        return (
            sum(_placement_cost(events, i, s, profile) for i, s in enumerate(path))
            + sum(_transition_cost(events[i - 1], path[i - 1], events[i], path[i], profile,
                                   0.08) for i in range(1, len(path)))
            + sum(_root_fifth_octave_cost(events, i, path[i - 2], path[i - 1], path[i],
                                         profile) for i in range(2, len(path)))
        )

    path, total = _search(events, states, profile, 0.08)
    assert total == pytest.approx(min(cost(p) for p in product(*states)))
    assert total == pytest.approx(cost(tuple(path)))
    assert [(s.candidate.string, s.candidate.fret) for s in path] == [
        (4, 6), (3, 8), (2, 8), (3, 8), (4, 6),
    ]


def test_same_string_jump_cost_distinguishes_sustain_rest_and_open_strings() -> None:
    profile = replace(FINGERING_PROFILES["avoid_open"], same_string_jump_weight=0.5)
    base = replace(profile, same_string_jump_weight=0)
    first = _SearchState(FingeringCandidate(3, 1), 1)
    last = _SearchState(FingeringCandidate(3, 8), 5)
    first_event = FingeringEvent(34, 0, 12)

    def added_cost(current: FingeringEvent, previous: _SearchState = first) -> float:
        return (_transition_cost(first_event, previous, current, last, profile, 0.08)
                - _transition_cost(first_event, previous, current, last, base, 0.08))

    sustained = added_cost(FingeringEvent(41, 12, 16))
    after_rest = added_cost(FingeringEvent(41, 24, 28))
    assert sustained > after_rest > 0
    assert added_cost(FingeringEvent(41, 12, 16),
                      _SearchState(FingeringCandidate(3, 0), 1)) == 0


@pytest.mark.parametrize("profile", ["balanced", "avoid_open", "combined"])
def test_five_string_reserves_low_b_for_notes_below_open_e(profile: str) -> None:
    pitches = [26, 27, 28, 30, 31, 26]
    events = [
        FingeringEvent(pitch, index * 2, index * 2 + 2)
        for index, pitch in enumerate(pitches)
    ]

    result = optimize_fingering(events, profile)

    assert [(choice.string, choice.fret) for choice in result] == [
        (5, 3), (5, 4), (4, 0), (4, 2), (4, 3), (5, 3),
    ]


@pytest.mark.parametrize("profile", ["balanced", "avoid_open", "combined"])
@pytest.mark.parametrize("pitches", [(33, 34), (28, 33, 38, 43, 48)])
def test_chords_avoid_upper_b_unless_needed_for_all_notes(
    profile: str, pitches: tuple[int, ...],
) -> None:
    grid = RhythmGrid(
        detector="test", bpm=120.0, beats_per_bar=4,
        first_downbeat_seconds=0.0, beat_times_seconds=(0.0, 0.5),
        onset_delay_seconds=0.0,
    )
    notes = [BassNote(pitch, 0.0, 0.25, "electric_bass") for pitch in pitches]

    result = build_fingering_timeline(notes, grid, profile)

    assert len(result.notes) == len(pitches)
    assert not result.dropped_pitches
    assert not result.pruned_chords
    assert len({note.string for note in result.notes}) == len(pitches)
    upper_b_notes = [note for note in result.notes if note.string == 5 and note.fret >= 5]
    assert len(upper_b_notes) == (1 if len(pitches) == 5 else 0)
