import pytest

from bass_transcriber.tab import FingeringEvent, generate_candidates, optimize_fingering


def test_candidate_generation_returns_every_playable_position() -> None:
    candidates = generate_candidates(38)  # D2

    assert [(choice.string, choice.fret) for choice in candidates] == [
        (2, 0),
        (3, 5),
        (4, 10),
        (5, 15),
    ]


def test_slap_profile_keeps_descending_run_on_a_string() -> None:
    events = [
        FingeringEvent(38, 0, 2),
        FingeringEvent(37, 2, 4),
        FingeringEvent(35, 4, 6),
    ]

    result = optimize_fingering(events, "slap_funk")

    assert [(choice.string, choice.fret) for choice in result] == [
        (3, 5),
        (3, 4),
        (3, 2),
    ]


def test_slap_profile_uses_fretted_conventional_octave_shape() -> None:
    events = [FingeringEvent(28, 0, 4), FingeringEvent(40, 4, 8)]

    result = optimize_fingering(events, "slap_funk")

    assert [(choice.string, choice.fret) for choice in result] == [(5, 5), (3, 7)]


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown fingering profile"):
        optimize_fingering([FingeringEvent(28, 0, 2)], "unknown")
