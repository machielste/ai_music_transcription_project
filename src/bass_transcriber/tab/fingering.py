"""Phrase-level string and fret selection for five-string bass."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# Guitar Pro numbers strings from highest to lowest.
BEADG_STRINGS: tuple[tuple[int, int], ...] = (
    (1, 43),  # G2
    (2, 38),  # D2
    (3, 33),  # A1
    (4, 28),  # E1
    (5, 23),  # B0
)
EADG_STRINGS: tuple[tuple[int, int], ...] = BEADG_STRINGS[:-1]


@dataclass(frozen=True, slots=True)
class FingeringEvent:
    """The timing and pitch information needed by the optimizer."""

    pitch: int
    start_slot: int
    end_slot: int


@dataclass(frozen=True, slots=True)
class FingeringCandidate:
    """One playable location for a pitch."""

    string: int
    fret: int


@dataclass(frozen=True, slots=True)
class FingeringProfile:
    """Weights controlling phrase-level fingering preferences."""

    fret_weight: float
    high_fret_weight: float
    position_shift_weight: float
    string_change_weight: float
    string_skip_weight: float
    open_string_penalty: float
    open_in_run_penalty: float
    open_in_octave_penalty: float
    same_string_run_reward: float
    repeated_string_change_penalty: float
    octave_shape_reward: float
    nonstandard_octave_penalty: float
    fast_transition_multiplier: float
    preferred_fret_min: int
    preferred_fret_max: int


FINGERING_PROFILES: dict[str, FingeringProfile] = {
    "balanced": FingeringProfile(
        fret_weight=0.06,
        high_fret_weight=0.30,
        position_shift_weight=0.75,
        string_change_weight=1.25,
        string_skip_weight=0.70,
        open_string_penalty=0.15,
        open_in_run_penalty=4.0,
        open_in_octave_penalty=1.0,
        same_string_run_reward=1.0,
        repeated_string_change_penalty=3.0,
        octave_shape_reward=2.0,
        nonstandard_octave_penalty=1.5,
        fast_transition_multiplier=1.5,
        preferred_fret_min=1,
        preferred_fret_max=12,
    ),
    "slap_funk": FingeringProfile(
        fret_weight=0.04,
        high_fret_weight=0.35,
        position_shift_weight=0.85,
        string_change_weight=1.50,
        string_skip_weight=0.90,
        open_string_penalty=0.35,
        open_in_run_penalty=8.0,
        open_in_octave_penalty=8.0,
        same_string_run_reward=1.8,
        repeated_string_change_penalty=4.0,
        octave_shape_reward=4.0,
        nonstandard_octave_penalty=3.0,
        fast_transition_multiplier=1.8,
        preferred_fret_min=2,
        preferred_fret_max=12,
    ),
    "avoid_open": FingeringProfile(
        fret_weight=0.05,
        high_fret_weight=0.30,
        position_shift_weight=0.80,
        string_change_weight=1.30,
        string_skip_weight=0.75,
        open_string_penalty=7.0,
        open_in_run_penalty=7.0,
        open_in_octave_penalty=7.0,
        same_string_run_reward=1.0,
        repeated_string_change_penalty=3.0,
        octave_shape_reward=2.0,
        nonstandard_octave_penalty=1.5,
        fast_transition_multiplier=1.5,
        preferred_fret_min=1,
        preferred_fret_max=12,
    ),
    "low_positions": FingeringProfile(
        fret_weight=0.35,
        high_fret_weight=0.70,
        position_shift_weight=0.45,
        string_change_weight=0.55,
        string_skip_weight=0.40,
        open_string_penalty=0.0,
        open_in_run_penalty=0.25,
        open_in_octave_penalty=0.0,
        same_string_run_reward=0.25,
        repeated_string_change_penalty=1.0,
        octave_shape_reward=0.5,
        nonstandard_octave_penalty=0.25,
        fast_transition_multiplier=1.2,
        preferred_fret_min=0,
        preferred_fret_max=7,
    ),
    "compact": FingeringProfile(
        fret_weight=0.04,
        high_fret_weight=0.25,
        position_shift_weight=1.50,
        string_change_weight=0.90,
        string_skip_weight=0.60,
        open_string_penalty=0.50,
        open_in_run_penalty=3.0,
        open_in_octave_penalty=2.0,
        same_string_run_reward=0.75,
        repeated_string_change_penalty=3.0,
        octave_shape_reward=2.0,
        nonstandard_octave_penalty=1.5,
        fast_transition_multiplier=2.0,
        preferred_fret_min=2,
        preferred_fret_max=14,
    ),
}


def generate_candidates(
    pitch: int,
    *,
    strings: Sequence[tuple[int, int]] = BEADG_STRINGS,
    fret_count: int = 24,
) -> tuple[FingeringCandidate, ...]:
    """Return every playable string/fret location for a MIDI pitch."""
    return tuple(
        FingeringCandidate(string, fret)
        for string, open_pitch in strings
        if 0 <= (fret := pitch - open_pitch) <= fret_count
    )


def optimize_fingering(
    events: Sequence[FingeringEvent],
    profile_name: str,
    *,
    strings: Sequence[tuple[int, int]] = BEADG_STRINGS,
) -> list[FingeringCandidate]:
    """Choose a globally low-cost fingering path using dynamic programming."""
    if not events:
        return []
    try:
        profile = FINGERING_PROFILES[profile_name]
    except KeyError as error:
        available = ", ".join(sorted(FINGERING_PROFILES))
        raise ValueError(
            f"unknown fingering profile {profile_name!r}; choose from {available}"
        ) from error

    candidates = [generate_candidates(event.pitch, strings=strings) for event in events]
    for event, options in zip(events, candidates, strict=True):
        if not options:
            raise ValueError(f"MIDI pitch {event.pitch} is outside five-string BEADG range")

    costs: list[list[float]] = []
    backpointers: list[list[int]] = []
    costs.append([_placement_cost(events, 0, candidate, profile) for candidate in candidates[0]])
    backpointers.append([-1] * len(candidates[0]))

    for index in range(1, len(events)):
        current_costs: list[float] = []
        current_backpointers: list[int] = []
        for current in candidates[index]:
            alternatives = [
                previous_cost
                + _transition_cost(
                    events[index - 1],
                    previous,
                    events[index],
                    current,
                    profile,
                )
                for previous_cost, previous in zip(
                    costs[index - 1], candidates[index - 1], strict=True
                )
            ]
            best_previous = min(range(len(alternatives)), key=alternatives.__getitem__)
            current_costs.append(
                alternatives[best_previous] + _placement_cost(events, index, current, profile)
            )
            current_backpointers.append(best_previous)
        costs.append(current_costs)
        backpointers.append(current_backpointers)

    selected = min(range(len(costs[-1])), key=costs[-1].__getitem__)
    path: list[FingeringCandidate] = []
    for index in range(len(events) - 1, -1, -1):
        path.append(candidates[index][selected])
        selected = backpointers[index][selected]
    path.reverse()
    return path


def _placement_cost(
    events: Sequence[FingeringEvent],
    index: int,
    candidate: FingeringCandidate,
    profile: FingeringProfile,
) -> float:
    cost = candidate.fret * profile.fret_weight
    if candidate.fret > profile.preferred_fret_max:
        cost += (candidate.fret - profile.preferred_fret_max) * profile.high_fret_weight
    elif 0 < candidate.fret < profile.preferred_fret_min:
        cost += (profile.preferred_fret_min - candidate.fret) * profile.high_fret_weight
    if candidate.fret == 0:
        cost += profile.open_string_penalty
        neighboring_steps = (
            index > 0 and abs(events[index].pitch - events[index - 1].pitch) <= 2
        ) or (index + 1 < len(events) and abs(events[index + 1].pitch - events[index].pitch) <= 2)
        if neighboring_steps:
            cost += profile.open_in_run_penalty
        neighboring_octave = (
            index > 0 and abs(events[index].pitch - events[index - 1].pitch) == 12
        ) or (index + 1 < len(events) and abs(events[index + 1].pitch - events[index].pitch) == 12)
        if neighboring_octave:
            cost += profile.open_in_octave_penalty
    return cost


def _transition_cost(
    previous_event: FingeringEvent,
    previous: FingeringCandidate,
    current_event: FingeringEvent,
    current: FingeringCandidate,
    profile: FingeringProfile,
) -> float:
    fret_shift = abs(current.fret - previous.fret)
    string_distance = abs(current.string - previous.string)
    cost = fret_shift * profile.position_shift_weight
    if string_distance:
        cost += profile.string_change_weight
        cost += max(0, string_distance - 1) * profile.string_skip_weight

    interval = current_event.pitch - previous_event.pitch
    if abs(interval) <= 2 and string_distance == 0:
        cost -= profile.same_string_run_reward
    if interval == 0 and string_distance:
        cost += profile.repeated_string_change_penalty

    if abs(interval) == 12:
        lower, higher = (previous, current) if interval > 0 else (current, previous)
        conventional_octave = higher.string == lower.string - 2 and higher.fret == lower.fret + 2
        if conventional_octave:
            cost -= profile.octave_shape_reward
        else:
            cost += profile.nonstandard_octave_penalty

    # One slot is a 32nd note. Fast passages magnify awkward movement.
    slot_gap = max(0, current_event.start_slot - previous_event.start_slot)
    if slot_gap <= 4:
        movement_cost = (
            fret_shift * profile.position_shift_weight
            + string_distance * profile.string_change_weight
        )
        cost += movement_cost * (profile.fast_transition_multiplier - 1.0)
    return cost
