"""Monophonic-first string, fret, and hand-position optimization for bass."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

# Guitar Pro numbers strings from highest to lowest.
BEADG_STRINGS: tuple[tuple[int, int], ...] = (
    (1, 43),  # G2
    (2, 38),  # D2
    (3, 33),  # A1
    (4, 28),  # E1
    (5, 23),  # B0
)
EADG_STRINGS: tuple[tuple[int, int], ...] = BEADG_STRINGS[:-1]

_HAND_SPAN_FRETS = 4
_REPEATED_PHRASE_NOTES = 6


@dataclass(frozen=True, slots=True)
class FingeringEvent:
    """The timing and pitch information needed by the optimizer."""

    pitch: int
    start_slot: int
    end_slot: int


@dataclass(frozen=True, slots=True)
class FingeringCandidate:
    """One playable string/fret location for a pitch."""

    string: int
    fret: int


@dataclass(frozen=True, slots=True)
class FingeringProfile:
    """Weights controlling the shared ergonomic model."""

    fret_weight: float
    high_fret_weight: float
    position_shift_weight: float
    string_change_weight: float
    string_skip_weight: float
    open_string_penalty: float
    open_in_run_penalty: float
    same_string_run_reward: float
    repeated_string_change_penalty: float
    octave_shape_reward: float
    nonstandard_octave_penalty: float
    fast_shift_weight: float
    rest_shift_discount: float
    hand_stretch_weight: float
    preferred_fret_min: int
    preferred_fret_max: int
    upper_b_string_penalty: float = 12.0


# These are the only profiles offered for new transcriptions. They share the
# same ergonomic model; avoid_open changes only the strength of the open-string
# preference. Old names are accepted below so existing saved results can still
# be re-exported.
FINGERING_PROFILES: dict[str, FingeringProfile] = {
    "balanced": FingeringProfile(
        fret_weight=0.04,
        high_fret_weight=0.30,
        position_shift_weight=0.75,
        string_change_weight=0.90,
        string_skip_weight=0.55,
        open_string_penalty=0.50,
        open_in_run_penalty=1.0,
        same_string_run_reward=0.45,
        repeated_string_change_penalty=2.0,
        octave_shape_reward=1.5,
        nonstandard_octave_penalty=0.75,
        fast_shift_weight=1.8,
        rest_shift_discount=3.0,
        hand_stretch_weight=0.08,
        preferred_fret_min=1,
        preferred_fret_max=12,
    ),
    "avoid_open": FingeringProfile(
        fret_weight=0.04,
        high_fret_weight=0.30,
        position_shift_weight=0.75,
        string_change_weight=0.90,
        string_skip_weight=0.55,
        open_string_penalty=6.0,
        open_in_run_penalty=2.0,
        same_string_run_reward=0.45,
        repeated_string_change_penalty=2.0,
        octave_shape_reward=1.5,
        nonstandard_octave_penalty=0.75,
        fast_shift_weight=1.8,
        rest_shift_discount=3.0,
        hand_stretch_weight=0.08,
        preferred_fret_min=1,
        preferred_fret_max=12,
    ),
}

_LEGACY_PROFILE_ALIASES: dict[str, str] = {
    "slap_funk": "balanced",
    "low_positions": "balanced",
    "compact": "balanced",
}


@dataclass(frozen=True, slots=True)
class _SearchState:
    candidate: FingeringCandidate
    hand_position: int


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
    seconds_per_slot: float = 0.0625,
    enforce_repeated_phrases: bool = True,
) -> list[FingeringCandidate]:
    """Choose a playable path while tracking the fretting hand independently.

    Open strings deliberately retain a hand-position state: playing fret zero
    does not require moving the fretting hand to the nut. Exact repeated
    pitch/rhythm phrases are locked to one candidate pattern in a second pass.
    """
    if not events:
        return []
    if seconds_per_slot <= 0:
        raise ValueError("seconds_per_slot must be positive")

    resolved_profile_name = _LEGACY_PROFILE_ALIASES.get(profile_name, profile_name)
    try:
        profile = FINGERING_PROFILES[resolved_profile_name]
    except KeyError as error:
        available = ", ".join((*sorted(FINGERING_PROFILES), *_LEGACY_PROFILE_ALIASES))
        raise ValueError(
            f"unknown fingering profile {profile_name!r}; choose from {available}"
        ) from error

    if (5, 23) not in strings:
        profile = replace(profile, upper_b_string_penalty=0.0)

    states = _generate_search_states(events, strings)
    base_path, _ = _search(events, states, profile, seconds_per_slot)
    if not enforce_repeated_phrases:
        return [state.candidate for state in base_path]

    locks = _repeated_phrase_locks(events, base_path)
    if not locks:
        return [state.candidate for state in base_path]

    locked_states = [
        tuple(
            state
            for state in event_states
            if index not in locks or state.candidate == locks[index]
        )
        for index, event_states in enumerate(states)
    ]
    repeated_path, _ = _search(events, locked_states, profile, seconds_per_slot)
    return [state.candidate for state in repeated_path]


def _generate_search_states(
    events: Sequence[FingeringEvent],
    strings: Sequence[tuple[int, int]],
) -> list[tuple[_SearchState, ...]]:
    candidates = [generate_candidates(event.pitch, strings=strings) for event in events]
    for event, options in zip(events, candidates, strict=True):
        if not options:
            tuning_name = "BEADG" if len(strings) == 5 else "EADG"
            raise ValueError(f"MIDI pitch {event.pitch} is outside {tuning_name} range")

    hand_positions = sorted(
        {
            hand_position
            for options in candidates
            for candidate in options
            if candidate.fret > 0
            for hand_position in _reachable_hand_positions(candidate.fret)
        }
    )
    if not hand_positions:
        hand_positions = [1]

    states: list[tuple[_SearchState, ...]] = []
    for options in candidates:
        event_states: list[_SearchState] = []
        for candidate in options:
            positions = (
                hand_positions
                if candidate.fret == 0
                else _reachable_hand_positions(candidate.fret)
            )
            event_states.extend(_SearchState(candidate, position) for position in positions)
        states.append(tuple(event_states))
    return states


def _reachable_hand_positions(fret: int) -> range:
    lowest = max(1, fret - (_HAND_SPAN_FRETS - 1))
    return range(lowest, fret + 1)


def _search(
    events: Sequence[FingeringEvent],
    states: Sequence[Sequence[_SearchState]],
    profile: FingeringProfile,
    seconds_per_slot: float,
) -> tuple[list[_SearchState], float]:
    if any(not event_states for event_states in states):
        raise ValueError("fingering constraints removed every candidate for an event")

    costs: list[list[float]] = [
        [_placement_cost(events, 0, state, profile) for state in states[0]]
    ]
    backpointers: list[list[int]] = [[-1] * len(states[0])]

    for index in range(1, len(events)):
        current_costs: list[float] = []
        current_backpointers: list[int] = []
        for current in states[index]:
            alternatives = [
                previous_cost
                + _transition_cost(
                    events[index - 1],
                    previous,
                    events[index],
                    current,
                    profile,
                    seconds_per_slot,
                )
                for previous_cost, previous in zip(
                    costs[index - 1], states[index - 1], strict=True
                )
            ]
            best_previous = min(range(len(alternatives)), key=alternatives.__getitem__)
            current_costs.append(
                alternatives[best_previous]
                + _placement_cost(events, index, current, profile)
            )
            current_backpointers.append(best_previous)
        costs.append(current_costs)
        backpointers.append(current_backpointers)

    selected = min(range(len(costs[-1])), key=costs[-1].__getitem__)
    total_cost = costs[-1][selected]
    path: list[_SearchState] = []
    for index in range(len(events) - 1, -1, -1):
        path.append(states[index][selected])
        selected = backpointers[index][selected]
    path.reverse()
    return path, total_cost


def _placement_cost(
    events: Sequence[FingeringEvent],
    index: int,
    state: _SearchState,
    profile: FingeringProfile,
) -> float:
    candidate = state.candidate
    if candidate.fret == 0:
        cost = profile.open_string_penalty
        if _is_scalar_neighbor(events, index):
            cost += profile.open_in_run_penalty
        return cost

    cost = candidate.fret * profile.fret_weight
    cost += upper_b_string_cost(candidate, profile)
    if candidate.fret > profile.preferred_fret_max:
        cost += (candidate.fret - profile.preferred_fret_max) * profile.high_fret_weight
    elif candidate.fret < profile.preferred_fret_min:
        cost += (profile.preferred_fret_min - candidate.fret) * profile.high_fret_weight
    finger_offset = candidate.fret - state.hand_position
    cost += finger_offset * profile.hand_stretch_weight
    return cost


def upper_b_string_cost(
    candidate: FingeringCandidate, profile: FingeringProfile,
) -> float:
    """Prefer the low B string for notes below open E, without banning positions."""
    if candidate.string == 5 and candidate.fret >= 5:
        return profile.upper_b_string_penalty
    return 0.0


def _is_scalar_neighbor(events: Sequence[FingeringEvent], index: int) -> bool:
    pitch = events[index].pitch
    neighboring_intervals = []
    if index > 0:
        neighboring_intervals.append(abs(pitch - events[index - 1].pitch))
    if index + 1 < len(events):
        neighboring_intervals.append(abs(events[index + 1].pitch - pitch))
    return any(0 < interval <= 2 for interval in neighboring_intervals)


def _transition_cost(
    previous_event: FingeringEvent,
    previous: _SearchState,
    current_event: FingeringEvent,
    current: _SearchState,
    profile: FingeringProfile,
    seconds_per_slot: float,
) -> float:
    position_shift = abs(current.hand_position - previous.hand_position)
    string_distance = abs(current.candidate.string - previous.candidate.string)
    onset_seconds = max(
        seconds_per_slot,
        (current_event.start_slot - previous_event.start_slot) * seconds_per_slot,
    )
    rest_seconds = max(
        0.0,
        (current_event.start_slot - previous_event.end_slot) * seconds_per_slot,
    )

    rapid_fraction = max(0.0, (0.30 - onset_seconds) / 0.30)
    timing_multiplier = 1.0 + rapid_fraction * profile.fast_shift_weight
    rest_discount = 1.0 + rest_seconds * profile.rest_shift_discount
    cost = (
        position_shift
        * profile.position_shift_weight
        * timing_multiplier
        / rest_discount
    )

    if position_shift:
        frets_per_second = position_shift / onset_seconds
        if frets_per_second > 18.0:
            cost += (frets_per_second - 18.0) ** 2 * 0.02

    if string_distance:
        string_cost = profile.string_change_weight
        string_cost += max(0, string_distance - 1) * profile.string_skip_weight
        cost += string_cost * (1.0 + rapid_fraction * 0.35)

    interval = current_event.pitch - previous_event.pitch
    if 0 < abs(interval) <= 2 and string_distance == 0:
        cost -= profile.same_string_run_reward
    if interval == 0 and string_distance:
        cost += profile.repeated_string_change_penalty

    if (
        0 < abs(interval) <= 2
        and string_distance
        and (previous.candidate.fret == 0 or current.candidate.fret == 0)
    ):
        cost += profile.open_in_run_penalty

    if (
        abs(interval) == 12
        and previous.candidate.fret > 0
        and current.candidate.fret > 0
    ):
        lower, higher = (
            (previous.candidate, current.candidate)
            if interval > 0
            else (current.candidate, previous.candidate)
        )
        conventional_octave = (
            higher.string == lower.string - 2
            and higher.fret == lower.fret + 2
        )
        if conventional_octave:
            cost -= profile.octave_shape_reward
        else:
            cost += profile.nonstandard_octave_penalty
    return cost


def _repeated_phrase_locks(
    events: Sequence[FingeringEvent],
    path: Sequence[_SearchState],
) -> dict[int, FingeringCandidate]:
    phrase_length = _REPEATED_PHRASE_NOTES
    if len(events) < phrase_length * 2:
        return {}
    if len({event.start_slot for event in events}) != len(events):
        return {}

    occurrences: defaultdict[tuple[tuple[int, int, int], ...], list[int]] = defaultdict(list)
    for start in range(len(events) - phrase_length + 1):
        key = _phrase_key(events, start, phrase_length)
        if len({token[0] for token in key}) >= 2:
            occurrences[key].append(start)

    parents = list(range(len(events)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for starts in occurrences.values():
        if len(starts) < 2:
            continue
        reference = starts[0]
        for start in starts[1:]:
            for offset in range(phrase_length):
                union(reference + offset, start + offset)

    equivalent_indices: defaultdict[int, list[int]] = defaultdict(list)
    for index in range(len(events)):
        equivalent_indices[find(index)].append(index)

    locks: dict[int, FingeringCandidate] = {}
    for indices in equivalent_indices.values():
        if len(indices) < 2:
            continue
        counts = Counter(path[index].candidate for index in indices)
        canonical = min(
            counts,
            key=lambda candidate: (
                -counts[candidate],
                candidate.fret,
                candidate.string,
            ),
        )
        locks.update(dict.fromkeys(indices, canonical))
    return locks


def _phrase_key(
    events: Sequence[FingeringEvent],
    start: int,
    length: int,
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (
            event.pitch,
            event.end_slot - event.start_slot,
            (
                events[index + 1].start_slot - event.start_slot
                if index + 1 < start + length
                else 0
            ),
        )
        for index, event in enumerate(
            events[start : start + length],
            start=start,
        )
    )


def normalize_profile_name(profile_name: str) -> str:
    """Map a saved legacy profile name to a currently supported policy."""
    return _LEGACY_PROFILE_ALIASES.get(profile_name, profile_name)


def supported_profile_names() -> Mapping[str, FingeringProfile]:
    """Return the profiles intended for new user selections."""
    return FINGERING_PROFILES
