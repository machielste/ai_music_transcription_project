"""Bass tablature fingering generation and optimization."""

from bass_transcriber.tab.fingering import (
    BEADG_STRINGS,
    EADG_STRINGS,
    FINGERING_PROFILES,
    FingeringCandidate,
    FingeringEvent,
    FingeringProfile,
    generate_candidates,
    optimize_fingering,
)
from bass_transcriber.tab.timeline import (
    FingeringTimeline,
    ResolvedTabNote,
    build_fingering_timeline,
    slot_duration_seconds,
    source_aligned_seconds,
)

__all__ = [
    "BEADG_STRINGS",
    "EADG_STRINGS",
    "FINGERING_PROFILES",
    "FingeringCandidate",
    "FingeringEvent",
    "FingeringProfile",
    "FingeringTimeline",
    "ResolvedTabNote",
    "build_fingering_timeline",
    "generate_candidates",
    "optimize_fingering",
    "slot_duration_seconds",
    "source_aligned_seconds",
]
