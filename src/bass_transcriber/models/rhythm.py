"""Neutral rhythm representations used by quantization and notation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RhythmGrid:
    """Detected beat/downbeat grid, independent of the detector library."""

    detector: str
    bpm: float
    beats_per_bar: int | None
    first_downbeat_seconds: float | None
    beat_times_seconds: tuple[float, ...]
    onset_delay_seconds: float

    def __post_init__(self) -> None:
        if self.bpm <= 0:
            raise ValueError("BPM must be positive")
        if self.beats_per_bar is not None and self.beats_per_bar < 1:
            raise ValueError("beats_per_bar must be positive when present")
        if len(self.beat_times_seconds) < 2:
            raise ValueError("a rhythm grid requires at least two beats")
