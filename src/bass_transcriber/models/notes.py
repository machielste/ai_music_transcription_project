"""Neutral note representation used between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BassInstrument = Literal["electric_bass", "acoustic_bass"]


@dataclass(frozen=True, slots=True)
class BassNote:
    """One raw, unquantized bass note emitted by a transcription backend."""

    pitch: int
    start_seconds: float
    end_seconds: float
    instrument: BassInstrument
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.pitch <= 127:
            raise ValueError(f"MIDI pitch must be between 0 and 127, got {self.pitch}")
        if self.start_seconds < 0:
            raise ValueError("note start must not be negative")
        if self.end_seconds < self.start_seconds:
            raise ValueError("note end must not precede note start")
