"""Spectral cleanup for false same-pitch note re-articulations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import librosa
import numpy as np
from numpy.typing import NDArray

from bass_transcriber.models import BassNote

_ANALYSIS_SAMPLE_RATE = 16_000
_ANALYSIS_RADIUS_SECONDS = 0.30
_FFT_SIZE = 2048
_WINDOW_SIZE = 1024
_HOP_SIZE = 80
_MAX_HARMONIC_FREQUENCY_HZ = 1200.0
_DEFAULT_MAX_GAP_SECONDS = 0.02
_DEFAULT_MINIMUM_FRAGMENT_SECONDS = 0.50
_DEFAULT_ENERGY_RISE_THRESHOLD_DB = 3.0
_DEFAULT_SPECTRAL_FLUX_THRESHOLD_DB = 2.5


@dataclass(frozen=True, slots=True)
class AttackMetrics:
    """Pitch-local evidence for a fresh attack at one candidate boundary."""

    energy_rise_db: float
    spectral_flux_db: float


@dataclass(frozen=True, slots=True)
class RetriggerDecision:
    """One inspected same-pitch boundary and the resulting cleanup action."""

    pitch: int
    instrument: str
    boundary_seconds: float
    gap_seconds: float
    previous_start_seconds: float
    previous_end_seconds: float
    next_start_seconds: float
    next_end_seconds: float
    energy_rise_db: float
    spectral_flux_db: float
    action: str
    reason: str


@dataclass(frozen=True, slots=True)
class RetriggerCleanupResult:
    """Cleaned notes plus serializable diagnostics for the optional pass."""

    notes: tuple[BassNote, ...]
    decisions: tuple[RetriggerDecision, ...]
    max_gap_seconds: float
    minimum_fragment_seconds: float
    energy_rise_threshold_db: float
    spectral_flux_threshold_db: float

    def diagnostics(self) -> dict[str, object]:
        merged_count = sum(decision.action == "merged" for decision in self.decisions)
        return {
            "enabled": True,
            "algorithm": "pitch_conditioned_stft_v1",
            "input_note_count": len(self.notes) + merged_count,
            "output_note_count": len(self.notes),
            "candidate_boundary_count": len(self.decisions),
            "merged_boundary_count": merged_count,
            "preserved_attack_count": len(self.decisions) - merged_count,
            "parameters": {
                "analysis_sample_rate": _ANALYSIS_SAMPLE_RATE,
                "analysis_radius_seconds": _ANALYSIS_RADIUS_SECONDS,
                "fft_size": _FFT_SIZE,
                "window_size": _WINDOW_SIZE,
                "hop_size": _HOP_SIZE,
                "max_harmonic_frequency_hz": _MAX_HARMONIC_FREQUENCY_HZ,
                "max_gap_seconds": self.max_gap_seconds,
                "minimum_fragment_seconds": self.minimum_fragment_seconds,
                "energy_rise_threshold_db": self.energy_rise_threshold_db,
                "spectral_flux_threshold_db": self.spectral_flux_threshold_db,
            },
            "decisions": [asdict(decision) for decision in self.decisions],
        }


AttackAnalyzer = Callable[[int, float], AttackMetrics]


def merge_false_retriggers(
    audio: Path,
    notes: list[BassNote],
    *,
    max_gap_seconds: float = _DEFAULT_MAX_GAP_SECONDS,
    minimum_fragment_seconds: float = _DEFAULT_MINIMUM_FRAGMENT_SECONDS,
    energy_rise_threshold_db: float = _DEFAULT_ENERGY_RISE_THRESHOLD_DB,
    spectral_flux_threshold_db: float = _DEFAULT_SPECTRAL_FLUX_THRESHOLD_DB,
    attack_analyzer: AttackAnalyzer | None = None,
) -> RetriggerCleanupResult:
    """Merge contiguous same-pitch notes when the audio has no fresh attack.

    Only boundaries between sustained successive notes of the same pitch and
    instrument are candidates. A boundary is retained when either its pitch-local
    energy rise or coherent harmonic spectral flux crosses its conservative
    threshold. The minimum fragment duration protects ordinary short repeated
    notes from this experimental cleanup.
    """
    if max_gap_seconds < 0:
        raise ValueError("maximum retrigger gap must not be negative")
    if minimum_fragment_seconds < 0:
        raise ValueError("minimum retrigger fragment duration must not be negative")

    grouped: dict[tuple[str, int], list[BassNote]] = defaultdict(list)
    for note in sorted(notes, key=_note_sort_key):
        grouped[(note.instrument, note.pitch)].append(note)

    has_candidates = any(
        _is_candidate(
            previous,
            following,
            max_gap_seconds=max_gap_seconds,
            minimum_fragment_seconds=minimum_fragment_seconds,
        )
        for pitch_notes in grouped.values()
        for previous, following in zip(pitch_notes, pitch_notes[1:], strict=False)
    )
    if attack_analyzer is None and has_candidates:
        attack_analyzer = _PitchConditionedSpectrogram(audio)

    cleaned: list[BassNote] = []
    decisions: list[RetriggerDecision] = []
    for pitch_notes in grouped.values():
        current = pitch_notes[0]
        previous_source = current
        for following in pitch_notes[1:]:
            gap_seconds = following.start_seconds - previous_source.end_seconds
            if not _is_candidate(
                previous_source,
                following,
                max_gap_seconds=max_gap_seconds,
                minimum_fragment_seconds=minimum_fragment_seconds,
            ):
                cleaned.append(current)
                current = following
                previous_source = following
                continue

            if attack_analyzer is None:
                raise RuntimeError("missing spectral analyzer for retrigger candidate")
            metrics = attack_analyzer(following.pitch, following.start_seconds)
            has_attack = (
                metrics.energy_rise_db >= energy_rise_threshold_db
                or metrics.spectral_flux_db >= spectral_flux_threshold_db
            )
            if has_attack:
                action = "preserved"
                reason = "pitch_local_attack_detected"
                cleaned.append(current)
                current = following
            else:
                action = "merged"
                reason = "no_pitch_local_attack"
                current = replace(
                    current,
                    end_seconds=max(current.end_seconds, following.end_seconds),
                    confidence=_merged_confidence(current.confidence, following.confidence),
                )

            decisions.append(
                RetriggerDecision(
                    pitch=following.pitch,
                    instrument=following.instrument,
                    boundary_seconds=following.start_seconds,
                    gap_seconds=gap_seconds,
                    previous_start_seconds=previous_source.start_seconds,
                    previous_end_seconds=previous_source.end_seconds,
                    next_start_seconds=following.start_seconds,
                    next_end_seconds=following.end_seconds,
                    energy_rise_db=metrics.energy_rise_db,
                    spectral_flux_db=metrics.spectral_flux_db,
                    action=action,
                    reason=reason,
                )
            )
            previous_source = following
        cleaned.append(current)

    return RetriggerCleanupResult(
        notes=tuple(sorted(cleaned, key=_note_sort_key)),
        decisions=tuple(sorted(decisions, key=lambda decision: decision.boundary_seconds)),
        max_gap_seconds=max_gap_seconds,
        minimum_fragment_seconds=minimum_fragment_seconds,
        energy_rise_threshold_db=energy_rise_threshold_db,
        spectral_flux_threshold_db=spectral_flux_threshold_db,
    )


class _PitchConditionedSpectrogram:
    """Analyze short STFT windows without retaining a full-song spectrogram."""

    def __init__(self, audio: Path) -> None:
        samples, _ = librosa.load(
            audio,
            sr=_ANALYSIS_SAMPLE_RATE,
            mono=True,
            dtype=np.float32,
        )
        self.samples: NDArray[np.float32] = samples
        self.radius_samples = round(_ANALYSIS_RADIUS_SECONDS * _ANALYSIS_SAMPLE_RATE)
        self.frequencies = librosa.fft_frequencies(
            sr=_ANALYSIS_SAMPLE_RATE,
            n_fft=_FFT_SIZE,
        )

    def __call__(self, pitch: int, boundary_seconds: float) -> AttackMetrics:
        center_sample = round(boundary_seconds * _ANALYSIS_SAMPLE_RATE)
        segment = np.zeros(self.radius_samples * 2, dtype=np.float32)
        source_start = max(0, center_sample - self.radius_samples)
        source_end = min(len(self.samples), center_sample + self.radius_samples)
        target_start = source_start - (center_sample - self.radius_samples)
        target_end = target_start + source_end - source_start
        segment[target_start:target_end] = self.samples[source_start:source_end]

        magnitude = np.abs(
            librosa.stft(
                segment,
                n_fft=_FFT_SIZE,
                hop_length=_HOP_SIZE,
                win_length=_WINDOW_SIZE,
                window="hann",
                center=True,
            )
        )
        harmonic_bins = self._harmonic_bins(pitch)
        harmonic_magnitude = magnitude[harmonic_bins]
        bin_db = 20.0 * np.log10(np.maximum(harmonic_magnitude, 1e-7))
        energy_db = 10.0 * np.log10(
            np.maximum(np.mean(np.square(harmonic_magnitude), axis=0), 1e-12)
        )

        frame_seconds = (
            np.arange(magnitude.shape[1]) * _HOP_SIZE - self.radius_samples
        ) / _ANALYSIS_SAMPLE_RATE
        before = energy_db[(frame_seconds >= -0.12) & (frame_seconds <= -0.04)]
        after = energy_db[(frame_seconds >= 0.01) & (frame_seconds <= 0.10)]
        energy_rise_db = float(np.median(after) - np.median(before))

        frame_differences = np.diff(bin_db, axis=1)
        difference_seconds = (frame_seconds[:-1] + frame_seconds[1:]) / 2.0
        local_differences = frame_differences[
            :,
            (difference_seconds >= -0.04) & (difference_seconds <= 0.06),
        ]
        coherent_positive_flux = np.median(
            np.maximum(local_differences, 0.0),
            axis=0,
        )
        spectral_flux_db = float(np.max(coherent_positive_flux))
        return AttackMetrics(energy_rise_db, spectral_flux_db)

    def _harmonic_bins(self, pitch: int) -> list[int]:
        fundamental_hz = float(librosa.midi_to_hz(pitch))
        result: set[int] = set()
        harmonic = 1
        while fundamental_hz * harmonic <= _MAX_HARMONIC_FREQUENCY_HZ:
            frequency = fundamental_hz * harmonic
            nearest = int(np.argmin(np.abs(self.frequencies - frequency)))
            result.update(
                range(
                    max(1, nearest - 1),
                    min(len(self.frequencies), nearest + 2),
                )
            )
            harmonic += 1
        return sorted(result)


def _merged_confidence(first: float | None, second: float | None) -> float | None:
    if first is None:
        return second
    if second is None:
        return first
    return min(first, second)


def _is_candidate(
    previous: BassNote,
    following: BassNote,
    *,
    max_gap_seconds: float,
    minimum_fragment_seconds: float,
) -> bool:
    gap_seconds = following.start_seconds - previous.end_seconds
    return (
        -max_gap_seconds <= gap_seconds <= max_gap_seconds
        and previous.end_seconds - previous.start_seconds >= minimum_fragment_seconds
        and following.end_seconds - following.start_seconds >= minimum_fragment_seconds
    )


def _note_sort_key(note: BassNote) -> tuple[float, int, float]:
    return note.start_seconds, note.pitch, note.end_seconds
