from pathlib import Path

import numpy as np
import pytest
import soundfile  # type: ignore[import-untyped]

from bass_transcriber.models import BassNote
from bass_transcriber.postprocess import AttackMetrics, merge_false_retriggers


def test_contiguous_same_pitch_notes_without_an_attack_are_merged() -> None:
    notes = [
        BassNote(40, 0.0, 1.0, "electric_bass"),
        BassNote(40, 1.0, 2.0, "electric_bass"),
    ]

    result = merge_false_retriggers(
        Path("unused.wav"),
        notes,
        attack_analyzer=lambda pitch, boundary: AttackMetrics(0.4, 0.2),
    )

    assert result.notes == (BassNote(40, 0.0, 2.0, "electric_bass"),)
    assert result.decisions[0].action == "merged"
    assert result.diagnostics()["merged_boundary_count"] == 1


def test_contiguous_same_pitch_notes_with_an_attack_are_preserved() -> None:
    notes = [
        BassNote(40, 0.0, 1.0, "electric_bass"),
        BassNote(40, 1.0, 2.0, "electric_bass"),
    ]

    result = merge_false_retriggers(
        Path("unused.wav"),
        notes,
        attack_analyzer=lambda pitch, boundary: AttackMetrics(4.0, 0.2),
    )

    assert result.notes == tuple(notes)
    assert result.decisions[0].action == "preserved"
    assert result.decisions[0].reason == "pitch_local_attack_detected"


def test_different_pitches_and_audible_gaps_are_not_candidates() -> None:
    notes = [
        BassNote(40, 0.0, 1.0, "electric_bass"),
        BassNote(41, 1.0, 2.0, "electric_bass"),
        BassNote(40, 1.1, 2.0, "electric_bass"),
    ]

    result = merge_false_retriggers(Path("unused.wav"), notes)

    assert result.notes == tuple(notes)
    assert result.decisions == ()


def test_short_repeated_notes_are_not_spectral_merge_candidates() -> None:
    notes = [
        BassNote(40, 0.0, 0.25, "electric_bass"),
        BassNote(40, 0.25, 0.5, "electric_bass"),
    ]

    result = merge_false_retriggers(Path("unused.wav"), notes)

    assert result.notes == tuple(notes)
    assert result.decisions == ()


@pytest.mark.parametrize("fresh_attack", [False, True])
def test_pitch_conditioned_spectrogram_distinguishes_a_fresh_attack(
    tmp_path: Path,
    fresh_attack: bool,
) -> None:
    sample_rate = 16_000
    seconds = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    samples = 0.2 * np.sin(2.0 * np.pi * 82.4069 * seconds)
    if fresh_attack:
        samples[round(0.88 * sample_rate) : sample_rate] = 0.0
        attack_samples = round(0.04 * sample_rate)
        samples[sample_rate : sample_rate + attack_samples] *= np.linspace(
            0.0,
            1.0,
            attack_samples,
            dtype=np.float32,
        )
    audio = tmp_path / "bass.wav"
    soundfile.write(str(audio), samples, sample_rate, subtype="FLOAT")
    notes = [
        BassNote(40, 0.2, 1.0, "electric_bass"),
        BassNote(40, 1.0, 1.8, "electric_bass"),
    ]

    result = merge_false_retriggers(audio, notes)

    expected_action = "preserved" if fresh_attack else "merged"
    assert result.decisions[0].action == expected_action
