"""Beat and downbeat detection from source audio."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import torch
from muscriptor.utils.beats import BeatGrid, detect_grid  # type: ignore[import-untyped]
from soundfile import read  # type: ignore[import-untyped]

from bass_transcriber.models import RhythmGrid

RhythmDetector = Literal["auto", "beat_this", "librosa"]


def detect_rhythm(
    audio: Path,
    *,
    note_onsets: list[float],
    device: str = "cuda",
    detector: RhythmDetector = "auto",
) -> RhythmGrid:
    """Detect a constant-tempo grid and measure transcription onset lag."""
    samples, sample_rate = read(str(audio), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(samples.T.copy())
    selected = _select_detector(detector)
    if selected == "beat_this":
        detected = detect_grid(waveform, int(sample_rate), device=device)
    else:
        detected = _detect_with_librosa(samples, int(sample_rate))
    aligned = detected.with_onset_delay(note_onsets)
    if aligned.beats is None:
        raise RuntimeError("beat detector returned no beat positions")
    return RhythmGrid(
        detector=selected,
        bpm=float(aligned.bpm),
        beats_per_bar=aligned.beats_per_bar,
        first_downbeat_seconds=(float(aligned.first_downbeat) if selected == "beat_this" else None),
        beat_times_seconds=tuple(float(beat) for beat in aligned.beats),
        onset_delay_seconds=float(aligned.onset_delay or 0.0),
    )


def _select_detector(requested: RhythmDetector) -> Literal["beat_this", "librosa"]:
    if requested != "auto":
        return requested
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / "beat_this-final0.ckpt"
    if checkpoint.is_file() and checkpoint.stat().st_size > 0:
        return "beat_this"
    return "librosa"


def _detect_with_librosa(samples: np.ndarray, sample_rate: int) -> BeatGrid:
    mono = samples.mean(axis=1)
    tempo, beat_times = librosa.beat.beat_track(
        y=mono,
        sr=sample_rate,
        units="time",
    )
    beats = np.asarray(beat_times, dtype=float)
    if len(beats) < 2:
        raise RuntimeError("librosa could not detect a usable beat sequence")
    bpm = float(np.asarray(tempo).reshape(-1)[0])
    return BeatGrid(
        bpm=bpm,
        beats_per_bar=None,
        first_downbeat=float(beats[0]),
        beats=beats,
    )
