from pathlib import Path

import numpy as np
import pytest
import soundfile  # type: ignore[import-untyped]

from bass_transcriber.debug import analyze_audio_file


def test_audio_analysis_reports_signal_levels_per_model_chunk(tmp_path: Path) -> None:
    audio_path = tmp_path / "levels.wav"
    sample_rate = 100
    samples = np.concatenate(
        (
            np.zeros(sample_rate * 5, dtype=np.float32),
            np.full(sample_rate * 2, 0.5, dtype=np.float32),
        )
    )
    soundfile.write(str(audio_path), samples, sample_rate, subtype="FLOAT")

    diagnostics = analyze_audio_file(audio_path)

    assert diagnostics["sample_rate"] == sample_rate
    assert diagnostics["channels"] == 1
    assert diagnostics["duration_seconds"] == pytest.approx(7.0)
    chunks = diagnostics["chunks"]
    assert isinstance(chunks, list)
    assert len(chunks) == 2
    assert chunks[0]["rms_amplitude"] == 0.0
    assert chunks[1]["rms_amplitude"] == pytest.approx(0.5)
