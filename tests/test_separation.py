import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from bass_transcriber import separation


def test_worker_output_is_validated_cached_and_invalidated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = tmp_path / "source.wav"
    output = tmp_path / "bass.wav"
    sf.write(audio, np.zeros((2205, 2)), 22050)
    calls = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert command[:5] == [separation.sys.executable, "-X", "utf8", "-m",
                               "bass_transcriber.separation"]
        sf.write(command[6], np.zeros((4410, 2)), 44100, subtype="FLOAT")
        Path(command[7]).write_text(json.dumps({
            "model": separation.MODEL, "package_version": separation.PACKAGE_VERSION,
            "weights_sha256": separation.WEIGHTS_SHA256,
            "config_sha256": separation.CONFIG_SHA256, "device": "cuda",
        }))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(separation.subprocess, "run", run)
    first = separation.separate_bass(audio, output)
    assert first["reused"] is False
    assert separation.separate_bass(audio, output)["reused"] is True
    assert len(calls) == 1
    # Modified output must never be accepted as a matching cached stem.
    sf.write(output, np.ones((4410, 2)) * .1, 44100, subtype="FLOAT")
    assert separation.separate_bass(audio, output)["reused"] is False
    assert len(calls) == 2
    # A changed source also invalidates the cache.
    sf.write(audio, np.ones((2205, 2)) * .1, 22050)
    assert separation.separate_bass(audio, output)["reused"] is False
    assert len(calls) == 3


def test_separation_worker_failure_is_reported_without_a_stem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"audio")
    output = tmp_path / "bass.wav"

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(separation.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="Bass separation failed"):
        separation.separate_bass(audio, output)
    assert not output.exists()
    assert output.with_suffix(".separation.log").is_file()


@pytest.mark.parametrize("invalid", ["duration", "nan"])
def test_stem_validation_rejects_corrupt_or_misaligned_audio(
    tmp_path: Path, invalid: str,
) -> None:
    audio, stem = tmp_path / "source.wav", tmp_path / "stem.wav"
    sf.write(audio, np.zeros((4410, 2)), 44100, subtype="FLOAT")
    values = np.zeros((4400 if invalid == "duration" else 4410, 2))
    if invalid == "nan":
        values[5, 0] = np.nan
    sf.write(stem, values, 44100, subtype="FLOAT")
    with pytest.raises(RuntimeError, match="duration|invalid audio"):
        separation.validate_stem(audio, stem)
