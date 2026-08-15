import pytest

from bass_transcriber.models import RhythmGrid


def test_rhythm_grid_requires_multiple_beats() -> None:
    with pytest.raises(ValueError, match="at least two beats"):
        RhythmGrid(
            detector="test",
            bpm=120.0,
            beats_per_bar=4,
            first_downbeat_seconds=0.0,
            beat_times_seconds=(0.0,),
            onset_delay_seconds=0.0,
        )
