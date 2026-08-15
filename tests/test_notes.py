import pytest

from bass_transcriber.models import BassNote


def test_note_rejects_an_end_before_its_start() -> None:
    with pytest.raises(ValueError, match="must not precede"):
        BassNote(
            pitch=40,
            start_seconds=1.0,
            end_seconds=0.5,
            instrument="electric_bass",
        )
