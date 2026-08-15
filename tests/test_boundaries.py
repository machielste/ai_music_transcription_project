from bass_transcriber.models import BassNote
from bass_transcriber.postprocess import clip_notes_to_duration


def test_notes_are_clipped_to_the_audio_boundary() -> None:
    notes = [
        BassNote(28, 9.5, 10.5, "electric_bass"),
        BassNote(33, 10.0, 10.5, "electric_bass"),
    ]

    clipped = clip_notes_to_duration(notes, 10.0)

    assert len(clipped) == 1
    assert clipped[0].start_seconds == 9.5
    assert clipped[0].end_seconds == 10.0
