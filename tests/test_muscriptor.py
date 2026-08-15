from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent

from bass_transcriber.transcription.muscriptor import MODEL_SIZES, notes_from_events


def test_supported_model_sizes_are_explicit() -> None:
    assert MODEL_SIZES == ("small", "medium", "large")


def test_event_stream_is_converted_to_sorted_neutral_notes() -> None:
    later = NoteStartEvent(
        pitch=43,
        start_time=1.0,
        index=0,
        instrument="electric_bass",
    )
    earlier = NoteStartEvent(
        pitch=40,
        start_time=0.25,
        index=1,
        instrument="electric_bass",
    )
    piano = NoteStartEvent(
        pitch=60,
        start_time=0.5,
        index=2,
        instrument="piano",
    )
    progress: list[tuple[int, int]] = []

    notes = notes_from_events(
        [
            ProgressEvent(completed=0, total=1),
            later,
            NoteEndEvent(end_time=1.5, start_event=later),
            earlier,
            NoteEndEvent(end_time=0.75, start_event=earlier),
            piano,
            NoteEndEvent(end_time=1.0, start_event=piano),
            ProgressEvent(completed=1, total=1),
        ],
        progress=lambda completed, total: progress.append((completed, total)),
    )

    assert [note.pitch for note in notes] == [40, 43]
    assert all(note.confidence is None for note in notes)
    assert progress == [(0, 1), (1, 1)]
