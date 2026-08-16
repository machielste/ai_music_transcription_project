from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent

from bass_transcriber.transcription.muscriptor import (
    MODEL_SIZES,
    TranscriptionTrace,
    notes_from_events,
)


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
    trace = TranscriptionTrace()

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
        trace=trace,
    )

    assert [note.pitch for note in notes] == [40, 43]
    assert all(note.confidence is None for note in notes)
    assert progress == [(0, 1), (1, 1)]
    diagnostics = trace.as_dict()
    assert diagnostics["event_type_counts"] == {
        "progress": 2,
        "note_start": 3,
        "note_end": 3,
    }
    assert diagnostics["instrument_note_start_counts"] == {
        "electric_bass": 2,
        "piano": 1,
    }
    assert diagnostics["chunks"] == [
        {
            "chunk_index": 0,
            "nominal_start_seconds": 0.0,
            "nominal_end_seconds": 5.0,
            "event_count": 6,
            "note_start_count": 3,
            "accepted_bass_note_start_count": 2,
            "instrument_note_start_counts": {"electric_bass": 2, "piano": 1},
            "has_no_note_events": False,
        }
    ]
    rejected = [
        event
        for event in diagnostics["events"]
        if event["type"] == "note_end" and event["decision"] == "rejected"
    ]
    assert rejected[0]["rejection_reason"] == "non_bass_instrument"
