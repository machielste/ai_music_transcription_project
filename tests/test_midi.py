from pathlib import Path

from mido import MidiFile

from bass_transcriber.export.midi import write_midi
from bass_transcriber.models import BassNote


def test_midi_preserves_timing_and_rearticulates_repeated_notes(tmp_path: Path) -> None:
    output = tmp_path / "notes.mid"
    notes = [
        BassNote(28, 0.0, 0.25, "electric_bass"),
        BassNote(28, 0.25, 0.5, "electric_bass"),
    ]

    write_midi(output, notes)

    midi = MidiFile(output)
    messages = [message for message in midi.tracks[0] if not message.is_meta]
    assert messages[0].type == "program_change"
    assert messages[0].program == 33
    assert [message.type for message in messages[1:]] == [
        "note_on",
        "note_off",
        "note_on",
        "note_off",
    ]
    assert [message.time for message in messages[1:]] == [0, 480, 0, 480]
    assert midi.length == 0.5
