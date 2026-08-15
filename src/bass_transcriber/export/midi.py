"""Standard MIDI File export for auditory transcription checks."""

from __future__ import annotations

from pathlib import Path

from mido import (  # type: ignore[import-untyped]
    Message,
    MetaMessage,
    MidiFile,
    MidiTrack,
    bpm2tempo,
    second2tick,
)

from bass_transcriber.models import BassInstrument, BassNote

TICKS_PER_BEAT = 960
DEFAULT_TEMPO_BPM = 120.0
DEFAULT_VELOCITY = 96

_PROGRAM_BY_INSTRUMENT: dict[BassInstrument, int] = {
    # Mido uses zero-based General MIDI program numbers.
    "acoustic_bass": 32,
    "electric_bass": 33,
}


def write_midi(
    output: Path,
    notes: list[BassNote],
    *,
    tempo_bpm: float = DEFAULT_TEMPO_BPM,
    velocity: int = DEFAULT_VELOCITY,
) -> None:
    """Write notes to MIDI while preserving their absolute times in seconds.

    The tempo is a timing carrier, not an inferred musical tempo. Conversion
    to ticks and back therefore preserves playback timing independently of the
    chosen value, subject only to sub-millisecond tick rounding.
    """
    if tempo_bpm <= 0:
        raise ValueError("tempo must be positive")
    if not 1 <= velocity <= 127:
        raise ValueError("velocity must be between 1 and 127")

    tempo = int(bpm2tempo(tempo_bpm))
    midi = MidiFile(type=0, ticks_per_beat=TICKS_PER_BEAT)
    track = MidiTrack()
    midi.tracks.append(track)
    track.append(MetaMessage("track_name", name="Bass transcription", time=0))
    track.append(MetaMessage("set_tempo", tempo=tempo, time=0))

    instrument: BassInstrument = notes[0].instrument if notes else "electric_bass"
    track.append(Message("program_change", program=_PROGRAM_BY_INSTRUMENT[instrument], time=0))

    # Priority 0 puts note-offs before note-ons at the same tick. This matters
    # for contiguous repeated slap notes of the same pitch.
    events: list[tuple[int, int, int, Message]] = []
    for sequence, note in enumerate(notes):
        start_tick = round(second2tick(note.start_seconds, TICKS_PER_BEAT, tempo))
        end_tick = round(second2tick(note.end_seconds, TICKS_PER_BEAT, tempo))
        end_tick = max(end_tick, start_tick + 1)
        events.append(
            (start_tick, 1, sequence, Message("note_on", note=note.pitch, velocity=velocity))
        )
        events.append((end_tick, 0, sequence, Message("note_off", note=note.pitch, velocity=0)))

    previous_tick = 0
    for absolute_tick, _, _, message in sorted(events):
        message.time = absolute_tick - previous_tick
        track.append(message)
        previous_tick = absolute_tick

    output.parent.mkdir(parents=True, exist_ok=True)
    midi.save(output)
