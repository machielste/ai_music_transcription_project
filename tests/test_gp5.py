from pathlib import Path

import guitarpro

from bass_transcriber.export.gp5 import _tempo_schedule, write_gp5
from bass_transcriber.models import BassNote, RhythmGrid


def test_gp5_round_trips_five_string_tuning_and_notes(tmp_path: Path) -> None:
    output = tmp_path / "bass.gp5"
    grid = RhythmGrid(
        detector="test",
        bpm=120.0,
        beats_per_bar=None,
        first_downbeat_seconds=None,
        beat_times_seconds=(0.0, 0.5, 1.0),
        onset_delay_seconds=0.0,
    )
    notes = [
        BassNote(25, 0.0, 0.25, "electric_bass"),
        BassNote(28, 0.25, 0.75, "electric_bass"),
    ]

    write_gp5(output, notes, grid, title="Test")

    song = guitarpro.parse(str(output))
    track = song.tracks[0]
    assert [string.value for string in track.strings] == [43, 38, 33, 28, 23]
    sounding = [
        note
        for measure in track.measures
        for beat in measure.voices[0].beats
        for note in beat.notes
        if note.type.name == "normal"
    ]
    assert [(note.string, note.value) for note in sounding] == [(5, 2), (4, 0)]


def test_fractional_tempo_schedule_has_bounded_cumulative_error() -> None:
    target_bpm = 117.454
    schedule = _tempo_schedule(target_bpm, 132)

    assert set(schedule) == {117, 118}
    target_bar_seconds = 240.0 / target_bpm
    elapsed = 0.0
    errors: list[float] = []
    for measure_index, tempo in enumerate(schedule):
        elapsed += 240.0 / tempo
        errors.append(elapsed - (measure_index + 1) * target_bar_seconds)

    assert max(abs(error) for error in errors) < 0.01
