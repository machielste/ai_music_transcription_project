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

    write_gp5(output, notes, grid, title="Test", fingering_profile=None)

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


def test_gp5_replaces_metadata_characters_not_supported_by_legacy_encoding(
    tmp_path: Path,
) -> None:
    output = tmp_path / "unicode-title.gp5"
    grid = RhythmGrid(
        detector="test",
        bpm=120.0,
        beats_per_bar=4,
        first_downbeat_seconds=0.0,
        beat_times_seconds=(0.0, 0.5),
        onset_delay_seconds=0.0,
    )

    write_gp5(
        output,
        [BassNote(28, 0.0, 0.25, "electric_bass")],
        grid,
        title="测试 – Bass",
    )

    song = guitarpro.parse(str(output))
    assert song.title == "?? – Bass"


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


def test_gp5_drops_out_of_range_notes_and_reports_them(tmp_path: Path) -> None:
    output = tmp_path / "bass.gp5"
    grid = RhythmGrid(
        detector="test",
        bpm=120.0,
        beats_per_bar=4,
        first_downbeat_seconds=0.0,
        beat_times_seconds=(0.0, 0.5),
        onset_delay_seconds=0.0,
    )
    notes = [
        BassNote(28, 0.0, 0.25, "electric_bass"),
        BassNote(71, 0.25, 0.5, "electric_bass"),
    ]

    result = write_gp5(output, notes, grid, title="Test")

    assert output.is_file()
    assert result.exported_note_count == 1
    assert result.dropped_pitches == (71,)


def test_gp5_four_string_mode_excludes_b_string_and_drops_notes_below_e(
    tmp_path: Path,
) -> None:
    output = tmp_path / "four-string.gp5"
    grid = RhythmGrid(
        detector="test",
        bpm=120.0,
        beats_per_bar=4,
        first_downbeat_seconds=0.0,
        beat_times_seconds=(0.0, 0.5),
        onset_delay_seconds=0.0,
    )
    notes = [
        BassNote(23, 0.0, 0.25, "electric_bass"),
        BassNote(28, 0.25, 0.5, "electric_bass"),
    ]

    result = write_gp5(output, notes, grid, title="Test", five_string=False)
    song = guitarpro.parse(str(output))

    assert [string.value for string in song.tracks[0].strings] == [43, 38, 33, 28]
    assert result.exported_note_count == 1
    assert result.dropped_pitches == (23,)


def test_gp5_reduces_an_impossible_chord_to_the_contextually_closest_pitch(
    tmp_path: Path,
) -> None:
    output = tmp_path / "contaminated-chord.gp5"
    grid = RhythmGrid(
        detector="test",
        bpm=120.0,
        beats_per_bar=4,
        first_downbeat_seconds=0.0,
        beat_times_seconds=(0.0, 0.5, 1.0),
        onset_delay_seconds=0.0,
    )
    notes = [
        BassNote(38, 0.0, 0.25, "electric_bass"),
        BassNote(50, 0.5, 0.75, "electric_bass"),
        BassNote(55, 0.5, 0.75, "electric_bass"),
        BassNote(59, 0.5, 0.75, "electric_bass"),
        BassNote(62, 0.5, 0.75, "electric_bass"),
        BassNote(67, 0.5, 0.75, "electric_bass"),
        BassNote(38, 1.0, 1.25, "electric_bass"),
    ]

    result = write_gp5(
        output,
        notes,
        grid,
        title="Contaminated chord",
        five_string=False,
    )
    song = guitarpro.parse(str(output))
    sounding_pitches = [
        note.realValue
        for measure in song.tracks[0].measures
        for beat in measure.voices[0].beats
        for note in beat.notes
        if note.type.name == "normal"
    ]

    assert sounding_pitches == [38, 50, 38]
    assert result.exported_note_count == 3
    assert len(result.pruned_chords) == 1
    assert result.pruned_chords[0].original_pitches == (50, 55, 59, 62, 67)
    assert result.pruned_chords[0].kept_pitch == 50
    assert result.pruned_chords[0].removed_pitches == (55, 59, 62, 67)
    assert result.pruned_chords[0].reason == "more_notes_than_strings"


def test_gp5_writes_simultaneous_octaves_as_a_tied_chord_across_measures(
    tmp_path: Path,
) -> None:
    output = tmp_path / "octave-chord.gp5"
    grid = RhythmGrid(
        detector="test",
        bpm=120.0,
        beats_per_bar=4,
        first_downbeat_seconds=0.0,
        beat_times_seconds=(0.0, 0.5),
        onset_delay_seconds=0.0,
    )
    notes = [
        BassNote(38, 0.0, 2.5, "electric_bass"),
        BassNote(50, 0.0, 2.5, "electric_bass"),
    ]

    write_gp5(output, notes, grid, title="Octave chord", fingering_profile=None)
    song = guitarpro.parse(str(output))
    first_measure = song.tracks[0].measures[0]
    second_measure = song.tracks[0].measures[1]
    first_chord = first_measure.voices[0].beats[0]
    tied_chord = second_measure.voices[0].beats[0]

    assert sorted(note.realValue for note in first_chord.notes) == [38, 50]
    assert {note.type.name for note in first_chord.notes} == {"normal"}
    assert sorted(note.realValue for note in tied_chord.notes) == [38, 50]
    assert {note.type.name for note in tied_chord.notes} == {"tie"}
    assert len({note.string for note in first_chord.notes}) == 2


def test_gp5_ties_only_the_longer_tone_when_chord_durations_differ(
    tmp_path: Path,
) -> None:
    output = tmp_path / "split-chord.gp5"
    grid = RhythmGrid(
        detector="test",
        bpm=120.0,
        beats_per_bar=4,
        first_downbeat_seconds=0.0,
        beat_times_seconds=(0.0, 0.5),
        onset_delay_seconds=0.0,
    )
    notes = [
        BassNote(43, 0.0, 0.5, "electric_bass"),
        BassNote(47, 0.0, 1.0, "electric_bass"),
    ]

    write_gp5(output, notes, grid, title="Split chord", fingering_profile=None)
    song = guitarpro.parse(str(output))
    sounding_beats = [
        beat
        for beat in song.tracks[0].measures[0].voices[0].beats
        if beat.notes
    ]

    assert sorted(note.realValue for note in sounding_beats[0].notes) == [43, 47]
    assert len({note.string for note in sounding_beats[0].notes}) == 2
    assert [(note.realValue, note.type.name) for note in sounding_beats[1].notes] == [
        (47, "tie")
    ]
