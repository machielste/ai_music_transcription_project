import json
from pathlib import Path

import guitarpro
import pytest

from bass_transcriber.export.fingering_json import (
    read_fingering_json,
    write_fingering_json,
)
from bass_transcriber.models import RhythmGrid
from bass_transcriber.pipeline import ProcessingResult, export_gp5_from_fingering_draft
from bass_transcriber.tab import EADG_STRINGS, FingeringTimeline, ResolvedTabNote


def _grid() -> RhythmGrid:
    return RhythmGrid(
        detector="test",
        bpm=120.0,
        beats_per_bar=4,
        first_downbeat_seconds=0.0,
        beat_times_seconds=(0.0, 0.5),
        onset_delay_seconds=0.0,
    )


def _write_draft(path: Path, source: Path) -> None:
    write_fingering_json(
        path,
        FingeringTimeline((ResolvedTabNote(38, 0, 4, 2, 0),), ()),
        source=source,
        title="Editable bass",
        profile="balanced",
        strings=EADG_STRINGS,
        rhythm=_grid(),
    )


def test_fingering_json_round_trips_resolved_notes(tmp_path: Path) -> None:
    source = tmp_path / "song.wav"
    output = tmp_path / "song.bass.fingering.json"

    _write_draft(output, source)
    document = read_fingering_json(output)

    assert document.source == source.resolve()
    assert document.title == "Editable bass"
    assert document.profile == "balanced"
    assert document.strings == EADG_STRINGS
    assert document.rhythm == _grid()
    assert document.notes == (ResolvedTabNote(38, 0, 4, 2, 0),)


def test_fingering_json_rejects_a_string_fret_that_changes_pitch(tmp_path: Path) -> None:
    source = tmp_path / "song.wav"
    output = tmp_path / "song.bass.fingering.json"
    _write_draft(output, source)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["notes"][0]["string"] = 3
    payload["notes"][0]["fret"] = 4
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="expected fret 5"):
        read_fingering_json(output)


def test_fingering_json_rejects_changes_to_pitch_or_timing(tmp_path: Path) -> None:
    source = tmp_path / "song.wav"
    output = tmp_path / "song.bass.fingering.json"
    _write_draft(output, source)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["notes"][0]["pitch"] = 39
    payload["notes"][0]["fret"] = 1
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="immutable content changed"):
        read_fingering_json(output)


def test_edited_fingering_json_is_the_authority_for_gp5_export(tmp_path: Path) -> None:
    source = tmp_path / "song.wav"
    source.write_bytes(b"audio")
    draft = tmp_path / "song.bass.fingering.json"
    output = tmp_path / "song.bass.gp5"
    _write_draft(draft, source)
    payload = json.loads(draft.read_text(encoding="utf-8"))
    payload["notes"][0]["string"] = 3
    payload["notes"][0]["fret"] = 5
    draft.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    result = ProcessingResult(
        output=output,
        note_count=1,
        bpm=120.0,
        debug_log=tmp_path / "debug.json",
        source=source,
        rhythm=_grid(),
        five_string=False,
        fingering_draft=draft,
    )

    export_result = export_gp5_from_fingering_draft(result)
    song = guitarpro.parse(str(output))
    sounding = [
        note
        for measure in song.tracks[0].measures
        for beat in measure.voices[0].beats
        for note in beat.notes
        if note.type.name == "normal"
    ]

    assert export_result.exported_note_count == 1
    assert [(note.string, note.value, note.realValue) for note in sounding] == [(3, 5, 38)]
