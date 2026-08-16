from importlib.resources import files
from pathlib import Path
from urllib.request import urlopen

import pytest

import bass_transcriber.fingering_debug as fingering_debug
from bass_transcriber.fingering_debug import (
    build_comparison_document,
    start_fingering_debugger,
)
from bass_transcriber.models import BassNote, RhythmGrid


def _grid() -> RhythmGrid:
    return RhythmGrid(
        detector="test",
        bpm=120.0,
        beats_per_bar=4,
        first_downbeat_seconds=0.0,
        beat_times_seconds=(0.0, 0.5, 1.0, 1.5),
        onset_delay_seconds=0.0,
    )


def test_comparison_uses_final_fingering_timelines_for_every_profile() -> None:
    notes = [
        BassNote(38, 0.0, 0.125, "electric_bass"),
        BassNote(37, 0.125, 0.25, "electric_bass"),
        BassNote(35, 0.25, 0.375, "electric_bass"),
    ]

    document = build_comparison_document(
        notes,
        _grid(),
        profiles=("balanced", "slap_funk", "legacy"),
    )

    assert document["reference_profile"] == "balanced"
    assert [string["name"] for string in document["strings"]] == [  # type: ignore[index]
        "G",
        "D",
        "A",
        "E",
        "B",
    ]
    strategies = document["strategies"]
    assert isinstance(strategies, list)
    assert [strategy["profile"] for strategy in strategies] == [
        "balanced",
        "slap_funk",
        "legacy",
    ]
    assert [
        (note["string"], note["fret"])
        for note in strategies[1]["notes"]
    ] == [(3, 5), (3, 4), (3, 2)]
    assert strategies[2]["notes"][0]["fret"] == 0
    assert document["playback_notes"] == [
        {"pitch": 38, "start_seconds": 0.0, "end_seconds": 0.125},
        {"pitch": 37, "start_seconds": 0.125, "end_seconds": 0.25},
        {"pitch": 35, "start_seconds": 0.25, "end_seconds": 0.375},
    ]


def test_comparison_supports_four_string_tuning_and_reports_dropped_notes() -> None:
    notes = [
        BassNote(23, 0.0, 0.25, "electric_bass"),
        BassNote(28, 0.25, 0.5, "electric_bass"),
    ]

    document = build_comparison_document(
        notes,
        _grid(),
        profiles=("balanced",),
        five_string=False,
    )

    assert [string["name"] for string in document["strings"]] == [  # type: ignore[index]
        "G",
        "D",
        "A",
        "E",
    ]
    strategies = document["strategies"]
    assert isinstance(strategies, list)
    assert strategies[0]["dropped_pitches"] == [23]
    assert [note["pitch"] for note in strategies[0]["notes"]] == [28]


def test_packaged_highway_ui_is_available() -> None:
    html = (
        files("bass_transcriber.debug_ui")
        .joinpath("fingering_highway.html")
        .read_text(encoding="utf-8")
    )

    assert "<canvas" not in html
    assert "document.createElement('canvas')" in html
    assert "perspective note highway" in html
    assert "function targetFretWindow" in html
    assert "fret - viewStart" in html
    assert "stringPosition" in html
    assert "stringCount - note.string" in html
    assert "comparison.strings.slice().reverse()" in html
    expected_colors = (
        "const STRING_COLORS = "
        "['#f28c28', '#3f8cff', '#f2d23c', '#ef4545', '#168a8a']"
    )
    assert expected_colors in html


def test_debugger_session_can_be_reopened_and_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"placeholder audio")
    opened: list[str] = []
    monkeypatch.setattr(fingering_debug.webbrowser, "open", opened.append)

    session = start_fingering_debugger(
        audio,
        [BassNote(28, 0.0, 0.25, "electric_bass")],
        _grid(),
        profiles=("balanced", "legacy"),
        open_browser=False,
    )
    try:
        assert urlopen(f"{session.url}health", timeout=2).read() == b"ok\n"
        session.open_browser()
        assert opened == [session.url]
    finally:
        session.stop()
