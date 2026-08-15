from pathlib import Path

import pytest

import bass_transcriber.pipeline as pipeline
from bass_transcriber.export.gp5 import GP5ExportResult
from bass_transcriber.models import BassNote, RhythmGrid
from bass_transcriber.pipeline import output_path_for


def test_output_path_uses_source_stem() -> None:
    assert output_path_for(Path("music/song.mp3"), Path("exports")) == Path("exports/song.bass.gp5")


def test_process_song_runs_pipeline_and_copies_gp5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mp3"
    source.write_bytes(b"audio")
    destination = tmp_path / "exports"
    progress: list[tuple[float, str]] = []
    notes = [BassNote(28, 0.0, 0.25, "electric_bass")]
    rhythm = RhythmGrid(
        detector="test",
        bpm=117.454,
        beats_per_bar=4,
        first_downbeat_seconds=0.0,
        beat_times_seconds=(0.0, 0.51),
        onset_delay_seconds=0.0,
    )

    monkeypatch.setattr(
        pipeline.shutil, "which", lambda name: "ffmpeg" if name == "ffmpeg" else None
    )
    monkeypatch.setattr(
        pipeline,
        "_convert_to_wav",
        lambda ffmpeg, input_path, output_path: output_path.write_bytes(b"wav"),
    )
    monkeypatch.setattr(pipeline, "transcribe_bass", lambda *args, **kwargs: notes)
    monkeypatch.setattr(pipeline, "detect_rhythm", lambda *args, **kwargs: rhythm)
    monkeypatch.setattr(
        pipeline,
        "write_gp5",
        lambda output, *args, **kwargs: (
            output.write_bytes(b"gp5"),
            GP5ExportResult(1, ()),
        )[1],
    )

    result = pipeline.process_song(
        source,
        destination,
        progress=lambda fraction, message: progress.append((fraction, message)),
    )

    assert result.output == destination / "source.bass.gp5"
    assert result.output.read_bytes() == b"gp5"
    assert result.note_count == 1
    assert result.bpm == pytest.approx(117.454)
    assert progress[0] == (0.01, "Preparing audio")
    assert progress[-1] == (1.0, "Finished: source.bass.gp5")


def test_process_song_optionally_copies_original_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mp3"
    source.write_bytes(b"original audio")
    destination = tmp_path / "exports"
    notes = [BassNote(28, 0.0, 0.25, "electric_bass")]
    rhythm = RhythmGrid(
        detector="test",
        bpm=120.0,
        beats_per_bar=4,
        first_downbeat_seconds=0.0,
        beat_times_seconds=(0.0, 0.5),
        onset_delay_seconds=0.0,
    )

    monkeypatch.setattr(pipeline.shutil, "which", lambda name: "ffmpeg")
    monkeypatch.setattr(
        pipeline,
        "_convert_to_wav",
        lambda ffmpeg, input_path, output_path: output_path.write_bytes(b"wav"),
    )
    monkeypatch.setattr(pipeline, "transcribe_bass", lambda *args, **kwargs: notes)
    monkeypatch.setattr(pipeline, "detect_rhythm", lambda *args, **kwargs: rhythm)
    monkeypatch.setattr(
        pipeline,
        "write_gp5",
        lambda output, *args, **kwargs: (
            output.write_bytes(b"gp5"),
            GP5ExportResult(1, ()),
        )[1],
    )

    result = pipeline.process_song(source, destination, copy_source=True)

    assert result.copied_source == destination / "source.mp3"
    assert result.copied_source.read_bytes() == b"original audio"


def test_process_song_surfaces_dropped_note_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mp3"
    source.write_bytes(b"audio")
    destination = tmp_path / "exports"
    notes = [BassNote(28, 0.0, 0.25, "electric_bass")]
    rhythm = RhythmGrid(
        detector="test",
        bpm=120.0,
        beats_per_bar=4,
        first_downbeat_seconds=0.0,
        beat_times_seconds=(0.0, 0.5),
        onset_delay_seconds=0.0,
    )

    monkeypatch.setattr(pipeline.shutil, "which", lambda name: "ffmpeg")
    monkeypatch.setattr(
        pipeline,
        "_convert_to_wav",
        lambda ffmpeg, input_path, output_path: output_path.write_bytes(b"wav"),
    )
    monkeypatch.setattr(pipeline, "transcribe_bass", lambda *args, **kwargs: notes)
    monkeypatch.setattr(pipeline, "detect_rhythm", lambda *args, **kwargs: rhythm)
    monkeypatch.setattr(
        pipeline,
        "write_gp5",
        lambda output, *args, **kwargs: (
            output.write_bytes(b"gp5"),
            GP5ExportResult(1, (71, 71, 72)),
        )[1],
    )

    result = pipeline.process_song(source, destination)

    assert result.warnings == (
        "Dropped 3 notes outside the 5-string BEADG range. MIDI pitches: 71 (2x), 72.",
    )
