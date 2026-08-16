import json
from pathlib import Path

import pytest

import bass_transcriber.pipeline as pipeline
from bass_transcriber.export.gp5 import GP5ExportResult
from bass_transcriber.models import BassNote, RhythmGrid
from bass_transcriber.pipeline import debug_path_for, output_path_for


def test_output_path_uses_source_stem() -> None:
    assert output_path_for(Path("music/song.mp3"), Path("exports")) == Path("exports/song.bass.gp5")
    assert debug_path_for(Path("music/song.mp3"), Path("exports")) == Path(
        "exports/song.bass.debug.json"
    )


def test_process_song_runs_pipeline_and_copies_gp5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mp3"
    source.write_bytes(b"audio")
    destination = tmp_path / "exports"
    progress: list[tuple[float, str]] = []
    transcription_kwargs: dict[str, object] = {}
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
    def transcribe(*args: object, **kwargs: object) -> list[BassNote]:
        transcription_kwargs.update(kwargs)
        return notes

    monkeypatch.setattr(pipeline, "transcribe_bass", transcribe)
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
    assert result.debug_log == destination / "source.bass.debug.json"
    debug_document = json.loads(result.debug_log.read_text(encoding="utf-8"))
    assert debug_document["schema_version"] == 1
    assert debug_document["document_type"] == "bass_transcriber_debug_run"
    assert debug_document["status"] == "succeeded"
    assert debug_document["source"]["sha256"]
    assert debug_document["configuration"]["instrument_mode"] == "auto"
    assert debug_document["export"]["exported_note_count"] == 1
    assert debug_document["result"]["note_count"] == 1
    assert all(stage["status"] == "succeeded" for stage in debug_document["stages"])
    assert transcription_kwargs["instrument"] is None
    assert transcription_kwargs["trace"] is not None
    assert progress[0] == (0.01, "Preparing audio")
    assert progress[-1] == (1.0, "Finished: source.bass.gp5")


def test_process_song_can_force_electric_bass_conditioning(
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
    transcription_kwargs: dict[str, object] = {}

    monkeypatch.setattr(pipeline.shutil, "which", lambda name: "ffmpeg")
    monkeypatch.setattr(
        pipeline,
        "_convert_to_wav",
        lambda ffmpeg, input_path, output_path: output_path.write_bytes(b"wav"),
    )

    def transcribe(*args: object, **kwargs: object) -> list[BassNote]:
        transcription_kwargs.update(kwargs)
        return notes

    monkeypatch.setattr(pipeline, "transcribe_bass", transcribe)
    monkeypatch.setattr(pipeline, "detect_rhythm", lambda *args, **kwargs: rhythm)
    monkeypatch.setattr(
        pipeline,
        "write_gp5",
        lambda output, *args, **kwargs: (
            output.write_bytes(b"gp5"),
            GP5ExportResult(1, ()),
        )[1],
    )

    pipeline.process_song(source, destination, force_electric_bass=True)

    assert transcription_kwargs["instrument"] == "electric_bass"


def test_process_song_optionally_merges_false_sustained_retriggers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mp3"
    source.write_bytes(b"audio")
    destination = tmp_path / "exports"
    raw_notes = [
        BassNote(40, 0.0, 1.0, "electric_bass"),
        BassNote(40, 1.0, 2.0, "electric_bass"),
    ]
    merged_notes = (BassNote(40, 0.0, 2.0, "electric_bass"),)
    rhythm = RhythmGrid(
        detector="test",
        bpm=120.0,
        beats_per_bar=4,
        first_downbeat_seconds=0.0,
        beat_times_seconds=(0.0, 0.5),
        onset_delay_seconds=0.0,
    )
    exported_notes: list[BassNote] = []

    monkeypatch.setattr(pipeline.shutil, "which", lambda name: "ffmpeg")
    monkeypatch.setattr(
        pipeline,
        "_convert_to_wav",
        lambda ffmpeg, input_path, output_path: output_path.write_bytes(b"wav"),
    )
    monkeypatch.setattr(pipeline, "transcribe_bass", lambda *args, **kwargs: raw_notes)
    monkeypatch.setattr(pipeline, "detect_rhythm", lambda *args, **kwargs: rhythm)

    class FakeCleanup:
        notes = merged_notes

        @staticmethod
        def diagnostics() -> dict[str, object]:
            return {
                "enabled": True,
                "input_note_count": 2,
                "output_note_count": 1,
                "merged_boundary_count": 1,
            }

    monkeypatch.setattr(
        pipeline,
        "merge_false_retriggers",
        lambda audio, notes: FakeCleanup(),
    )

    def write_gp5(
        output: Path,
        notes: list[BassNote],
        *args: object,
        **kwargs: object,
    ) -> GP5ExportResult:
        exported_notes.extend(notes)
        output.write_bytes(b"gp5")
        return GP5ExportResult(len(notes), ())

    monkeypatch.setattr(pipeline, "write_gp5", write_gp5)

    result = pipeline.process_song(
        source,
        destination,
        merge_sustained_retriggers=True,
    )

    assert exported_notes == list(merged_notes)
    assert result.note_count == 1
    debug_document = json.loads(result.debug_log.read_text(encoding="utf-8"))
    assert debug_document["configuration"]["merge_sustained_retriggers"] is True
    assert debug_document["postprocessing"]["merged_boundary_count"] == 1


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


def test_process_song_retains_structured_debug_log_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mp3"
    source.write_bytes(b"audio")
    destination = tmp_path / "exports"

    monkeypatch.setattr(pipeline.shutil, "which", lambda name: "ffmpeg")
    monkeypatch.setattr(
        pipeline,
        "_convert_to_wav",
        lambda ffmpeg, input_path, output_path: output_path.write_bytes(b"wav"),
    )

    def fail_transcription(*args: object, **kwargs: object) -> list[BassNote]:
        raise RuntimeError("synthetic model failure")

    monkeypatch.setattr(pipeline, "transcribe_bass", fail_transcription)

    with pytest.raises(RuntimeError, match="synthetic model failure"):
        pipeline.process_song(source, destination)

    debug_log = destination / "source.bass.debug.json"
    document = json.loads(debug_log.read_text(encoding="utf-8"))
    assert document["status"] == "failed"
    assert document["error"]["type"] == "RuntimeError"
    assert document["error"]["message"] == "synthetic model failure"
    transcription_stage = next(
        stage for stage in document["stages"] if stage["name"] == "transcription"
    )
    assert transcription_stage["status"] == "failed"
