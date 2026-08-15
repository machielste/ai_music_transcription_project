"""End-to-end song processing used by user interfaces."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from bass_transcriber.export.gp5 import write_gp5
from bass_transcriber.export.json import write_notes_json
from bass_transcriber.export.rhythm_json import write_rhythm_json
from bass_transcriber.rhythm import detect_rhythm
from bass_transcriber.transcription.muscriptor import transcribe_bass

ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True, slots=True)
class ProcessingResult:
    """Summary of a completed end-to-end transcription."""

    output: Path
    note_count: int
    bpm: float
    copied_source: Path | None = None


def output_path_for(source: Path, destination: Path) -> Path:
    """Return the final GP5 path for a source and destination folder."""
    return destination / f"{source.stem}.bass.gp5"


def process_song(
    source: Path,
    destination: Path,
    *,
    copy_source: bool = False,
    progress: ProgressCallback | None = None,
) -> ProcessingResult:
    """Run the current production pipeline and copy out one final GP5 file."""
    if not source.is_file():
        raise FileNotFoundError(f"music file does not exist: {source}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is not available on PATH")

    destination.mkdir(parents=True, exist_ok=True)
    final_output = output_path_for(source, destination)
    _notify(progress, 0.01, "Preparing audio")

    with tempfile.TemporaryDirectory(prefix="bass-transcriber-") as temporary:
        work = Path(temporary)
        wav = work / "source.wav"
        _convert_to_wav(ffmpeg, source, wav)

        _notify(progress, 0.05, "Loading MuScriptor large model")

        def transcription_progress(completed: int, total: int) -> None:
            fraction = completed / total if total else 0.0
            _notify(
                progress,
                0.05 + 0.72 * fraction,
                f"Transcribing audio chunk {completed}/{total}",
            )

        notes = transcribe_bass(
            wav,
            size="large",
            instrument=None,
            device="cuda",
            progress=transcription_progress,
        )
        notes_json = work / "notes.json"
        write_notes_json(
            notes_json,
            notes,
            source=wav,
            model_size="large",
            instrument_mode="auto",
        )

        _notify(progress, 0.80, "Detecting tempo and beat grid")
        rhythm = detect_rhythm(
            wav,
            note_onsets=[note.start_seconds for note in notes],
            device="cuda",
            detector="auto",
        )
        write_rhythm_json(work / "rhythm.json", rhythm, source=wav)

        _notify(progress, 0.93, "Writing synchronized five-string GP5")
        temporary_gp5 = work / "result.gp5"
        write_gp5(temporary_gp5, notes, rhythm, title=f"{source.stem} - Bass")
        shutil.copy2(temporary_gp5, final_output)

    copied_source: Path | None = None
    source_copy = destination / source.name
    if copy_source and source.resolve() != source_copy.resolve():
        _notify(progress, 0.99, f"Copying original audio: {source.name}")
        shutil.copy2(source, source_copy)
        copied_source = source_copy

    _notify(progress, 1.0, f"Finished: {final_output.name}")
    return ProcessingResult(final_output, len(notes), rhythm.bpm, copied_source)


def _convert_to_wav(ffmpeg: str, source: Path, output: Path) -> None:
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-c:a",
        "pcm_s24le",
        str(output),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or "unknown FFmpeg error"
        raise RuntimeError(f"audio conversion failed: {detail}") from error


def _notify(callback: ProgressCallback | None, fraction: float, message: str) -> None:
    if callback is not None:
        callback(fraction, message)
