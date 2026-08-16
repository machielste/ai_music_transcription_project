"""End-to-end song processing used by user interfaces."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from bass_transcriber.debug import DebugRun, analyze_audio_file
from bass_transcriber.export.gp5 import write_gp5
from bass_transcriber.export.json import write_notes_json
from bass_transcriber.export.rhythm_json import write_rhythm_json
from bass_transcriber.models import BassInstrument
from bass_transcriber.rhythm import detect_rhythm
from bass_transcriber.transcription.muscriptor import TranscriptionTrace, transcribe_bass

ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True, slots=True)
class ProcessingResult:
    """Summary of a completed end-to-end transcription."""

    output: Path
    note_count: int
    bpm: float
    debug_log: Path
    copied_source: Path | None = None
    warnings: tuple[str, ...] = ()


def output_path_for(source: Path, destination: Path) -> Path:
    """Return the final GP5 path for a source and destination folder."""
    return destination / f"{source.stem}.bass.gp5"


def debug_path_for(source: Path, destination: Path) -> Path:
    """Return the persistent machine-readable debug-log path for a run."""
    return destination / f"{source.stem}.bass.debug.json"


def process_song(
    source: Path,
    destination: Path,
    *,
    copy_source: bool = False,
    force_electric_bass: bool = False,
    fingering_profile: str | None = "balanced",
    five_string: bool = True,
    progress: ProgressCallback | None = None,
) -> ProcessingResult:
    """Run the current production pipeline and copy out one final GP5 file."""
    if not source.is_file():
        raise FileNotFoundError(f"music file does not exist: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    final_output = output_path_for(source, destination)
    final_debug_log = debug_path_for(source, destination)
    ffmpeg = shutil.which("ffmpeg")
    instrument: BassInstrument | None = "electric_bass" if force_electric_bass else None
    instrument_mode = "electric_bass" if force_electric_bass else "auto"
    debug_run = DebugRun(
        source,
        final_output,
        configuration={
            "copy_source": copy_source,
            "force_electric_bass": force_electric_bass,
            "instrument_mode": instrument_mode,
            "model_size": "large",
            "transcription_device": "cuda",
            "rhythm_device": "cuda",
            "rhythm_detector_requested": "auto",
            "fingering_profile": fingering_profile,
            "five_string": five_string,
            "tuning": "BEADG" if five_string else "EADG",
        },
        ffmpeg_path=ffmpeg,
    )
    transcription_trace = TranscriptionTrace()

    try:
        if ffmpeg is None:
            raise RuntimeError("FFmpeg is not available on PATH")
        _notify(progress, 0.01, "Preparing audio")
        with tempfile.TemporaryDirectory(prefix="bass-transcriber-") as temporary:
            work = Path(temporary)
            wav = work / "source.wav"
            with debug_run.stage("audio_conversion"):
                _convert_to_wav(ffmpeg, source, wav)
            with debug_run.stage("audio_analysis"):
                debug_run.set_section("audio", analyze_audio_file(wav))

            mode_description = (
                " with electric-bass conditioning" if force_electric_bass else ""
            )
            _notify(progress, 0.05, f"Loading MuScriptor large model{mode_description}")

            def transcription_progress(completed: int, total: int) -> None:
                fraction = completed / total if total else 0.0
                _notify(
                    progress,
                    0.05 + 0.72 * fraction,
                    f"Transcribing audio chunk {completed}/{total}",
                )

            with debug_run.stage("transcription"):
                notes = transcribe_bass(
                    wav,
                    size="large",
                    instrument=instrument,
                    device="cuda",
                    progress=transcription_progress,
                    trace=transcription_trace,
                )
            debug_run.set_section("transcription", transcription_trace.as_dict())

            with debug_run.stage("temporary_note_artifact"):
                notes_json = work / "notes.json"
                write_notes_json(
                    notes_json,
                    notes,
                    source=wav,
                    model_size="large",
                    instrument_mode=instrument_mode,
                )

            _notify(progress, 0.80, "Detecting tempo and beat grid")
            with debug_run.stage("rhythm_detection"):
                rhythm = detect_rhythm(
                    wav,
                    note_onsets=[note.start_seconds for note in notes],
                    device="cuda",
                    detector="auto",
                )
            debug_run.set_section("rhythm", asdict(rhythm))
            with debug_run.stage("temporary_rhythm_artifact"):
                write_rhythm_json(work / "rhythm.json", rhythm, source=wav)

            string_description = "five-string" if five_string else "four-string"
            _notify(progress, 0.93, f"Writing synchronized {string_description} GP5")
            temporary_gp5 = work / "result.gp5"
            with debug_run.stage("gp5_export"):
                export_result = write_gp5(
                    temporary_gp5,
                    notes,
                    rhythm,
                    title=f"{source.stem} - Bass",
                    fingering_profile=fingering_profile,
                    five_string=five_string,
                )
            debug_run.set_section(
                "export",
                {
                    "input_note_count": len(notes),
                    "exported_note_count": export_result.exported_note_count,
                    "dropped_note_count": len(export_result.dropped_pitches),
                    "dropped_pitches": list(export_result.dropped_pitches),
                },
            )
            with debug_run.stage("gp5_copy_to_destination"):
                shutil.copy2(temporary_gp5, final_output)

        pipeline_warnings: list[str] = []
        if export_result.dropped_pitches:
            pitch_counts = {
                pitch: export_result.dropped_pitches.count(pitch)
                for pitch in sorted(set(export_result.dropped_pitches))
            }
            pitch_summary = ", ".join(
                f"{pitch} ({count}x)" if count > 1 else str(pitch)
                for pitch, count in pitch_counts.items()
            )
            count = len(export_result.dropped_pitches)
            tuning_name = "BEADG" if five_string else "EADG"
            string_count = 5 if five_string else 4
            warning_message = (
                f"Dropped {count} note{'s' if count != 1 else ''} outside the "
                f"{string_count}-string {tuning_name} range. "
                f"MIDI pitch{'es' if len(pitch_counts) != 1 else ''}: "
                f"{pitch_summary}."
            )
            pipeline_warnings.append(warning_message)
            debug_run.add_warning(
                warning_message,
                category="gp5_range_filter",
                details={"pitch_counts": pitch_counts},
            )

        copied_source: Path | None = None
        source_copy = destination / source.name
        if copy_source and source.resolve() != source_copy.resolve():
            _notify(progress, 0.99, f"Copying original audio: {source.name}")
            with debug_run.stage("source_copy"):
                shutil.copy2(source, source_copy)
            copied_source = source_copy

        debug_run.set_section(
            "result",
            {
                "note_count": len(notes),
                "bpm": rhythm.bpm,
                "copied_source": str(copied_source.resolve()) if copied_source else None,
                "warnings": pipeline_warnings,
            },
        )
        debug_run.finish_success()
        debug_run.write(final_debug_log)
        _notify(progress, 1.0, f"Finished: {final_output.name}")
        return ProcessingResult(
            final_output,
            len(notes),
            rhythm.bpm,
            final_debug_log,
            copied_source,
            tuple(pipeline_warnings),
        )
    except Exception as error:
        debug_run.set_section("transcription", transcription_trace.as_dict())
        debug_run.finish_failure(error)
        try:
            debug_run.write(final_debug_log)
        except Exception as debug_error:
            error.add_note(f"Additionally failed to write debug log: {debug_error}")
        raise


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
