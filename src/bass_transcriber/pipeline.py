"""End-to-end song processing used by user interfaces."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from bass_transcriber.debug import DebugRun, analyze_audio_file
from bass_transcriber.export.fingering_json import (
    read_fingering_json,
    write_fingering_json,
)
from bass_transcriber.export.gp5 import GP5ExportResult, write_gp5, write_resolved_gp5
from bass_transcriber.export.json import read_notes_json, write_notes_json
from bass_transcriber.export.rhythm_json import write_rhythm_json
from bass_transcriber.models import BassInstrument, BassNote, RhythmGrid
from bass_transcriber.postprocess import merge_false_retriggers
from bass_transcriber.rhythm import detect_rhythm
from bass_transcriber.tab import (
    BEADG_STRINGS,
    EADG_STRINGS,
    build_fingering_timeline,
    source_aligned_seconds,
)
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
    source: Path | None = None
    notes: tuple[BassNote, ...] = ()
    rhythm: RhythmGrid | None = None
    five_string: bool = True
    fingering_profile: str | None = "balanced"
    raw_notes: Path | None = None
    fingering_draft: Path | None = None
    reused_raw_notes: bool = False


def output_path_for(source: Path, destination: Path) -> Path:
    """Return the final GP5 path for a source and destination folder."""
    return destination / f"{source.stem}.bass.gp5"


def debug_path_for(source: Path, destination: Path) -> Path:
    """Return the persistent machine-readable debug-log path for a run."""
    return destination / f"{source.stem}.bass.debug.json"


def raw_notes_path_for(source: Path, destination: Path) -> Path:
    """Return the reusable unprocessed model-output path for a song."""
    return destination / f"{source.stem}.bass.raw.notes.json"


def fingering_path_for(source: Path, destination: Path) -> Path:
    """Return the editable resolved-fingering draft path for a song."""
    return destination / f"{source.stem}.bass.fingering.json"


def process_song(
    source: Path,
    destination: Path,
    *,
    copy_source: bool = False,
    force_electric_bass: bool = False,
    merge_sustained_retriggers: bool = False,
    raw_notes_input: Path | None = None,
    fingering_profile: str | None = "balanced",
    five_string: bool = True,
    generate_gp5: bool = True,
    progress: ProgressCallback | None = None,
) -> ProcessingResult:
    """Run transcription through an editable draft and optionally write GP5."""
    if not source.is_file():
        raise FileNotFoundError(f"music file does not exist: {source}")
    if raw_notes_input is not None and not raw_notes_input.is_file():
        raise FileNotFoundError(f"raw model output does not exist: {raw_notes_input}")
    destination.mkdir(parents=True, exist_ok=True)
    final_output = output_path_for(source, destination)
    final_debug_log = debug_path_for(source, destination)
    final_raw_notes = raw_notes_path_for(source, destination)
    final_fingering_draft = fingering_path_for(source, destination)
    debug_target = final_output if generate_gp5 else final_fingering_draft
    ffmpeg = shutil.which("ffmpeg")
    instrument: BassInstrument | None = "electric_bass" if force_electric_bass else None
    instrument_mode = "electric_bass" if force_electric_bass else "auto"
    debug_run = DebugRun(
        source,
        debug_target,
        configuration={
            "copy_source": copy_source,
            "force_electric_bass": force_electric_bass,
            "merge_sustained_retriggers": merge_sustained_retriggers,
            "raw_notes_input": (
                str(raw_notes_input.resolve()) if raw_notes_input is not None else None
            ),
            "transcription_mode": "reused_raw_output" if raw_notes_input else "muscriptor",
            "instrument_mode": instrument_mode,
            "model_size": "large",
            "transcription_device": "cuda",
            "rhythm_device": "cuda",
            "rhythm_detector_requested": "auto",
            "fingering_profile": fingering_profile,
            "five_string": five_string,
            "tuning": "BEADG" if five_string else "EADG",
            "generate_gp5": generate_gp5,
        },
        ffmpeg_path=ffmpeg,
    )
    transcription_trace = TranscriptionTrace()
    reused_raw_notes = raw_notes_input is not None

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

            if raw_notes_input is not None:
                _notify(progress, 0.05, f"Loading raw model output: {raw_notes_input.name}")
                with debug_run.stage("raw_note_import"):
                    raw_document = read_notes_json(raw_notes_input)
                    raw_notes = raw_document.notes
                model_size = raw_document.model_size
                raw_instrument_mode = raw_document.instrument_mode
                debug_run.set_section(
                    "transcription",
                    {
                        "skipped": True,
                        "mode": "reused_raw_output",
                        "input": str(raw_notes_input.resolve()),
                        "model_size": model_size,
                        "instrument_mode": raw_instrument_mode,
                        "note_count": len(raw_notes),
                    },
                )
            else:
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
                    raw_notes = transcribe_bass(
                        wav,
                        size="large",
                        instrument=instrument,
                        device="cuda",
                        progress=transcription_progress,
                        trace=transcription_trace,
                    )
                debug_run.set_section("transcription", transcription_trace.as_dict())
                model_size = "large"
                raw_instrument_mode = instrument_mode

            with debug_run.stage("raw_note_artifact"):
                write_notes_json(
                    final_raw_notes,
                    raw_notes,
                    source=source,
                    model_size=model_size,
                    instrument_mode=raw_instrument_mode,
                )

            _notify(progress, 0.80, "Detecting tempo and beat grid")
            with debug_run.stage("rhythm_detection"):
                rhythm = detect_rhythm(
                    wav,
                    note_onsets=[note.start_seconds for note in raw_notes],
                    device="cuda",
                    detector="auto",
                )
            debug_run.set_section("rhythm", asdict(rhythm))
            with debug_run.stage("temporary_rhythm_artifact"):
                write_rhythm_json(work / "rhythm.json", rhythm, source=wav)

            notes = raw_notes
            if merge_sustained_retriggers:
                _notify(progress, 0.88, "Checking repeated notes for fresh bass attacks")
                with debug_run.stage("spectral_retrigger_cleanup"):
                    cleanup = merge_false_retriggers(wav, raw_notes)
                    notes = list(cleanup.notes)
                debug_run.set_section("postprocessing", cleanup.diagnostics())
            else:
                debug_run.set_section(
                    "postprocessing",
                    {
                        "enabled": False,
                        "algorithm": "pitch_conditioned_stft_v1",
                        "input_note_count": len(raw_notes),
                        "output_note_count": len(raw_notes),
                    },
                )

            string_description = "five-string" if five_string else "four-string"
            strings = BEADG_STRINGS if five_string else EADG_STRINGS
            title = f"{source.stem} - Bass"
            _notify(progress, 0.93, f"Generating editable {string_description} fingering draft")
            with debug_run.stage("fingering_draft"):
                timeline = build_fingering_timeline(
                    notes,
                    rhythm,
                    fingering_profile,
                    strings=strings,
                )
                write_fingering_json(
                    final_fingering_draft,
                    timeline,
                    source=source,
                    title=title,
                    profile=fingering_profile,
                    strings=strings,
                    rhythm=rhythm,
                )
            export_result = GP5ExportResult(
                len(timeline.notes),
                timeline.dropped_pitches,
                timeline.pruned_chords,
            )
            if generate_gp5:
                _notify(progress, 0.95, f"Writing synchronized {string_description} GP5")
                temporary_gp5 = work / "result.gp5"
                with debug_run.stage("gp5_export"):
                    # Retain this public entry point for existing API callers. The GUI
                    # exports edited drafts through write_resolved_gp5 instead.
                    export_result = write_gp5(
                        temporary_gp5,
                        notes,
                        rhythm,
                        title=title,
                        fingering_profile=fingering_profile,
                        five_string=five_string,
                    )
                with debug_run.stage("gp5_copy_to_destination"):
                    shutil.copy2(temporary_gp5, final_output)
            pruned_chord_details = [
                {
                    **asdict(chord),
                    "approx_source_seconds": round(
                        source_aligned_seconds(chord.start_slot, rhythm),
                        3,
                    ),
                }
                for chord in export_result.pruned_chords
            ]
            debug_run.set_section(
                "export",
                {
                    "input_note_count": len(notes),
                    "exported_note_count": export_result.exported_note_count,
                    "dropped_note_count": len(export_result.dropped_pitches),
                    "dropped_pitches": list(export_result.dropped_pitches),
                    "pruned_chord_count": len(export_result.pruned_chords),
                    "pruned_chords": pruned_chord_details,
                    "fingering_draft": str(final_fingering_draft.resolve()),
                    "gp5_created": generate_gp5,
                },
            )

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

        if export_result.pruned_chords:
            removed_count = sum(
                len(chord.removed_pitches) for chord in export_result.pruned_chords
            )
            chord_summaries = []
            for chord in export_result.pruned_chords:
                approximate_seconds = source_aligned_seconds(chord.start_slot, rhythm)
                removed = ", ".join(str(pitch) for pitch in chord.removed_pitches)
                chord_summaries.append(
                    f"near {approximate_seconds:.2f}s kept MIDI "
                    f"{chord.kept_pitch} and removed {removed}"
                )
            chord_count = len(export_result.pruned_chords)
            pruning_stage = (
                "during GP5 export"
                if generate_gp5
                else "while generating the fingering draft"
            )
            warning_message = (
                f"Reduced {chord_count} impossible simultaneous note "
                f"group{'s' if chord_count != 1 else ''} to one bassline note per group "
                f"{pruning_stage}; removed {removed_count} likely contaminating "
                f"note{'s' if removed_count != 1 else ''} instead of aborting: "
                + "; ".join(chord_summaries)
                + "."
            )
            pipeline_warnings.append(warning_message)
            debug_run.add_warning(
                warning_message,
                category="gp5_impossible_chord_pruning",
                details={"chords": pruned_chord_details},
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
                "raw_notes": str(final_raw_notes.resolve()),
                "fingering_draft": str(final_fingering_draft.resolve()),
                "gp5": str(final_output.resolve()) if generate_gp5 else None,
                "reused_raw_notes": reused_raw_notes,
            },
        )
        debug_run.finish_success()
        debug_run.write(final_debug_log)
        completed_artifact = final_output if generate_gp5 else final_fingering_draft
        _notify(progress, 1.0, f"Finished: {completed_artifact.name}")
        return ProcessingResult(
            output=final_output,
            note_count=len(notes),
            bpm=rhythm.bpm,
            debug_log=final_debug_log,
            copied_source=copied_source,
            warnings=tuple(pipeline_warnings),
            source=source,
            notes=tuple(notes),
            rhythm=rhythm,
            five_string=five_string,
            fingering_profile=fingering_profile,
            raw_notes=final_raw_notes,
            fingering_draft=final_fingering_draft,
            reused_raw_notes=reused_raw_notes,
        )
    except Exception as error:
        if not reused_raw_notes:
            debug_run.set_section("transcription", transcription_trace.as_dict())
        debug_run.finish_failure(error)
        try:
            debug_run.write(final_debug_log)
        except Exception as debug_error:
            error.add_note(f"Additionally failed to write debug log: {debug_error}")
        raise


def rewrite_gp5_fingering(
    result: ProcessingResult,
    fingering_profile: str | None,
) -> GP5ExportResult:
    """Rewrite a completed GP5 with another fingering without retranscribing."""
    if result.source is None or result.rhythm is None or not result.notes:
        raise ValueError("processing result does not retain reusable transcription data")
    result.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="bass-transcriber-fingering-",
        dir=result.output.parent,
    ) as temporary:
        temporary_output = Path(temporary) / result.output.name
        export_result = write_gp5(
            temporary_output,
            list(result.notes),
            result.rhythm,
            title=f"{result.source.stem} - Bass",
            fingering_profile=fingering_profile,
            five_string=result.five_string,
        )
        temporary_output.replace(result.output)
    return export_result


def rewrite_fingering_draft(
    result: ProcessingResult,
    fingering_profile: str | None,
) -> GP5ExportResult:
    """Regenerate the editable draft with another deterministic strategy."""
    if result.source is None or result.rhythm is None or not result.notes:
        raise ValueError("processing result does not retain reusable transcription data")
    draft = result.fingering_draft
    if draft is None:
        draft = fingering_path_for(result.source, result.output.parent)
    strings = BEADG_STRINGS if result.five_string else EADG_STRINGS
    timeline = build_fingering_timeline(
        result.notes,
        result.rhythm,
        fingering_profile,
        strings=strings,
    )
    write_fingering_json(
        draft,
        timeline,
        source=result.source,
        title=f"{result.source.stem} - Bass",
        profile=fingering_profile,
        strings=strings,
        rhythm=result.rhythm,
    )
    return GP5ExportResult(
        len(timeline.notes),
        timeline.dropped_pitches,
        timeline.pruned_chords,
    )


def export_gp5_from_fingering_draft(result: ProcessingResult) -> GP5ExportResult:
    """Validate the current on-disk draft and atomically convert it to GP5."""
    if result.fingering_draft is None:
        raise ValueError("processing result does not have a fingering draft")
    document = read_fingering_json(result.fingering_draft)
    result.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="bass-transcriber-gp5-",
        dir=result.output.parent,
    ) as temporary:
        temporary_output = Path(temporary) / result.output.name
        write_resolved_gp5(
            temporary_output,
            document.notes,
            document.rhythm,
            title=document.title,
            strings=document.strings,
        )
        temporary_output.replace(result.output)
    return GP5ExportResult(
        len(document.notes),
        document.dropped_pitches,
        document.pruned_chords,
    )


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
