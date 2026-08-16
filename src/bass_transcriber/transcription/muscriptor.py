"""Thin integration boundary around MuScriptor model artifacts."""

from __future__ import annotations

import re
import time
import warnings
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, cast

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
from muscriptor import TranscriptionModel  # type: ignore[import-untyped]
from muscriptor.events import (  # type: ignore[import-untyped]
    NoteEndEvent,
    NoteStartEvent,
    ProgressEvent,
)
from soundfile import info  # type: ignore[import-untyped]

from bass_transcriber.models import BassInstrument, BassNote
from bass_transcriber.postprocess import clip_notes_to_duration

ModelSize = Literal["small", "medium", "large"]
MODEL_SIZES: tuple[ModelSize, ...] = ("small", "medium", "large")


class ModelFetchError(RuntimeError):
    """A model download failure the user can resolve."""


@dataclass(frozen=True, slots=True)
class ModelFiles:
    """Paths to a cached MuScriptor checkpoint and its architecture config."""

    size: ModelSize
    weights: Path
    config: Path


ProgressCallback = Callable[[int, int], None]


@dataclass(slots=True)
class TranscriptionTrace:
    """Complete serializable trace of model inference and event filtering."""

    events: list[dict[str, object]] = field(default_factory=list)
    model_warnings: list[dict[str, object]] = field(default_factory=list)
    boundary_adjustments: list[dict[str, object]] = field(default_factory=list)
    model_size: str | None = None
    device: str | None = None
    requested_instruments: list[str] | None = None
    model_load_seconds: float | None = None
    inference_seconds: float | None = None
    audio_duration_seconds: float | None = None
    bass_notes_before_boundary_cleanup: int = 0
    bass_notes_after_boundary_cleanup: int = 0

    def add_model_warning(self, warning: warnings.WarningMessage) -> None:
        """Retain a warning emitted while the lazy model stream was consumed."""
        message = str(warning.message)
        chunk_match = re.search(r"chunk (\d+) \(seek=([0-9.]+)s\)", message)
        record: dict[str, object] = {
            "category": warning.category.__name__,
            "message": message,
            "filename": warning.filename,
            "lineno": warning.lineno,
        }
        if chunk_match is not None:
            record["chunk_index"] = int(chunk_match.group(1))
            record["seek_seconds"] = float(chunk_match.group(2))
        self.model_warnings.append(record)

    def as_dict(self) -> dict[str, object]:
        """Return the complete trace plus derived event and chunk summaries."""
        type_counts = Counter(str(event["type"]) for event in self.events)
        instrument_counts = Counter(
            str(event["instrument"])
            for event in self.events
            if event["type"] == "note_start" and "instrument" in event
        )
        progress = [event for event in self.events if event["type"] == "progress"]
        total_chunks = max(
            (cast(int, event["total"]) for event in progress),
            default=0,
        )
        chunks: list[dict[str, object]] = []
        for chunk_index in range(total_chunks):
            chunk_events = [
                event
                for event in self.events
                if event.get("stream_chunk_index") == chunk_index
            ]
            starts = [event for event in chunk_events if event["type"] == "note_start"]
            accepted_starts = [
                event
                for event in chunk_events
                if event["type"] == "note_start"
                and event.get("instrument") in ("electric_bass", "acoustic_bass")
            ]
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "nominal_start_seconds": chunk_index * 5.0,
                    "nominal_end_seconds": (chunk_index + 1) * 5.0,
                    "event_count": len(chunk_events),
                    "note_start_count": len(starts),
                    "accepted_bass_note_start_count": len(accepted_starts),
                    "instrument_note_start_counts": dict(
                        Counter(str(event["instrument"]) for event in starts)
                    ),
                    "has_no_note_events": not starts,
                }
            )
        return {
            "model_size": self.model_size,
            "device": self.device,
            "requested_instruments": self.requested_instruments,
            "model_load_seconds": self.model_load_seconds,
            "inference_seconds": self.inference_seconds,
            "audio_duration_seconds": self.audio_duration_seconds,
            "event_count": len(self.events),
            "event_type_counts": dict(type_counts),
            "instrument_note_start_counts": dict(instrument_counts),
            "bass_notes_before_boundary_cleanup": self.bass_notes_before_boundary_cleanup,
            "bass_notes_after_boundary_cleanup": self.bass_notes_after_boundary_cleanup,
            "chunks": chunks,
            "model_warnings": self.model_warnings,
            "boundary_adjustments": self.boundary_adjustments,
            "events": self.events,
        }


def fetch_model(size: ModelSize) -> ModelFiles:
    """Download a published checkpoint into the Hugging Face cache.

    MuScriptor's own ``TranscriptionModel.load_model(size)`` resolves the same
    repository and reuses these cached files during inference.
    """
    repo_id = f"MuScriptor/muscriptor-{size}"
    try:
        weights = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        config = hf_hub_download(repo_id=repo_id, filename="config.json")
    except (GatedRepoError, RepositoryNotFoundError) as error:
        raise ModelFetchError(
            f"access to {repo_id} is not enabled; accept its Hugging Face conditions "
            "and run 'uvx hf auth login'"
        ) from error
    except HfHubHTTPError as error:
        status_code = getattr(error.response, "status_code", None)
        if status_code in (401, 403):
            raise ModelFetchError(
                f"Hugging Face denied access to {repo_id}; check the model conditions "
                "and your local login"
            ) from error
        raise
    return ModelFiles(size=size, weights=Path(weights), config=Path(config))


def transcribe_bass(
    audio: Path,
    *,
    size: ModelSize = "small",
    instrument: BassInstrument | None = "electric_bass",
    device: str = "cuda",
    progress: ProgressCallback | None = None,
    trace: TranscriptionTrace | None = None,
) -> list[BassNote]:
    """Transcribe bass notes, optionally using a hard instrument constraint.

    With ``instrument=None``, MuScriptor identifies all instruments itself and
    :func:`notes_from_events` retains only events it labelled as bass.
    """
    if trace is not None:
        trace.model_size = size
        trace.device = device
        trace.requested_instruments = None if instrument is None else [instrument]

    load_started = time.perf_counter()
    model = TranscriptionModel.load_model(size, device=device)
    if trace is not None:
        trace.model_load_seconds = time.perf_counter() - load_started
    instruments = None if instrument is None else [instrument]
    inference_started = time.perf_counter()
    caught_warnings: list[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            try:
                warnings.simplefilter("always")
                events = model.transcribe(audio, instruments=instruments)
                notes = notes_from_events(events, progress=progress, trace=trace)
            finally:
                caught_warnings = list(caught)
    finally:
        if trace is not None:
            trace.inference_seconds = time.perf_counter() - inference_started
            for warning in caught_warnings:
                trace.add_model_warning(warning)

    duration_seconds = float(info(str(audio)).duration)
    clipped = clip_notes_to_duration(notes, duration_seconds)
    if trace is not None:
        trace.audio_duration_seconds = duration_seconds
        trace.bass_notes_before_boundary_cleanup = len(notes)
        trace.bass_notes_after_boundary_cleanup = len(clipped)
        for note in notes:
            if note.start_seconds >= duration_seconds:
                trace.boundary_adjustments.append(
                    {
                        "action": "dropped_after_audio_end",
                        "audio_duration_seconds": duration_seconds,
                        "note": asdict(note),
                    }
                )
            elif note.end_seconds > duration_seconds:
                trace.boundary_adjustments.append(
                    {
                        "action": "clipped_at_audio_end",
                        "original_end_seconds": note.end_seconds,
                        "new_end_seconds": duration_seconds,
                        "note": asdict(note),
                    }
                )
    return clipped


def notes_from_events(
    events: Iterable[object],
    *,
    progress: ProgressCallback | None = None,
    trace: TranscriptionTrace | None = None,
) -> list[BassNote]:
    """Convert MuScriptor's paired event stream into neutral notes."""
    notes: list[BassNote] = []
    sequence = 0
    stream_chunk_index = 0
    for event in events:
        if isinstance(event, ProgressEvent):
            if trace is not None:
                trace.events.append(
                    {
                        "sequence": sequence,
                        "type": "progress",
                        "completed": int(event.completed),
                        "total": int(event.total),
                    }
                )
                sequence += 1
            stream_chunk_index = int(event.completed)
            if progress is not None:
                progress(int(event.completed), int(event.total))
            continue
        if isinstance(event, NoteStartEvent):
            if trace is not None:
                trace.events.append(
                    {
                        "sequence": sequence,
                        "type": "note_start",
                        "stream_chunk_index": stream_chunk_index,
                        "event_index": int(event.index),
                        "pitch": int(event.pitch),
                        "start_seconds": float(event.start_time),
                        "instrument": str(event.instrument),
                    }
                )
                sequence += 1
            continue
        if not isinstance(event, NoteEndEvent):
            if trace is not None:
                trace.events.append(
                    {
                        "sequence": sequence,
                        "type": "unknown",
                        "stream_chunk_index": stream_chunk_index,
                        "python_type": f"{type(event).__module__}.{type(event).__qualname__}",
                    }
                )
                sequence += 1
            continue

        start = event.start_event
        accepted = start.instrument in ("electric_bass", "acoustic_bass")
        if trace is not None:
            trace.events.append(
                {
                    "sequence": sequence,
                    "type": "note_end",
                    "stream_chunk_index": stream_chunk_index,
                    "event_index": int(start.index),
                    "pitch": int(start.pitch),
                    "start_seconds": float(start.start_time),
                    "end_seconds": float(event.end_time),
                    "duration_seconds": float(event.end_time - start.start_time),
                    "instrument": str(start.instrument),
                    "decision": "accepted" if accepted else "rejected",
                    "rejection_reason": None if accepted else "non_bass_instrument",
                }
            )
            sequence += 1
        if not accepted:
            continue
        notes.append(
            BassNote(
                pitch=int(start.pitch),
                start_seconds=float(start.start_time),
                end_seconds=float(event.end_time),
                instrument=start.instrument,
                confidence=None,
            )
        )
    return sorted(notes, key=lambda note: (note.start_seconds, note.pitch, note.end_seconds))
