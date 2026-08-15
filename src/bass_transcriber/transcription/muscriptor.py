"""Thin integration boundary around MuScriptor model artifacts."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
from muscriptor import TranscriptionModel  # type: ignore[import-untyped]
from muscriptor.events import (  # type: ignore[import-untyped]
    NoteEndEvent,
    ProgressEvent,
)

from bass_transcriber.models import BassInstrument, BassNote

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
    instrument: BassInstrument = "electric_bass",
    device: str = "cuda",
    progress: ProgressCallback | None = None,
) -> list[BassNote]:
    """Transcribe audio with MuScriptor constrained to one bass family."""
    model = TranscriptionModel.load_model(size, device=device)
    events = model.transcribe(audio, instruments=[instrument])
    return notes_from_events(events, progress=progress)


def notes_from_events(
    events: Iterable[object],
    *,
    progress: ProgressCallback | None = None,
) -> list[BassNote]:
    """Convert MuScriptor's paired event stream into neutral notes."""
    notes: list[BassNote] = []
    for event in events:
        if isinstance(event, ProgressEvent):
            if progress is not None:
                progress(int(event.completed), int(event.total))
            continue
        if not isinstance(event, NoteEndEvent):
            continue

        start = event.start_event
        if start.instrument not in ("electric_bass", "acoustic_bass"):
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
