"""Versioned machine-readable diagnostics for complete pipeline runs."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
import traceback
import uuid
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import TracebackType
from typing import Literal

import numpy as np
import soundfile  # type: ignore[import-untyped]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_versions() -> dict[str, str | None]:
    packages = (
        "bass-transcriber",
        "muscriptor",
        "torch",
        "librosa",
        "soundfile",
        "pyguitarpro",
        "mido",
        "bs-roformer-infer",
    )
    result: dict[str, str | None] = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _torch_environment() -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {"available": False}

    cuda_available = torch.cuda.is_available()
    result: dict[str, object] = {
        "available": True,
        "version": str(torch.__version__),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
    }
    if cuda_available:
        device = torch.cuda.current_device()
        result.update(
            {
                "cuda_device_index": device,
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_compute_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    return result


def analyze_audio_file(path: Path, *, chunk_seconds: float = 5.0) -> dict[str, object]:
    """Return file properties and per-chunk signal levels without retaining audio."""
    try:
        with soundfile.SoundFile(str(path)) as audio:
            sample_rate = int(audio.samplerate)
            frames_per_chunk = max(1, round(chunk_seconds * sample_rate))
            chunks: list[dict[str, object]] = []
            chunk_index = 0
            while True:
                samples = audio.read(
                    frames_per_chunk,
                    dtype="float32",
                    always_2d=True,
                )
                if len(samples) == 0:
                    break
                square_mean = float(np.mean(np.square(samples, dtype=np.float64)))
                rms = math.sqrt(square_mean)
                peak = float(np.max(np.abs(samples)))
                start_seconds = chunk_index * chunk_seconds
                chunks.append(
                    {
                        "chunk_index": chunk_index,
                        "start_seconds": start_seconds,
                        "end_seconds": start_seconds + len(samples) / sample_rate,
                        "frame_count": len(samples),
                        "rms_amplitude": rms,
                        "rms_dbfs": 20.0 * math.log10(max(rms, 1e-12)),
                        "peak_amplitude": peak,
                        "peak_dbfs": 20.0 * math.log10(max(peak, 1e-12)),
                    }
                )
                chunk_index += 1
            return {
                "path": str(path.resolve()),
                "sample_rate": sample_rate,
                "channels": int(audio.channels),
                "frame_count": int(audio.frames),
                "duration_seconds": float(audio.frames / sample_rate),
                "format": str(audio.format),
                "subtype": str(audio.subtype),
                "chunk_seconds": chunk_seconds,
                "chunks": chunks,
            }
    except Exception as error:
        return {
            "path": str(path.resolve()),
            "probe_error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }


class DebugRun:
    """Mutable run recorder that writes one stable JSON document."""

    def __init__(
        self,
        source: Path,
        output: Path,
        *,
        configuration: dict[str, object],
        ffmpeg_path: str | None,
    ) -> None:
        source_stat = source.stat()
        self.started_monotonic = time.perf_counter()
        self.stages: list[dict[str, object]] = []
        self.warnings: list[dict[str, object]] = []
        self.document: dict[str, object] = {
            "schema_version": 1,
            "document_type": "bass_transcriber_debug_run",
            "run_id": str(uuid.uuid4()),
            "status": "running",
            "started_utc": _utc_now(),
            "finished_utc": None,
            "elapsed_seconds": None,
            "source": {
                "path": str(source.resolve()),
                "name": source.name,
                "extension": source.suffix.lower(),
                "size_bytes": source_stat.st_size,
                "modified_utc": datetime.fromtimestamp(
                    source_stat.st_mtime, UTC
                ).isoformat(),
                "sha256": _sha256(source),
            },
            "output": {
                "gp5_path": str(output.resolve()),
            },
            "configuration": configuration,
            "environment": {
                "python_version": sys.version,
                "platform": platform.platform(),
                "executable": sys.executable,
                "packages": _package_versions(),
                "torch": _torch_environment(),
                "ffmpeg_path": ffmpeg_path,
            },
            "stages": self.stages,
            "warnings": self.warnings,
            "error": None,
        }

    def stage(self, name: str) -> _DebugStage:
        """Return a context manager that records duration and exceptions."""
        return _DebugStage(self.stages, name)

    def set_section(self, name: str, value: object) -> None:
        """Set or replace a top-level diagnostic section."""
        self.document[name] = value

    def add_warning(
        self,
        message: str,
        *,
        category: str = "pipeline",
        details: dict[str, object] | None = None,
    ) -> None:
        """Append a structured warning."""
        warning: dict[str, object] = {
            "category": category,
            "message": message,
        }
        if details is not None:
            warning["details"] = details
        self.warnings.append(warning)

    def finish_success(self) -> None:
        """Mark the run as successfully completed."""
        self._finish("succeeded")

    def finish_failure(self, error: BaseException) -> None:
        """Mark the run as failed and retain structured traceback information."""
        self.document["error"] = {
            "type": type(error).__name__,
            "module": type(error).__module__,
            "message": str(error),
            "traceback": traceback.format_exception(type(error), error, error.__traceback__),
        }
        self._finish("failed")

    def _finish(self, status: str) -> None:
        self.document["status"] = status
        self.document["finished_utc"] = _utc_now()
        self.document["elapsed_seconds"] = time.perf_counter() - self.started_monotonic

    def write(self, output: Path) -> None:
        """Atomically write the current diagnostic document."""
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{self.document['run_id']}.tmp")
        temporary.write_text(
            json.dumps(self.document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)


class _DebugStage:
    def __init__(self, stages: list[dict[str, object]], name: str) -> None:
        self.stages = stages
        self.name = name
        self.started_monotonic = 0.0
        self.record: dict[str, object] = {}

    def __enter__(self) -> _DebugStage:
        self.started_monotonic = time.perf_counter()
        self.record = {
            "name": self.name,
            "status": "running",
            "started_utc": _utc_now(),
            "finished_utc": None,
            "elapsed_seconds": None,
            "error": None,
        }
        self.stages.append(self.record)
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback_object: TracebackType | None,
    ) -> Literal[False]:
        del traceback_object
        self.record["status"] = "failed" if error is not None else "succeeded"
        self.record["finished_utc"] = _utc_now()
        self.record["elapsed_seconds"] = time.perf_counter() - self.started_monotonic
        if error is not None:
            self.record["error"] = {
                "type": error_type.__name__ if error_type is not None else "Exception",
                "message": str(error),
            }
        return False
