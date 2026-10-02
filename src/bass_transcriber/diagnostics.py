"""Local environment checks for the transcription pipeline."""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

from huggingface_hub import get_token

DiagnosticStatus = Literal["ok", "warning", "error"]


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """One human-readable environment check."""

    name: str
    status: DiagnosticStatus
    detail: str


def collect_diagnostics() -> list[Diagnostic]:
    """Inspect prerequisites without downloading model weights."""
    checks = [
        Diagnostic("Python", "ok", sys.version.split()[0]),
        _check_executable("FFmpeg", "ffmpeg"),
        _check_torch(),
        _check_hugging_face_auth(),
        _check_separator(),
    ]
    return checks


def has_errors(checks: list[Diagnostic]) -> bool:
    """Return whether any required prerequisite failed."""
    return any(check.status == "error" for check in checks)


def _check_separator() -> Diagnostic:
    from bass_transcriber.separation import PACKAGE_VERSION

    try:
        installed = version("bs-roformer-infer")
    except PackageNotFoundError:
        return Diagnostic("Bass separator", "error", "not installed; run 'uv sync'")
    if installed != PACKAGE_VERSION:
        return Diagnostic("Bass separator", "error", "unexpected version; run 'uv sync'")
    return Diagnostic(
        "Bass separator", "ok",
        f"BS-RoFormer SW; bs-roformer-infer {installed}; weights cached on first use",
    )


def _check_executable(name: str, executable: str) -> Diagnostic:
    path = shutil.which(executable)
    if path is None:
        return Diagnostic(name, "error", f"'{executable}' is not on PATH")
    return Diagnostic(name, "ok", path)


def _check_torch() -> Diagnostic:
    try:
        import torch
    except ImportError:
        return Diagnostic("PyTorch/CUDA", "error", "PyTorch is not installed")

    if not torch.cuda.is_available():
        return Diagnostic(
            "PyTorch/CUDA",
            "error",
            f"PyTorch {torch.__version__} cannot access CUDA",
        )

    device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device)
    capability = ".".join(str(part) for part in torch.cuda.get_device_capability(device))
    return Diagnostic(
        "PyTorch/CUDA",
        "ok",
        f"PyTorch {torch.__version__}; {name}; compute capability {capability}",
    )


def _check_hugging_face_auth() -> Diagnostic:
    if get_token() is None:
        return Diagnostic(
            "Hugging Face",
            "warning",
            "not authenticated; accept the MuScriptor model conditions, then run "
            "'uvx hf auth login'",
        )
    return Diagnostic("Hugging Face", "ok", "an access token is configured")
