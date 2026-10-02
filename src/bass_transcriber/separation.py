"""Pinned BS-RoFormer SW bass separation in an isolated GPU worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path
from typing import cast

import soundfile as sf  # type: ignore[import-untyped]

MODEL = "roformer-model-bs-roformer-sw-by-jarredou"
WEIGHTS_SHA256 = "24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e"
CONFIG_SHA256 = "52df622c95ff3c1f4e1389f476ed737581a2c2dc12324d52c9763be9ccd2be2b"
PACKAGE_VERSION = "0.1.5"


def file_sha256(path: Path) -> str:
    """Hash audio and model artifacts without reading them all into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_stem(audio: Path, stem: Path) -> None:
    """Reject shifted-length, nonfinite or incorrectly formatted worker output."""
    original = sf.info(str(audio))
    separated = sf.info(str(stem))
    if separated.samplerate != 44100 or separated.channels != 2:
        raise RuntimeError("Bass stem must be stereo audio at 44.1 kHz")
    if abs(original.duration - separated.duration) > 2 / 44100:
        raise RuntimeError("Bass stem duration does not match the source audio")
    import numpy as np

    with sf.SoundFile(str(stem)) as handle:
        for block in handle.blocks(blocksize=65536, dtype="float32"):
            if not np.isfinite(block).all():
                raise RuntimeError("Bass stem contains invalid audio samples")


def separate_bass(
    audio: Path,
    output: Path,
    *,
    device: str = "cuda",
    original_source: Path | None = None,
) -> dict[str, object]:
    """Separate bass; retain logs and provenance, releasing worker VRAM on exit."""
    output.parent.mkdir(parents=True, exist_ok=True)
    source = original_source or audio
    source_hash = file_sha256(source)
    audio_hash = file_sha256(audio)
    metadata_path = output.with_suffix(".separation.json")
    log = output.with_suffix(".separation.log")
    if output.is_file() and metadata_path.is_file():
        try:
            cached = cast(dict[str, object], json.loads(metadata_path.read_text(encoding="utf-8")))
            if (
                cached.get("source_sha256") == source_hash
                and cached.get("input_sha256") == audio_hash
                and cached.get("weights_sha256") == WEIGHTS_SHA256
                and cached.get("config_sha256") == CONFIG_SHA256
                and cached.get("package_version") == PACKAGE_VERSION
                and cached.get("device") == device
                and cached.get("stem_sha256") == file_sha256(output)
            ):
                validate_stem(audio, output)
                return {**cached, "reused": True}
        except (OSError, ValueError, RuntimeError):
            pass
    with tempfile.TemporaryDirectory(prefix="bass-separation-", dir=output.parent) as temporary:
        work = Path(temporary)
        stem = work / "bass.wav"
        metadata = work / "metadata.json"
        command = [
            sys.executable, "-X", "utf8", "-m", "bass_transcriber.separation",
            str(audio.resolve()), str(stem.resolve()), str(metadata.resolve()),
            "--device", device,
        ]
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command, stdout=handle, stderr=subprocess.STDOUT, check=False,
            )
        if completed.returncode:
            detail = log.read_text(encoding="utf-8")[-3000:]
            raise RuntimeError(f"Bass separation failed. Log: {log}\n{detail}")
        validate_stem(audio, stem)
        result = cast(dict[str, object], json.loads(metadata.read_text(encoding="utf-8")))
        result.update(
            source=str(source.resolve()), source_sha256=source_hash,
            input_sha256=audio_hash,
            stem=str(output.resolve()), stem_sha256=file_sha256(stem), reused=False,
            log=str(log.resolve()),
        )
        stem.replace(output)
        metadata_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result


def _worker(audio: Path, output: Path, metadata: Path, device: str) -> None:
    # Model imports belong only in the short-lived worker, not the UI process.
    import numpy as np
    import torch
    import yaml  # type: ignore[import-untyped]
    from bs_roformer import ensure_model_assets  # type: ignore[import-untyped]
    from bs_roformer.inference import SafeLoaderWithTuple  # type: ignore[import-untyped]
    from bs_roformer.utils import demix_track, get_model_from_config  # type: ignore[import-untyped]
    from ml_collections import ConfigDict  # type: ignore[import-untyped]

    started = time.perf_counter()
    if version("bs-roformer-infer") != PACKAGE_VERSION:
        raise RuntimeError("Unexpected separator package version")
    weights, config_path = ensure_model_assets(MODEL)
    if file_sha256(weights) != WEIGHTS_SHA256 or file_sha256(config_path) != CONFIG_SHA256:
        raise RuntimeError("Separator checkpoint/configuration differs from the tested version")
    config = ConfigDict(yaml.load(config_path.read_text(), Loader=SafeLoaderWithTuple))
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is required for bass separation")
    decoded = output.with_name("input.wav")
    subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-i", str(audio), "-ar", "44100",
         "-ac", "2", "-c:a", "pcm_f32le", str(decoded)], check=True,
    )
    mix, sr = sf.read(str(decoded), dtype="float32", always_2d=True)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for bass separation")
    model = get_model_from_config("bs_roformer", config)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval().to(device)
    stems, _ = demix_track(
        config, model, torch.from_numpy(mix.T.copy()), torch.device(device),
    )
    bass = stems["bass"].T
    if bass.shape != mix.shape or not np.isfinite(bass).all():
        raise RuntimeError("Separator returned invalid bass audio")
    sf.write(str(output), bass, sr, subtype="FLOAT")
    metadata.write_text(json.dumps({
        "model": MODEL, "package_version": version("bs-roformer-infer"),
        "weights_sha256": WEIGHTS_SHA256, "config_sha256": CONFIG_SHA256,
        "weights": str(weights), "config": str(config_path), "device": device,
        "sample_rate": sr, "frames": len(bass), "num_overlap": config.inference.num_overlap,
        "seconds": time.perf_counter() - started,
    }, indent=2), encoding="utf-8")


def main() -> None:
    """Internal worker command; errors propagate to the persistent run log."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    _worker(args.audio, args.output, args.metadata, args.device)


if __name__ == "__main__":
    main()
