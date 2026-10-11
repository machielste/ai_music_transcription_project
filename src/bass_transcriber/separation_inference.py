"""Bass-only overlap-add adapter for the pinned BS-RoFormer inference package.

Keep whole-recording tensors on CPU; only one model input/output chunk lives on
the accelerator. Match the upstream chunk boundaries, reflection and windows.

Adapted from bs-roformer-infer's demix_track, Copyright (c) 2025 OpenMIRLab.
Its MIT license is included in separation_inference.LICENSE.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def demix_bass(
    config: Any,
    model: nn.Module,
    mix: Tensor,
    device: torch.device,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> Tensor:
    """Return channels × frames on CPU without full-length accelerator allocations."""
    if mix.device.type != "cpu" or mix.ndim != 2 or mix.shape[1] == 0:
        raise ValueError("Separation input must be nonempty channels × frames on CPU")
    configured_chunk = getattr(config.inference, "chunk_size", None)
    chunk_size = int(
        configured_chunk
        if configured_chunk is not None
        else getattr(getattr(config, "audio", None), "chunk_size", 588800)
    )
    overlap = int(config.inference.num_overlap)
    if chunk_size < 10 or overlap < 1 or overlap > chunk_size:
        raise ValueError("Invalid separator chunk size or overlap")
    step = chunk_size // overlap
    fade_size = chunk_size // 10
    border = chunk_size - step
    original_frames = mix.shape[1]
    padded = original_frames > 2 * border and border > 0
    if padded:
        mix = F.pad(mix, (border, border), mode="reflect")

    target = config.training.target_instrument
    instruments = list(config.training.instruments) if target is None else [target]
    if "bass" not in instruments:
        raise ValueError("Separator model has no bass output")
    bass_index = instruments.index("bass")
    frames = mix.shape[1]
    result = torch.zeros(mix.shape, dtype=torch.float32, device="cpu")
    # The weights are identical for both channels and all stems.
    counter = torch.zeros(frames, dtype=torch.float32, device="cpu")
    window = torch.ones(chunk_size, dtype=torch.float32, device="cpu")
    window[:fade_size] *= torch.linspace(0, 1, fade_size)
    window[-fade_size:] *= torch.linspace(1, 0, fade_size)
    total = (frames + step - 1) // step

    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        for completed, start in enumerate(range(0, frames, step), start=1):
            part = mix[:, start : start + chunk_size]
            length = part.shape[1]
            if length < chunk_size:
                mode = "reflect" if length > chunk_size // 2 + 1 else "constant"
                part = F.pad(part, (0, chunk_size - length), mode=mode)
            weights = window.clone()
            if start == 0:
                weights[:fade_size] = 1
            elif start + chunk_size >= frames:
                weights[-fade_size:] = 1

            prediction = model(part.to(device).unsqueeze(0))
            if prediction.shape != (1, len(instruments), mix.shape[0], chunk_size):
                raise RuntimeError("Separator returned an unexpected chunk shape")
            # Apply the same float32 window on the model device before transferring
            # just bass. Release all GPU outputs before starting the next chunk.
            bass = prediction[0, bass_index, :, :length]
            weighted = (bass * weights[:length].to(device)).to("cpu", dtype=torch.float32)
            result[:, start : start + length] += weighted
            counter[start : start + length] += weights[:length]
            del prediction, bass, weighted
            if progress is not None:
                progress(completed, total)

    result.div_(counter)
    # Match upstream treatment of undefined samples at window boundaries.
    result.nan_to_num_()
    if padded:
        result = result[:, border:-border]
    assert result.shape[1] == original_frames
    return result
