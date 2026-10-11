from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from bass_transcriber.separation_inference import demix_bass


class ContextModel(nn.Module):
    """Deterministic multi-stem predictions that depend on chunk boundaries."""

    def forward(self, audio: Tensor) -> Tensor:
        averaged = F.avg_pool1d(audio, 3, stride=1, padding=1)
        return torch.stack((audio * 0.25, averaged, -audio), dim=1)


def _config(overlap: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        inference=SimpleNamespace(chunk_size=32, num_overlap=overlap),
        training=SimpleNamespace(target_instrument=None, instruments=["drums", "bass", "other"]),
    )


@pytest.mark.parametrize("frames", [1, 7, 16, 17, 31, 32, 33, 63, 64, 65, 95, 137])
@pytest.mark.parametrize("overlap", [1, 2, 3])
def test_bass_matches_upstream_windows_padding_and_tail_chunks(frames: int, overlap: int) -> None:
    from bs_roformer.utils import demix_track

    mix = torch.randn(2, frames, generator=torch.Generator().manual_seed(42))
    expected, _ = demix_track(_config(overlap), ContextModel(), mix.clone(), torch.device("cpu"))
    updates = []
    actual = demix_bass(
        _config(overlap),
        ContextModel(),
        mix,
        torch.device("cpu"),
        progress=lambda completed, total: updates.append((completed, total)),
    )
    torch.testing.assert_close(actual, torch.from_numpy(expected["bass"]), rtol=0, atol=0)
    assert actual.shape == (2, frames)
    assert actual.device.type == "cpu"
    total = updates[-1][1]
    assert updates == [(completed, total) for completed in range(1, total + 1)]


def test_single_bass_target_is_supported() -> None:
    config = _config()
    config.training.target_instrument = "bass"

    class BassModel(nn.Module):
        def forward(self, audio: Tensor) -> Tensor:
            return audio.unsqueeze(1)

    mix = torch.randn(2, 100)
    actual = demix_bass(config, BassModel(), mix, torch.device("cpu"))
    torch.testing.assert_close(actual, mix)


def test_unexpected_model_shape_is_reported() -> None:
    with pytest.raises(RuntimeError, match="chunk shape"):
        demix_bass(_config(), nn.Identity(), torch.zeros(2, 100), torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_inference_sends_only_fixed_size_chunks_and_returns_cpu_bass() -> None:
    class CheckedModel(ContextModel):
        def forward(self, audio: Tensor) -> Tensor:
            assert audio.device.type == "cuda"
            assert audio.shape == (1, 2, 32)
            return super().forward(audio)

    mix = torch.randn(2, 1000)
    expected = demix_bass(_config(), ContextModel(), mix, torch.device("cpu"))
    actual = demix_bass(_config(), CheckedModel(), mix, torch.device("cuda"))
    assert actual.device.type == "cpu"
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
