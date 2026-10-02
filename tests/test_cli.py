import json
from pathlib import Path

import pytest

from bass_transcriber import __version__
from bass_transcriber.cli import build_parser, main
from bass_transcriber.diagnostics import Diagnostic
from bass_transcriber.models import BassNote


def test_version_is_defined() -> None:
    assert __version__ == "0.1.0"


def test_cli_without_arguments_prints_help(capsys: object) -> None:
    assert main([]) == 0

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "ToneLib Jam" in captured.out


def test_doctor_prints_diagnostics(monkeypatch: object, capsys: object) -> None:
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "bass_transcriber.cli.collect_diagnostics",
        lambda: [Diagnostic("Example", "ok", "ready")],
    )

    assert main(["doctor"]) == 0

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "Example: ready" in captured.out


def test_doctor_fails_on_required_error(monkeypatch: object) -> None:
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "bass_transcriber.cli.collect_diagnostics",
        lambda: [Diagnostic("CUDA", "error", "unavailable")],
    )

    assert main(["doctor"]) == 1


def test_gp5_fingering_profile_can_be_selected() -> None:
    args = build_parser().parse_args(
        [
            "export",
            "gp5",
            "notes.json",
            "--rhythm",
            "rhythm.json",
            "--fingering-profile",
            "avoid_open",
            "--strings",
            "4",
        ]
    )

    assert args.fingering_profile == "avoid_open"
    assert args.strings == 4


def test_fingering_debugger_profiles_can_be_selected() -> None:
    args = build_parser().parse_args(
        [
            "debug",
            "fingerings",
            "notes.json",
            "--rhythm",
            "rhythm.json",
            "--profiles",
            "balanced",
            "legacy",
            "--strings",
            "4",
            "--no-open",
        ]
    )

    assert args.debug_command == "fingerings"
    assert args.profiles == ["balanced", "legacy"]
    assert args.strings == 4
    assert args.no_open is True


def test_cli_separates_before_conditioned_transcription_and_preserves_original_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, output = tmp_path / "mix.mp3", tmp_path / "notes.json"
    source.write_bytes(b"source")
    stem = tmp_path / "notes.json.bass.stem.wav"
    calls = []

    def separate(audio: Path, target: Path, **kwargs: object) -> dict[str, object]:
        calls.append("separation")
        assert audio == source
        assert target == stem
        target.write_bytes(b"stem")
        return {"model": "test"}

    def transcribe(audio: Path, **kwargs: object) -> list[BassNote]:
        calls.append("transcription")
        assert audio == stem
        assert kwargs["instrument"] == "electric_bass"
        return [BassNote(28, 0.0, 0.25, "electric_bass")]

    monkeypatch.setattr("bass_transcriber.cli.separate_bass", separate)
    monkeypatch.setattr("bass_transcriber.cli.transcribe_bass", transcribe)
    assert main(["transcribe", str(source), "--output", str(output),
                 "--separate-bass", "--instrument", "auto"]) == 0
    assert calls == ["separation", "transcription"]
    payload = json.loads(output.read_text())
    assert payload["source"] == str(source.resolve())
    assert payload["transcription_audio"] == str(stem.resolve())
    assert payload["separation"]["model"] == "test"
