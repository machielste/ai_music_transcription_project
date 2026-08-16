from bass_transcriber import __version__
from bass_transcriber.cli import build_parser, main
from bass_transcriber.diagnostics import Diagnostic


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
            "slap_funk",
            "--strings",
            "4",
        ]
    )

    assert args.fingering_profile == "slap_funk"
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
