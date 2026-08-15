from bass_transcriber import __version__
from bass_transcriber.cli import main


def test_version_is_defined() -> None:
    assert __version__ == "0.1.0"


def test_cli_without_arguments_prints_help(capsys: object) -> None:
    assert main([]) == 0

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "ToneLib Jam" in captured.out
