from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest

from bass_transcriber.context_menu import AUDIO_EXTENSIONS, menu_entries
from bass_transcriber.output_folders import create_run_folder
from bass_transcriber.shell_job import run_job


def test_folder_reservations_preserve_previous_and_concurrent_results(tmp_path: Path) -> None:
    source = Path("Song name.mp3")
    with ThreadPoolExecutor(max_workers=4) as executor:
        folders = list(
            executor.map(lambda _: create_run_folder(source, tmp_path, 4, "balanced"), range(8))
        )
    assert len(set(folders)) == 8
    assert all(folder.is_dir() and folder.parent == tmp_path for folder in folders)
    assert tmp_path / "Song name.4-string.balanced" in folders
    assert create_run_folder(source, tmp_path, 5, "avoid_open").name == (
        "Song name.5-string.avoid_open"
    )


def test_menu_commands_preserve_paths_and_all_settings() -> None:
    entries = menu_entries(Path("C:/Python env/pythonw.exe"), Path("C:/My repo/outputs"))
    commands = [values[""] for key, values in entries.items() if key.endswith(r"\command")]
    assert len(commands) == 4
    for command in commands:
        assert command.startswith(
            f'"{Path("C:/Python env/pythonw.exe")}" -m bass_transcriber.shell_job'
        )
        assert f'--output-root "{Path("C:/My repo/outputs")}"' in command
        assert command.endswith(' "%1"')
    for strings in (4, 5):
        for profile in ("balanced", "avoid_open"):
            assert (
                sum(
                    f"--strings {strings} --fingering-profile {profile}" in command
                    for command in commands
                )
                == 1
            )


def test_cascades_reference_complete_per_user_command_groups() -> None:
    entries = menu_entries(Path("pythonw.exe"), Path("outputs"))
    for extension in AUDIO_EXTENSIONS:
        root = rf"Software\Classes\SystemFileAssociations\{extension}\shell\BassTranscriber"
        group = entries[root]["ExtendedSubCommandsKey"]
        for strings in (4, 5):
            string_group = entries[rf"Software\Classes\{group}\shell\{strings}strings"]
            profiles = string_group["ExtendedSubCommandsKey"]
            for profile in ("balanced", "avoid_open"):
                command = entries[rf"Software\Classes\{profiles}\shell\{profile}\command"][""]
                assert f"--strings {strings} --fingering-profile {profile}" in command


def test_shell_job_generates_gp5_with_selected_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "song.mp3"
    source.touch()
    calls = []

    def process(audio: Path, destination: Path, **settings: object) -> SimpleNamespace:
        calls.append((audio, destination, settings))
        return SimpleNamespace(
            output=destination / "song.bass.gp5", note_count=10, bpm=120.0, warnings=(),
            tonelib_song=destination / "song.bass.song",
        )

    monkeypatch.setattr("bass_transcriber.pipeline.process_song", process)
    events: Queue[tuple[str, object]] = Queue()
    run_job(source, tmp_path / "outputs", 4, "avoid_open", events)
    assert calls[0][0] == source
    assert calls[0][1].parent == tmp_path / "outputs"
    assert calls[0][2]["five_string"] is False
    assert calls[0][2]["fingering_profile"] == "avoid_open"
    assert calls[0][2]["generate_gp5"] is True
    assert calls[0][2]["copy_source"] is True
    assert calls[0][2]["generate_tonelib"] is True
    assert [events.get_nowait()[0] for _ in range(events.qsize())] == ["folder", "done"]


def test_shell_job_failure_is_visible_and_retained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "song.mp3"
    source.touch()

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("Model unavailable")

    monkeypatch.setattr("bass_transcriber.pipeline.process_song", fail)
    events: Queue[tuple[str, object]] = Queue()
    run_job(source, tmp_path / "outputs", 5, "balanced", events)
    event, folder = events.get_nowait()
    assert event == "folder" and isinstance(folder, Path)
    assert (folder / "error.txt").read_text(encoding="utf-8") == "RuntimeError: Model unavailable"
    assert events.get_nowait() == ("error", "RuntimeError: Model unavailable")
