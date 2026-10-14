from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest

from bass_transcriber.context_menu import AUDIO_EXTENSIONS, OUTPUT_MODES, PROFILES, menu_entries
from bass_transcriber.gui import _FINGERING_OPTIONS
from bass_transcriber.output_folders import create_run_folder
from bass_transcriber.shell_job import run_job
from bass_transcriber.tab import FINGERING_PROFILES


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
    assert len(commands) == 12
    for command in commands:
        assert command.startswith(
            f'"{Path("C:/Python env/pythonw.exe")}" -m bass_transcriber.shell_job'
        )
        assert f'--output-root "{Path("C:/My repo/outputs")}"' in command
        assert command.endswith(' "%1"')
    for strings in (4, 5):
        for profile in PROFILES:
            for mode in OUTPUT_MODES:
                assert (
                    sum(
                        f"--strings {strings} --fingering-profile {profile} --mode {mode}"
                        in command
                        for command in commands
                    )
                    == 1
                )


def test_cascades_reference_complete_per_user_command_groups() -> None:
    entries = menu_entries(Path("pythonw.exe"), Path("outputs"))
    roots = [
        rf"Software\Classes\SystemFileAssociations\{extension}\shell\BassTranscriber"
        for extension in AUDIO_EXTENSIONS
    ] + [r"Software\Classes\Directory\shell\BassTranscriber"]
    for root in roots:
        group = entries[root]["ExtendedSubCommandsKey"]
        for mode, label in OUTPUT_MODES.items():
            mode_group = entries[rf"Software\Classes\{group}\shell\{mode}"]
            assert mode_group["MUIVerb"] == label
            settings = mode_group["ExtendedSubCommandsKey"]
            for strings in (4, 5):
                for profile in PROFILES:
                    verb = rf"Software\Classes\{settings}\shell\{strings}strings.{profile}"
                    assert entries[verb]["MUIVerb"].startswith(f"{strings}-string (")
                    assert entries[verb]["MUIVerb"].endswith(PROFILES[profile])
                    command = entries[verb + r"\command"][""]
                    assert f"--strings {strings} --fingering-profile {profile}" in command
                    assert f"--mode {mode}" in command


def test_cascade_fits_explorers_entry_budget() -> None:
    entries = menu_entries(Path("pythonw.exe"), Path("outputs"))

    def count_group(group: str) -> int:
        prefix = rf"Software\Classes\{group}\shell" + "\\"
        total = 0
        for key, values in entries.items():
            if not key.startswith(prefix) or "\\" in key.removeprefix(prefix):
                continue
            total += 1
            if "ExtendedSubCommandsKey" in values:
                total += count_group(values["ExtendedSubCommandsKey"])
        return total

    assert count_group("BassTranscriber.Commands") <= 16


def test_desktop_and_explorer_offer_all_supported_fingering_strategies() -> None:
    assert set(_FINGERING_OPTIONS.values()) == set(PROFILES) == set(FINGERING_PROFILES)
    assert _FINGERING_OPTIONS["Combined"] == "combined"


@pytest.mark.parametrize("profile", ["avoid_open", "combined"])
def test_shell_job_generates_gp5_with_selected_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
) -> None:
    source = tmp_path / "song.mp3"
    source.touch()
    calls = []

    def process(audio: Path, destination: Path, **settings: object) -> SimpleNamespace:
        calls.append((audio, destination, settings))
        return SimpleNamespace(
            output=destination / "song.bass.gp5",
            note_count=10,
            bpm=120.0,
            warnings=(),
            tonelib_song=destination / "song.bass.song",
        )

    monkeypatch.setattr("bass_transcriber.pipeline.process_song", process)
    events: Queue[tuple[str, object]] = Queue()
    run_job(source, tmp_path / "outputs", 4, profile, events)
    assert calls[0][0] == source
    assert calls[0][1].parent == tmp_path / "outputs"
    assert calls[0][2]["five_string"] is False
    assert calls[0][2]["fingering_profile"] == profile
    assert calls[0][2]["generate_gp5"] is True
    assert calls[0][2]["copy_source"] is True
    assert calls[0][2]["generate_tonelib"] is True
    assert calls[0][2]["separate_bass"] is True
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


@pytest.mark.parametrize("mode", ["tonelib_only", "debug"])
def test_output_modes_keep_requested_artifacts_and_remove_temporary_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    source = tmp_path / "song.mp3"
    source.write_bytes(b"original music")
    workspaces = []

    def process(audio: Path, destination: Path, **settings: object) -> SimpleNamespace:
        workspaces.append(destination)
        assert settings["copy_source"] is (mode == "debug")
        song = destination / "song.bass.song"
        song.write_bytes(b"self-contained project")
        gp5 = destination / "song.bass.gp5"
        gp5.touch()
        (destination / "debug.json").touch()
        (destination / "stem.wav").touch()
        (destination / "nested").mkdir()
        (destination / "nested" / "trace.json").touch()
        if settings["copy_source"]:
            (destination / audio.name).write_bytes(audio.read_bytes())
        return SimpleNamespace(
            output=gp5,
            tonelib_song=song,
            note_count=10,
            bpm=120.0,
            warnings=("Test warning",),
        )

    monkeypatch.setattr("bass_transcriber.pipeline.process_song", process)
    events: Queue[tuple[str, object]] = Queue()
    run_job(source, tmp_path / "outputs", 4, "balanced", events, mode=mode)
    event, folder = events.get_nowait()
    assert event == "folder" and isinstance(folder, Path)
    assert (folder / "song.bass.song").read_bytes() == b"self-contained project"
    if mode == "tonelib_only":
        assert [path.name for path in folder.iterdir()] == ["song.bass.song"]
        assert not workspaces[0].exists()
    else:
        assert workspaces == [folder]
        assert (folder / "song.bass.gp5").exists()
        assert (folder / "debug.json").exists()
        assert (folder / "song.mp3").read_bytes() == source.read_bytes()
        assert (folder / "nested" / "trace.json").exists()
    assert source.read_bytes() == b"original music"
    event, summary = events.get_nowait()
    assert event == "done" and "Test warning" in summary
    assert ("Created song.bass.gp5" in summary) is (mode == "debug")


@pytest.mark.parametrize("mode", ["tonelib_only", "debug"])
def test_folder_job_filters_files_reports_overall_progress_and_continues_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    source = tmp_path / "Music folder"
    source.mkdir()
    for name in ("A.FLAC", "B.mp3", "B.wav", "ignore.txt"):
        (source / name).touch()
    (source / "nested").mkdir()
    (source / "nested" / "skip.mp3").touch()
    # A directory with a music extension is not an audio file.
    (source / "skip.ogg").mkdir()
    output = tmp_path / "Output folder"
    calls = []
    workspaces = []

    def process(audio: Path, destination: Path, **settings: object) -> SimpleNamespace:
        calls.append(audio.name)
        workspaces.append(destination)
        settings["progress"](0.5, "Transcribing")
        stem = settings["output_stem"]
        (destination / f"{stem}.debug.json").touch()
        if audio.name == "A.FLAC":
            raise RuntimeError("Bad audio")
        song = destination / f"{stem}.bass.song"
        song.write_bytes(audio.name.encode())
        return SimpleNamespace(
            output=destination / f"{audio.stem}.bass.gp5",
            tonelib_song=song,
            note_count=10,
            bpm=120.0,
            warnings=(),
        )

    monkeypatch.setattr("bass_transcriber.pipeline.process_song", process)
    events: Queue[tuple[str, object]] = Queue()
    run_job(source, output, 5, "avoid_open", events, mode=mode)
    assert calls == ["A.FLAC", "B.mp3", "B.wav", "skip.mp3"]
    run_folder = output / "Music folder.5-string.avoid_open"
    assert list(output.iterdir()) == [run_folder]
    assert (run_folder / "A.error.txt").read_text() == ("RuntimeError: Bad audio")
    assert (run_folder / "B.mp3.bass.song").read_bytes() == b"B.mp3"
    assert (run_folder / "B.wav.bass.song").read_bytes() == b"B.wav"
    assert (run_folder / "nested" / "skip.bass.song").read_bytes() == b"skip.mp3"
    if mode == "tonelib_only":
        assert all(not work.exists() for work in workspaces)
        assert not list(output.rglob("*.debug.json"))
    else:
        assert len(list(output.rglob("*.debug.json"))) == 4
    assert events.get_nowait() == ("folder", run_folder)
    remaining = [events.get_nowait() for _ in range(events.qsize())]
    progress = [payload for event, payload in remaining if event == "progress"]
    assert [item[0] for item in progress] == pytest.approx([n / 8 for n in range(1, 9)])
    assert progress[0][1] == "[1/4] A.FLAC: Transcribing"
    assert progress[1][1] == "[1/4] A.FLAC: Failed"
    assert progress[-1][1] == "[4/4] nested/skip.mp3: Finished"
    event, summary = remaining[-1]
    assert event == "error"
    assert "Completed 3/4 music files" in summary
    assert "A.FLAC\nRuntimeError: Bad audio" in summary
    assert "B.wav\n" in summary
    assert "Created B.wav.bass.song" in summary


@pytest.mark.parametrize("mode", ["tonelib_only", "debug"])
def test_recursive_collections_repeat_in_one_run_folder_and_skip_existing_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    source = tmp_path / "Library.v1"
    originals = [
        Path("Intro.mp3"),
        Path("Artist/Album 1/Song.flac"),
        Path("Artist/Album 2/Song.wav"),
    ]
    for relative in originals:
        audio = source / relative
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(relative.as_posix().encode())
    (source / "Empty collection").mkdir()
    output = source / "outputs"
    (output / "old").mkdir(parents=True)
    (output / "old" / "ignored.mp3").touch()
    calls = []

    def process(audio: Path, destination: Path, **settings: object) -> SimpleNamespace:
        calls.append(audio.relative_to(source))
        stem = settings["output_stem"]
        song = destination / f"{stem}.bass.song"
        song.write_bytes(audio.read_bytes())
        if mode == "debug":
            (destination / f"{stem}.bass.debug.json").touch()
            (destination / audio.name).write_bytes(audio.read_bytes())
        return SimpleNamespace(
            output=destination / f"{stem}.bass.gp5",
            tonelib_song=song,
            note_count=10,
            bpm=120.0,
            warnings=(),
        )

    monkeypatch.setattr("bass_transcriber.pipeline.process_song", process)
    for suffix in ("", ".2"):
        events: Queue[tuple[str, object]] = Queue()
        run_job(source, output, 4, "balanced", events, mode=mode)
        run_folder = output / f"Library.v1.4-string.balanced{suffix}"
        assert events.get_nowait() == ("folder", run_folder)
        messages = [events.get_nowait() for _ in range(events.qsize())]
        assert messages[-1][0] == "done"
        for relative in originals:
            assert (run_folder / relative.parent / f"{relative.stem}.bass.song").read_bytes() == (
                (source / relative).read_bytes()
            )
        assert not (run_folder / "Empty collection").exists()
        assert not (run_folder / "outputs").exists()
        assert len(list(run_folder.rglob("*.song"))) == 3
        if mode == "tonelib_only":
            assert len([path for path in run_folder.rglob("*") if path.is_file()]) == 3
    assert set(output.iterdir()) == {
        output / "old",
        output / "Library.v1.4-string.balanced",
        output / "Library.v1.4-string.balanced.2",
    }
    assert calls == sorted(originals, key=lambda path: path.as_posix().casefold()) * 2


def test_empty_folder_reports_error_without_creating_outputs(tmp_path: Path) -> None:
    source = tmp_path / "empty"
    source.mkdir()
    output = tmp_path / "outputs"
    events: Queue[tuple[str, object]] = Queue()
    run_job(source, output, 4, "balanced", events, mode="tonelib_only")
    event, message = events.get_nowait()
    assert event == "error" and "No supported music files" in message
    assert not output.exists()


def test_tonelib_only_missing_project_is_a_visible_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "song.mp3"
    source.touch()
    monkeypatch.setattr(
        "bass_transcriber.pipeline.process_song",
        lambda *a, **kw: SimpleNamespace(tonelib_song=None),
    )
    events: Queue[tuple[str, object]] = Queue()
    run_job(source, tmp_path / "outputs", 4, "balanced", events, mode="tonelib_only")
    _, folder = events.get_nowait()
    assert [path.name for path in folder.iterdir()] == ["error.txt"]
    assert events.get_nowait()[0] == "error"
