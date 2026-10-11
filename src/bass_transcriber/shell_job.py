"""One-click Explorer job with a responsive progress window."""

from __future__ import annotations

import argparse
import os
import queue
import shutil
import tempfile
import threading
import tkinter as tk
from collections import Counter, defaultdict
from collections.abc import Callable
from contextlib import nullcontext, suppress
from pathlib import Path
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from bass_transcriber.context_menu import AUDIO_EXTENSIONS, OUTPUT_MODES, PROFILES
from bass_transcriber.output_folders import create_run_folder


def _music_files(source: Path, output_root: Path) -> list[Path]:
    if source.is_dir():
        files: list[Path] = []
        excluded = output_root.resolve() if output_root.resolve() != source.resolve() else None

        def traversal_error(error: OSError) -> None:
            raise error

        for directory, subfolders, filenames in os.walk(source, onerror=traversal_error):
            parent = Path(directory)
            # Avoid link/junction cycles and re-importing previous generated outputs.
            subfolders[:] = [
                name
                for name in subfolders
                if not (parent / name).is_symlink()
                and not (parent / name).is_junction()
                and (parent / name).resolve() != excluded
            ]
            files.extend(
                parent / name
                for name in filenames
                if Path(name).suffix.lower() in AUDIO_EXTENSIONS and (parent / name).is_file()
            )
        files.sort(
            key=lambda path: (
                path.relative_to(source).as_posix().casefold(),
                path.relative_to(source).as_posix(),
            )
        )
        if not files:
            raise ValueError(f"No supported music files in folder: {source}")
        return files
    if not source.is_file():
        raise FileNotFoundError(f"Music file or folder does not exist: {source}")
    if source.suffix.lower() not in AUDIO_EXTENSIONS:
        raise ValueError(f"Unsupported music file: {source}")
    return [source]


def _output_stems(sources: list[Path]) -> dict[Path, str]:
    """Keep normal track names, disambiguating same-stem files within each collection."""
    collections: dict[Path, list[Path]] = defaultdict(list)
    for audio in sources:
        collections[audio.parent].append(audio)
    stems = {}
    for tracks in collections.values():
        counts = Counter(audio.stem.casefold() for audio in tracks)
        original_names = {audio.name.casefold() for audio in tracks}
        used: set[str] = set()
        for audio in tracks:
            base = audio.name if counts[audio.stem.casefold()] > 1 else audio.stem
            stem = base
            number = 2
            while stem.casefold() in used or f"{stem}.bass.stem.wav".casefold() in original_names:
                stem = f"{base}.{number}"
                number += 1
            used.add(stem.casefold())
            stems[audio] = stem
    return stems


def _generate_song(
    source: Path,
    destination: Path,
    strings: int,
    profile: str,
    mode: str,
    progress: Callable[[float, str], None],
    output_stem: str,
) -> str:
    # Load the backend after the window is visible, inside the worker.
    from bass_transcriber.pipeline import process_song

    workspace = (
        tempfile.TemporaryDirectory(prefix="bass-transcriber-tonelib-only-")
        if mode == "tonelib_only"
        else nullcontext(str(destination))
    )
    with workspace as work:
        result = process_song(
            source,
            Path(work),
            copy_source=mode == "debug",
            five_string=strings == 5,
            fingering_profile=profile,
            merge_sustained_retriggers=True,
            separate_bass=True,
            generate_gp5=True,
            generate_tonelib=True,
            output_stem=output_stem,
            progress=progress,
        )
        if mode == "tonelib_only":
            if result.tonelib_song is None or not result.tonelib_song.is_file():
                raise RuntimeError("Processing did not create a ToneLib project")
            final_song = destination / result.tonelib_song.name
            with final_song.open("xb") as output, result.tonelib_song.open("rb") as song:
                shutil.copyfileobj(song, output)
            created = f"Created {final_song.name}\n"
        else:
            created = f"Created {result.output.name}\n"
            if result.tonelib_song:
                created += f"Created {result.tonelib_song.name}\n"
        return (
            created
            + f"{result.note_count} notes · {result.bpm:.3f} BPM"
            + ("\n\nWarnings:\n" + "\n".join(result.warnings) if result.warnings else "")
        )


def run_job(
    source: Path,
    output_root: Path,
    strings: int,
    profile: str,
    report: queue.Queue[tuple[str, object]],
    *,
    mode: str = "debug",
) -> None:
    """Process a file or a folder sequentially, continuing after individual failures."""
    try:
        if mode not in OUTPUT_MODES:
            raise ValueError(f"Unknown output mode: {mode}")
        sources = _music_files(source, output_root)
        batch = source.is_dir()
        stems = _output_stems(sources)
        run_folder = create_run_folder(source, output_root, strings, profile)
        report.put(("folder", run_folder))
    except Exception as error:
        report.put(("error", f"{type(error).__name__}: {error}"))
        return

    summaries = []
    failures = 0
    for index, audio in enumerate(sources):
        destination: Path | None = None
        failed = False
        name = audio.relative_to(source).as_posix() if batch else audio.name

        def progress(
            fraction: float, message: str, *, index: int = index, name: str = name
        ) -> None:
            prefix = f"[{index + 1}/{len(sources)}] {name}: " if batch else ""
            report.put(("progress", ((index + fraction) / len(sources), prefix + message)))

        try:
            destination = run_folder / audio.parent.relative_to(source) if batch else run_folder
            destination.mkdir(parents=True, exist_ok=True)
            summary = _generate_song(
                audio,
                destination,
                strings,
                profile,
                mode,
                progress,
                stems[audio],
            )
        except Exception as error:
            failed = True
            failures += 1
            summary = f"{type(error).__name__}: {error}"
            if destination is not None:
                # Report errors even if the output folder becomes unwritable.
                with suppress(OSError):
                    error_name = f"{stems[audio]}.error.txt" if batch else "error.txt"
                    (destination / error_name).write_text(summary, encoding="utf-8")
        summaries.append(f"{name}\n{summary}" if batch else summary)
        if batch:
            progress(1.0, "Failed" if failed else "Finished")

    prefix = f"Completed {len(sources) - failures}/{len(sources)} music files.\n\n" if batch else ""
    report.put(("error" if failures else "done", prefix + "\n\n".join(summaries)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="music file or folder (including subfolders)")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--strings", type=int, choices=(4, 5), required=True)
    parser.add_argument("--fingering-profile", choices=PROFILES, required=True)
    parser.add_argument("--mode", choices=OUTPUT_MODES, default="debug")
    args = parser.parse_args()
    root = tk.Tk()
    root.title("Bass Transcriber — generating tablature")
    frame = ttk.Frame(root, padding=20)
    frame.pack(fill="both", expand=True)
    ttk.Label(
        frame,
        text=f"{args.audio.name}\n{args.strings}-string bass · {PROFILES[args.fingering_profile]}"
        f" · {OUTPUT_MODES[args.mode]}",
    ).pack(anchor="w")
    status = tk.StringVar(value="Starting…")
    ttk.Label(frame, textvariable=status, wraplength=600).pack(anchor="w", pady=12)
    progress = ttk.Progressbar(frame, length=600, maximum=100)
    progress.pack(fill="x")
    folder: Path | None = None
    finished = False

    def open_folder() -> None:
        if folder is not None:
            os.startfile(folder)

    open_button = ttk.Button(
        frame, text="Open output folder", command=open_folder, state="disabled"
    )
    open_button.pack(anchor="w", pady=(12, 0))
    summary_box = ScrolledText(frame, width=75, height=12, wrap="word", state="disabled")
    report: queue.Queue[tuple[str, object]] = queue.Queue()

    def poll() -> None:
        nonlocal folder, finished
        try:
            while True:
                event, payload = report.get_nowait()
                if event == "folder":
                    assert isinstance(payload, Path)
                    folder = payload
                    open_button.state(["!disabled"])
                elif event == "progress":
                    assert isinstance(payload, tuple)
                    progress["value"] = payload[0] * 100
                    status.set(str(payload[1]))
                else:
                    finished = True
                    if event == "done":
                        progress["value"] = 100
                    root.title(
                        "Bass Transcriber — complete"
                        if event == "done"
                        else "Bass Transcriber — failed"
                    )
                    status.set("Complete." if event == "done" else "Finished with errors.")
                    summary_box.configure(state="normal")
                    summary_box.insert(
                        "1.0", str(payload) + (f"\n\nOutput folder: {folder}" if folder else "")
                    )
                    summary_box.configure(state="disabled")
                    summary_box.pack(fill="both", expand=True, pady=(12, 0), before=open_button)
        except queue.Empty:
            pass
        root.after(100, poll)

    def close() -> None:
        if finished:
            root.destroy()
        else:
            root.iconify()

    # Closing during generation minimizes; it must not terminate a running export.
    root.protocol("WM_DELETE_WINDOW", close)
    root.after(100, poll)
    threading.Thread(
        target=run_job,
        args=(args.audio.resolve(), args.output_root, args.strings, args.fingering_profile, report),
        kwargs={"mode": args.mode},
        daemon=True,
    ).start()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
