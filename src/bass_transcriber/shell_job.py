"""One-click Explorer job with a responsive progress window."""

from __future__ import annotations

import argparse
import os
import queue
import threading
import tkinter as tk
from contextlib import suppress
from pathlib import Path
from tkinter import ttk

from bass_transcriber.context_menu import PROFILES
from bass_transcriber.output_folders import create_run_folder


def run_job(
    source: Path,
    output_root: Path,
    strings: int,
    profile: str,
    report: queue.Queue[tuple[str, object]],
) -> None:
    """Generate all artifacts, retaining the job folder and diagnostics on failure."""
    destination: Path | None = None
    try:
        if not source.is_file():
            raise FileNotFoundError(f"Music file does not exist: {source}")
        destination = create_run_folder(source, output_root, strings, profile)
        report.put(("folder", destination))
        # Load the backend after the window is visible, inside the worker.
        from bass_transcriber.pipeline import process_song

        result = process_song(
            source,
            destination,
            copy_source=True,
            five_string=strings == 5,
            fingering_profile=profile,
            merge_sustained_retriggers=True,
            generate_gp5=True,
            generate_tonelib=True,
            progress=lambda fraction, message: report.put(("progress", (fraction, message))),
        )
        report.put(
            (
                "done",
                f"Created {result.output.name}\n"
                + (f"Created {result.tonelib_song.name}\n" if result.tonelib_song else "")
                + f"{result.note_count} notes · {result.bpm:.3f} BPM"
                + ("\n\nWarnings:\n" + "\n".join(result.warnings) if result.warnings else ""),
            )
        )
    except Exception as error:
        detail = f"{type(error).__name__}: {error}"
        if destination is not None:
            # The window must still report errors if the output folder becomes unwritable.
            with suppress(OSError):
                (destination / "error.txt").write_text(detail, encoding="utf-8")
        report.put(("error", detail))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--strings", type=int, choices=(4, 5), required=True)
    parser.add_argument("--fingering-profile", choices=PROFILES, required=True)
    args = parser.parse_args()
    root = tk.Tk()
    root.title("Bass Transcriber — generating tablature")
    frame = ttk.Frame(root, padding=20)
    frame.pack(fill="both", expand=True)
    ttk.Label(
        frame,
        text=f"{args.audio.name}\n{args.strings}-string bass · {PROFILES[args.fingering_profile]}",
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
                    status.set(str(payload) + (f"\n\nOutput folder: {folder}" if folder else ""))
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
        daemon=True,
    ).start()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
