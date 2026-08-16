"""Native desktop wrapper for the end-to-end transcription pipeline."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from bass_transcriber.pipeline import (
    ProcessingResult,
    debug_path_for,
    output_path_for,
    process_song,
)

_AUDIO_TYPES = [
    ("Music files", "*.mp3 *.wav *.flac *.ogg *.m4a"),
    ("All files", "*.*"),
]

_FINGERING_OPTIONS = {
    "Balanced (recommended)": "balanced",
    "Slap / funk": "slap_funk",
    "Avoid open strings": "avoid_open",
    "Prefer low positions": "low_positions",
    "Compact hand position": "compact",
    "Legacy lowest fret (optimizer off)": None,
}


class TranscriberApp:
    """Small Tk application that runs the pipeline outside the UI thread."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Bass Transcriber")
        self.root.minsize(680, 350)

        self.source = tk.StringVar()
        self.destination = tk.StringVar(value=str((Path.cwd() / "outputs").resolve()))
        self.copy_source = tk.BooleanVar(value=False)
        self.force_electric_bass = tk.BooleanVar(value=False)
        self.five_string = tk.BooleanVar(value=False)
        self.fingering_style = tk.StringVar(value="Balanced (recommended)")
        self.status = tk.StringVar(value="Select a music file and output folder.")
        self.progress = tk.DoubleVar(value=0.0)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()

        self._build_layout()
        self.root.after(100, self._poll_events)

    def _build_layout(self) -> None:
        frame = ttk.Frame(self.root, padding=18)
        frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Music file").grid(row=0, column=0, sticky="w", pady=6)
        ttk.Entry(frame, textvariable=self.source).grid(
            row=0, column=1, sticky="ew", padx=10, pady=6
        )
        ttk.Button(frame, text="Browse…", command=self._choose_source).grid(row=0, column=2, pady=6)

        ttk.Label(frame, text="Output folder").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Entry(frame, textvariable=self.destination).grid(
            row=1, column=1, sticky="ew", padx=10, pady=6
        )
        ttk.Button(frame, text="Browse…", command=self._choose_destination).grid(
            row=1, column=2, pady=6
        )

        ttk.Label(
            frame,
            text="Large MuScriptor",
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(10, 4))

        ttk.Checkbutton(
            frame,
            text="Force electric-bass instrument conditioning (experimental)",
            variable=self.force_electric_bass,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 8))

        ttk.Checkbutton(
            frame,
            text="Also copy the original music file to the output folder",
            variable=self.copy_source,
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(4, 8))

        ttk.Checkbutton(
            frame,
            text="Use 5-string bass (BEADG); turn off for 4-string EADG",
            variable=self.five_string,
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(4, 8))

        ttk.Label(frame, text="Fingering style").grid(row=6, column=0, sticky="w", pady=6)
        ttk.Combobox(
            frame,
            textvariable=self.fingering_style,
            values=tuple(_FINGERING_OPTIONS),
            state="readonly",
        ).grid(row=6, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=6)

        ttk.Progressbar(
            frame,
            variable=self.progress,
            maximum=100.0,
            mode="determinate",
        ).grid(row=7, column=0, columnspan=3, sticky="ew", pady=8)

        ttk.Label(frame, textvariable=self.status, wraplength=620).grid(
            row=8, column=0, columnspan=3, sticky="w", pady=6
        )

        self.process_button = ttk.Button(
            frame,
            text="Process to GP5",
            command=self._start_processing,
        )
        self.process_button.grid(row=9, column=0, columnspan=3, pady=(14, 0))

    def _choose_source(self) -> None:
        selected = filedialog.askopenfilename(title="Select music file", filetypes=_AUDIO_TYPES)
        if selected:
            self.source.set(selected)

    def _choose_destination(self) -> None:
        selected = filedialog.askdirectory(
            title="Select output folder",
            initialdir=self.destination.get(),
        )
        if selected:
            self.destination.set(selected)

    def _start_processing(self) -> None:
        source = Path(self.source.get().strip())
        destination_text = self.destination.get().strip()
        if not source.is_file():
            messagebox.showerror("Invalid music file", "Select an existing music file.")
            return
        if not destination_text:
            messagebox.showerror("Invalid output folder", "Select an output folder.")
            return
        destination = Path(destination_text)
        output = output_path_for(source, destination)
        replacements = [output] if output.exists() else []
        debug_log = debug_path_for(source, destination)
        if debug_log.exists():
            replacements.append(debug_log)
        source_copy = destination / source.name
        if (
            self.copy_source.get()
            and source.resolve() != source_copy.resolve()
            and source_copy.exists()
        ):
            replacements.append(source_copy)
        if replacements:
            names = "\n".join(path.name for path in replacements)
            if not messagebox.askyesno(
                "Replace existing files?",
                f"The following files already exist and will be replaced:\n\n{names}",
            ):
                return

        self.process_button.state(["disabled"])
        self.progress.set(0.0)
        self.status.set("Starting…")
        worker = threading.Thread(
            target=self._run_pipeline,
            args=(
                source,
                destination,
                self.copy_source.get(),
                self.force_electric_bass.get(),
                _FINGERING_OPTIONS[self.fingering_style.get()],
                self.five_string.get(),
            ),
            daemon=True,
        )
        worker.start()

    def _run_pipeline(
        self,
        source: Path,
        destination: Path,
        copy_source: bool,
        force_electric_bass: bool,
        fingering_profile: str | None,
        five_string: bool,
    ) -> None:
        def report(fraction: float, message: str) -> None:
            self.events.put(("progress", (fraction, message)))

        try:
            result = process_song(
                source,
                destination,
                copy_source=copy_source,
                force_electric_bass=force_electric_bass,
                fingering_profile=fingering_profile,
                five_string=five_string,
                progress=report,
            )
        except Exception as error:  # The UI must report backend failures cleanly.
            debug_log = debug_path_for(source, destination)
            detail = str(error)
            if debug_log.is_file():
                detail += f"\n\nMachine-readable debug log: {debug_log}"
            self.events.put(("error", detail))
        else:
            self.events.put(("done", result))

    def _poll_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "progress":
                    if not (
                        isinstance(payload, tuple)
                        and len(payload) == 2
                        and isinstance(payload[0], (int, float))
                        and isinstance(payload[1], str)
                    ):
                        raise TypeError("unexpected progress event")
                    fraction, message = payload
                    self.progress.set(float(fraction) * 100.0)
                    self.status.set(message)
                elif event == "done":
                    result = payload
                    if not isinstance(result, ProcessingResult):
                        raise TypeError("unexpected processing result")
                    self.process_button.state(["!disabled"])
                    self.progress.set(100.0)
                    self.status.set(f"Finished: {result.output}")
                    messagebox.showinfo(
                        (
                            "Transcription complete with warnings"
                            if result.warnings
                            else "Transcription complete"
                        ),
                        f"Created {result.output.name}\n"
                        f"{result.note_count} notes · {result.bpm:.3f} BPM\n"
                        f"Debug log: {result.debug_log.name}"
                        + (
                            f"\nCopied {result.copied_source.name}"
                            if result.copied_source is not None
                            else ""
                        )
                        + (
                            "\n\nWarnings:\n- " + "\n- ".join(result.warnings)
                            if result.warnings
                            else ""
                        ),
                    )
                elif event == "error":
                    self.process_button.state(["!disabled"])
                    self.status.set(f"Failed: {payload}")
                    messagebox.showerror("Processing failed", str(payload))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)


def main() -> int:
    """Open the desktop application."""
    root = tk.Tk()
    TranscriberApp(root)
    root.mainloop()
    return 0
