"""Native desktop wrapper for the end-to-end transcription pipeline."""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from bass_transcriber.fingering_debug import (
    FingeringDebugSession,
    start_fingering_debugger,
)
from bass_transcriber.pipeline import (
    ProcessingResult,
    debug_path_for,
    export_gp5_from_fingering_draft,
    fingering_path_for,
    process_song,
    raw_notes_path_for,
    rewrite_fingering_draft,
)

_AUDIO_TYPES = [
    ("Music files", "*.mp3 *.wav *.flac *.ogg *.m4a"),
    ("All files", "*.*"),
]

_RAW_NOTE_TYPES = [
    ("Raw model output", "*.notes.json"),
    ("JSON files", "*.json"),
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
        self.root.minsize(780, 540)

        self.source = tk.StringVar()
        self.destination = tk.StringVar(value=str((Path.cwd() / "outputs").resolve()))
        self.raw_notes_input = tk.StringVar()
        self.copy_source = tk.BooleanVar(value=False)
        self.force_electric_bass = tk.BooleanVar(value=False)
        self.merge_sustained_retriggers = tk.BooleanVar(value=True)
        self.five_string = tk.BooleanVar(value=False)
        self.fingering_style = tk.StringVar(value="Balanced (recommended)")
        self.status = tk.StringVar(value="Select a music file and output folder.")
        self.progress = tk.DoubleVar(value=0.0)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.latest_result: ProcessingResult | None = None
        self.debug_session: FingeringDebugSession | None = None

        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
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

        ttk.Label(frame, text="Raw model output (optional)").grid(
            row=2, column=0, sticky="w", pady=6
        )
        ttk.Entry(frame, textvariable=self.raw_notes_input).grid(
            row=2, column=1, sticky="ew", padx=10, pady=6
        )
        ttk.Button(frame, text="Browse…", command=self._choose_raw_notes).grid(
            row=2, column=2, pady=6
        )

        ttk.Label(
            frame,
            text="Large MuScriptor",
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(10, 4))

        self.force_electric_checkbox = ttk.Checkbutton(
            frame,
            text="Force electric-bass instrument conditioning (experimental), requires isolated bass track.",  # noqa: E501
            variable=self.force_electric_bass,
        )
        self.force_electric_checkbox.grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(4, 8)
        )

        ttk.Checkbutton(
            frame,
            text=(
                "Merge false sustained-note retriggers using spectral attack detection "
                "(experimental)"
            ),
            variable=self.merge_sustained_retriggers,
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(4, 8))

        ttk.Checkbutton(
            frame,
            text="Also copy the original music file to the output folder",
            variable=self.copy_source,
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(4, 8))

        ttk.Checkbutton(
            frame,
            text="Use 5-string bass (BEADG); turn off for 4-string EADG",
            variable=self.five_string,
        ).grid(row=7, column=0, columnspan=3, sticky="w", pady=(4, 8))

        ttk.Label(frame, text="Selected fingering style").grid(
            row=8, column=0, sticky="w", pady=6
        )
        ttk.Combobox(
            frame,
            textvariable=self.fingering_style,
            values=tuple(_FINGERING_OPTIONS),
            state="readonly",
        ).grid(row=8, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=6)

        ttk.Progressbar(
            frame,
            variable=self.progress,
            maximum=100.0,
            mode="determinate",
        ).grid(row=9, column=0, columnspan=3, sticky="ew", pady=8)

        ttk.Label(frame, textvariable=self.status, wraplength=620).grid(
            row=10, column=0, columnspan=3, sticky="w", pady=6
        )

        self.process_button = ttk.Button(
            frame,
            text="Generate fingering draft",
            command=self._start_processing,
        )
        self.process_button.grid(row=11, column=0, sticky="ew", padx=(0, 5), pady=(14, 0))

        self.compare_button = ttk.Button(
            frame,
            text="Open fingering comparison",
            command=self._open_fingering_comparison,
            state="disabled",
        )
        self.compare_button.grid(row=11, column=1, sticky="ew", padx=5, pady=(14, 0))

        self.apply_fingering_button = ttk.Button(
            frame,
            text="Apply selected fingering",
            command=self._apply_selected_fingering,
            state="disabled",
        )
        self.apply_fingering_button.grid(
            row=11, column=2, sticky="ew", padx=(5, 0), pady=(14, 0)
        )

        self.open_draft_button = ttk.Button(
            frame,
            text="Open fingering JSON",
            command=self._open_fingering_draft,
            state="disabled",
        )
        self.open_draft_button.grid(
            row=12, column=0, columnspan=2, sticky="ew", padx=(0, 5), pady=(8, 0)
        )

        self.export_button = ttk.Button(
            frame,
            text="Validate and convert to GP5",
            command=self._start_draft_export,
            state="disabled",
        )
        self.export_button.grid(
            row=12, column=2, sticky="ew", padx=(5, 0), pady=(8, 0)
        )

        self.raw_notes_input.trace_add("write", self._raw_notes_selection_changed)

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

    def _choose_raw_notes(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select raw model output",
            filetypes=_RAW_NOTE_TYPES,
        )
        if selected:
            self.raw_notes_input.set(selected)

    def _raw_notes_selection_changed(self, *_args: object) -> None:
        if self.raw_notes_input.get().strip():
            self.force_electric_bass.set(False)
            self.force_electric_checkbox.state(["disabled"])
        else:
            self.force_electric_checkbox.state(["!disabled"])

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
        raw_notes_text = self.raw_notes_input.get().strip()
        raw_notes_input = Path(raw_notes_text) if raw_notes_text else None
        if raw_notes_input is not None and not raw_notes_input.is_file():
            messagebox.showerror(
                "Invalid raw model output",
                "Select an existing .notes.json file or clear the optional field.",
            )
            return
        fingering_draft = fingering_path_for(source, destination)
        replacements = [fingering_draft] if fingering_draft.exists() else []
        debug_log = debug_path_for(source, destination)
        if debug_log.exists():
            replacements.append(debug_log)
        raw_notes_output = raw_notes_path_for(source, destination)
        if raw_notes_output.exists() and raw_notes_output not in replacements:
            replacements.append(raw_notes_output)
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

        self._stop_debugger()
        self.latest_result = None
        self.process_button.state(["disabled"])
        self.compare_button.state(["disabled"])
        self.apply_fingering_button.state(["disabled"])
        self.open_draft_button.state(["disabled"])
        self.export_button.state(["disabled"])
        self.progress.set(0.0)
        self.status.set("Starting…")
        worker = threading.Thread(
            target=self._run_pipeline,
            args=(
                source,
                destination,
                self.copy_source.get(),
                self.force_electric_bass.get(),
                self.merge_sustained_retriggers.get(),
                raw_notes_input,
                _FINGERING_OPTIONS[self.fingering_style.get()],
                self.five_string.get(),
            ),
            daemon=True,
        )
        worker.start()

    def _open_fingering_comparison(self) -> None:
        result = self.latest_result
        if result is None or result.source is None or result.rhythm is None:
            messagebox.showerror(
                "No completed transcription",
                "Process a song before opening the fingering comparison.",
            )
            return
        if self.debug_session is not None:
            self.debug_session.open_browser()
            self.status.set(f"Fingering comparison: {self.debug_session.url}")
            return
        try:
            self.debug_session = start_fingering_debugger(
                result.source,
                result.notes,
                result.rhythm,
                five_string=result.five_string,
                open_browser=True,
            )
        except (OSError, ValueError) as error:
            messagebox.showerror("Could not open comparison", str(error))
            return
        self.status.set(f"Fingering comparison: {self.debug_session.url}")

    def _apply_selected_fingering(self) -> None:
        result = self.latest_result
        if result is None or result.fingering_draft is None:
            messagebox.showerror(
                "No completed transcription",
                "Process a song before applying a fingering style.",
            )
            return
        style_label = self.fingering_style.get()
        if not messagebox.askyesno(
            "Apply fingering style?",
            f"Replace {result.fingering_draft.name} using {style_label}?\n\n"
            "Any manual edits in the draft will be lost.",
        ):
            return
        profile = _FINGERING_OPTIONS[style_label]
        self._set_action_buttons_enabled(False)
        self.status.set(f"Applying {style_label}…")
        worker = threading.Thread(
            target=self._run_fingering_draft,
            args=(result, profile, style_label),
            daemon=True,
        )
        worker.start()

    def _open_fingering_draft(self) -> None:
        result = self.latest_result
        if result is None or result.fingering_draft is None:
            messagebox.showerror(
                "No fingering draft",
                "Generate a fingering draft before opening it.",
            )
            return
        if not result.fingering_draft.is_file():
            messagebox.showerror(
                "Missing fingering draft",
                f"The draft no longer exists:\n{result.fingering_draft}",
            )
            return
        try:
            _open_local_file(result.fingering_draft)
        except OSError as error:
            messagebox.showerror("Could not open fingering draft", str(error))
            return
        self.status.set(f"Editing fingering draft: {result.fingering_draft}")

    def _start_draft_export(self) -> None:
        result = self.latest_result
        if result is None or result.fingering_draft is None:
            messagebox.showerror(
                "No fingering draft",
                "Generate a fingering draft before converting it to GP5.",
            )
            return
        if not result.fingering_draft.is_file():
            messagebox.showerror(
                "Missing fingering draft",
                f"The draft no longer exists:\n{result.fingering_draft}",
            )
            return
        if result.output.exists() and not messagebox.askyesno(
            "Replace existing GP5?",
            f"Replace {result.output.name} with the current fingering draft?",
        ):
            return
        self._set_action_buttons_enabled(False)
        self.status.set(f"Validating {result.fingering_draft.name}…")
        threading.Thread(
            target=self._run_draft_export,
            args=(result,),
            daemon=True,
        ).start()

    def _run_fingering_draft(
        self,
        result: ProcessingResult,
        profile: str | None,
        style_label: str,
    ) -> None:
        try:
            export_result = rewrite_fingering_draft(result, profile)
        except Exception as error:  # The UI must report backend failures cleanly.
            self.events.put(("fingering_error", str(error)))
        else:
            updated = replace(result, fingering_profile=profile)
            self.events.put(
                (
                    "fingering_done",
                    (updated, style_label, export_result.exported_note_count),
                )
            )

    def _run_draft_export(self, result: ProcessingResult) -> None:
        try:
            export_result = export_gp5_from_fingering_draft(result)
        except Exception as error:  # The UI must report backend failures cleanly.
            self.events.put(("draft_export_error", str(error)))
        else:
            self.events.put(("draft_export_done", (result, export_result.exported_note_count)))

    def _set_action_buttons_enabled(self, enabled: bool) -> None:
        state = ["!disabled"] if enabled else ["disabled"]
        self.process_button.state(state)
        self.compare_button.state(state)
        self.apply_fingering_button.state(state)
        self.open_draft_button.state(state)
        self.export_button.state(state)

    def _stop_debugger(self) -> None:
        if self.debug_session is None:
            return
        self.debug_session.stop()
        self.debug_session = None

    def _close(self) -> None:
        self._stop_debugger()
        self.root.destroy()

    def _run_pipeline(
        self,
        source: Path,
        destination: Path,
        copy_source: bool,
        force_electric_bass: bool,
        merge_sustained_retriggers: bool,
        raw_notes_input: Path | None,
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
                merge_sustained_retriggers=merge_sustained_retriggers,
                raw_notes_input=raw_notes_input,
                fingering_profile=fingering_profile,
                five_string=five_string,
                generate_gp5=False,
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
                    self._set_action_buttons_enabled(True)
                    self.latest_result = result
                    self.progress.set(100.0)
                    if result.fingering_draft is None:
                        raise TypeError("processing result is missing its fingering draft")
                    self.status.set(f"Fingering draft ready: {result.fingering_draft}")
                    messagebox.showinfo(
                        (
                            "Transcription complete with warnings"
                            if result.warnings
                            else "Transcription complete"
                        ),
                        f"Created editable draft {result.fingering_draft.name}\n"
                        f"{result.note_count} notes · {result.bpm:.3f} BPM\n"
                        "Edit the JSON if needed, then choose Validate and convert to GP5.\n"
                        f"Debug log: {result.debug_log.name}"
                        + (
                            f"\nRaw model output: {result.raw_notes.name}"
                            if result.raw_notes is not None
                            else ""
                        )
                        + (
                            "\nMuScriptor skipped: reused selected raw output"
                            if result.reused_raw_notes
                            else ""
                        )
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
                elif event == "fingering_done":
                    if not (
                        isinstance(payload, tuple)
                        and len(payload) == 3
                        and isinstance(payload[0], ProcessingResult)
                        and isinstance(payload[1], str)
                        and isinstance(payload[2], int)
                    ):
                        raise TypeError("unexpected fingering export result")
                    result, style_label, note_count = payload
                    self.latest_result = result
                    self._set_action_buttons_enabled(True)
                    self.status.set(f"Applied {style_label}: {result.fingering_draft}")
                    messagebox.showinfo(
                        "Fingering applied",
                        f"Updated {result.fingering_draft.name} using {style_label}.\n"
                        f"Resolved {note_count} notes. You can now edit or export the draft.",
                    )
                elif event == "fingering_error":
                    self._set_action_buttons_enabled(True)
                    self.status.set(f"Could not apply fingering: {payload}")
                    messagebox.showerror("Fingering draft failed", str(payload))
                elif event == "draft_export_done":
                    if not (
                        isinstance(payload, tuple)
                        and len(payload) == 2
                        and isinstance(payload[0], ProcessingResult)
                        and isinstance(payload[1], int)
                    ):
                        raise TypeError("unexpected draft export result")
                    result, note_count = payload
                    self._set_action_buttons_enabled(True)
                    self.progress.set(100.0)
                    self.status.set(f"Created GP5: {result.output}")
                    messagebox.showinfo(
                        "GP5 export complete",
                        f"Validated {result.fingering_draft.name} and created "
                        f"{result.output.name}.\nExported {note_count} notes.",
                    )
                elif event == "draft_export_error":
                    self._set_action_buttons_enabled(True)
                    self.status.set(f"Could not convert draft: {payload}")
                    messagebox.showerror("Fingering draft is invalid", str(payload))
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


def _open_local_file(path: Path) -> None:
    """Open a local artifact in the platform's associated application."""
    windows_opener = getattr(os, "startfile", None)
    if windows_opener is not None:
        windows_opener(path)
        return
    command = ["open", str(path)] if sys.platform == "darwin" else ["xdg-open", str(path)]
    subprocess.Popen(command)  # noqa: S603 - explicit local file selected by the user
