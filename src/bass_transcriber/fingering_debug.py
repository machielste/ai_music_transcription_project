"""Local browser debugger for comparing fingering strategies."""

from __future__ import annotations

import json
import mimetypes
import threading
import webbrowser
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

from bass_transcriber.models import BassNote, RhythmGrid
from bass_transcriber.tab import (
    BEADG_STRINGS,
    EADG_STRINGS,
    FINGERING_PROFILES,
    ResolvedTabNote,
    build_fingering_timeline,
    source_aligned_seconds,
)

DEFAULT_DEBUG_PROFILES: tuple[str, ...] = (
    "balanced",
    "slap_funk",
    "avoid_open",
    "low_positions",
    "compact",
    "legacy",
)

_PROFILE_LABELS = {
    "balanced": "Balanced",
    "slap_funk": "Slap / funk",
    "avoid_open": "Avoid open strings",
    "low_positions": "Low positions",
    "compact": "Compact movement",
    "legacy": "Legacy lowest fret",
}

_STRING_NAMES_BY_OPEN_PITCH = {
    43: "G",
    38: "D",
    33: "A",
    28: "E",
    23: "B",
}


def build_comparison_document(
    notes: Sequence[BassNote],
    grid: RhythmGrid,
    *,
    profiles: Sequence[str] = DEFAULT_DEBUG_PROFILES,
    five_string: bool = True,
    source_name: str = "Source audio",
) -> dict[str, object]:
    """Build the JSON document consumed by the browser debugger."""
    if not notes:
        raise ValueError("cannot debug an empty transcription")
    if not profiles:
        raise ValueError("select at least one fingering profile")

    unknown = [
        profile
        for profile in profiles
        if profile != "legacy" and profile not in FINGERING_PROFILES
    ]
    if unknown:
        available = ", ".join((*FINGERING_PROFILES, "legacy"))
        raise ValueError(f"unknown fingering profile {unknown[0]!r}; choose from {available}")

    strings = BEADG_STRINGS if five_string else EADG_STRINGS
    strategy_documents: list[dict[str, object]] = []
    playback_notes: list[dict[str, object]] | None = None
    duration_seconds = 0.0

    for profile in profiles:
        timeline = build_fingering_timeline(
            notes,
            grid,
            None if profile == "legacy" else profile,
            strings=strings,
        )
        resolved_notes = [_note_document(note, grid) for note in timeline.notes]
        if playback_notes is None:
            playback_notes = [
                {
                    "pitch": note["pitch"],
                    "start_seconds": note["start_seconds"],
                    "end_seconds": note["end_seconds"],
                }
                for note in resolved_notes
            ]
        if resolved_notes:
            duration_seconds = max(
                duration_seconds,
                max(cast(float, note["end_seconds"]) for note in resolved_notes),
            )
        strategy_documents.append(
            {
                "profile": profile,
                "label": _PROFILE_LABELS[profile],
                "notes": resolved_notes,
                "metrics": _timeline_metrics(timeline.notes),
                "dropped_pitches": list(timeline.dropped_pitches),
            }
        )

    if not playback_notes:
        tuning_name = "BEADG" if five_string else "EADG"
        raise ValueError(f"cannot debug: every note is outside the {tuning_name} range")

    return {
        "schema_version": 1,
        "document_type": "bass_fingering_comparison",
        "source_name": source_name,
        "bpm": grid.bpm,
        "onset_delay_seconds": grid.onset_delay_seconds,
        "duration_seconds": duration_seconds,
        "reference_profile": cast(str, strategy_documents[0]["profile"]),
        "strings": [
            {
                "number": number,
                "name": _STRING_NAMES_BY_OPEN_PITCH[open_pitch],
                "open_pitch": open_pitch,
            }
            for number, open_pitch in strings
        ],
        "playback_notes": playback_notes,
        "strategies": strategy_documents,
    }


def serve_fingering_debugger(
    audio: Path,
    notes: Sequence[BassNote],
    grid: RhythmGrid,
    *,
    profiles: Sequence[str] = DEFAULT_DEBUG_PROFILES,
    five_string: bool = True,
    port: int = 0,
    open_browser: bool = True,
) -> None:
    """Serve the comparison UI on loopback until interrupted."""
    session = start_fingering_debugger(
        audio,
        notes,
        grid,
        profiles=profiles,
        five_string=five_string,
        port=port,
        open_browser=open_browser,
    )
    print(f"Fingering debugger: {session.url}")
    print("Press Ctrl+C to stop the debugger.")
    try:
        session.wait()
    except KeyboardInterrupt:
        pass
    finally:
        session.stop()


def start_fingering_debugger(
    audio: Path,
    notes: Sequence[BassNote],
    grid: RhythmGrid,
    *,
    profiles: Sequence[str] = DEFAULT_DEBUG_PROFILES,
    five_string: bool = True,
    port: int = 0,
    open_browser: bool = True,
) -> FingeringDebugSession:
    """Start a loopback debugger in a daemon thread and return its session."""
    if not audio.is_file():
        raise FileNotFoundError(f"source audio does not exist: {audio}")
    comparison = build_comparison_document(
        notes,
        grid,
        profiles=profiles,
        five_string=five_string,
        source_name=audio.name,
    )
    html = (
        files("bass_transcriber.debug_ui")
        .joinpath("fingering_highway.html")
        .read_bytes()
    )
    server = _FingeringDebugServer(
        ("127.0.0.1", port),
        _FingeringDebugHandler,
        html=html,
        comparison_json=(json.dumps(comparison, separators=(",", ":")) + "\n").encode(),
        audio_path=audio,
    )
    bound_host, bound_port = cast(tuple[str, int], server.server_address)
    url = f"http://{bound_host}:{bound_port}/"
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.25},
        name="fingering-debugger",
        daemon=True,
    )
    session = FingeringDebugSession(server, thread, url)
    thread.start()
    if open_browser:
        webbrowser.open(url)
    return session


def _note_document(note: ResolvedTabNote, grid: RhythmGrid) -> dict[str, object]:
    return {
        "pitch": note.pitch,
        "start_slot": note.start_slot,
        "end_slot": note.end_slot,
        "start_seconds": source_aligned_seconds(note.start_slot, grid),
        "end_seconds": source_aligned_seconds(note.end_slot, grid),
        "string": note.string,
        "fret": note.fret,
    }


def _timeline_metrics(notes: Sequence[ResolvedTabNote]) -> dict[str, int]:
    fret_travel = 0
    string_changes = 0
    string_skips = 0
    for previous, current in zip(notes, notes[1:], strict=False):
        fret_travel += abs(current.fret - previous.fret)
        string_distance = abs(current.string - previous.string)
        if string_distance:
            string_changes += 1
            string_skips += max(0, string_distance - 1)
    return {
        "fret_travel": fret_travel,
        "string_changes": string_changes,
        "string_skips": string_skips,
        "open_strings": sum(note.fret == 0 for note in notes),
        "max_fret": max((note.fret for note in notes), default=0),
    }


class _FingeringDebugServer(ThreadingHTTPServer):
    html: bytes
    comparison_json: bytes
    audio_path: Path

    def __init__(
        self,
        server_address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        *,
        html: bytes,
        comparison_json: bytes,
        audio_path: Path,
    ) -> None:
        self.html = html
        self.comparison_json = comparison_json
        self.audio_path = audio_path
        super().__init__(server_address, handler)


class _FingeringDebugHandler(BaseHTTPRequestHandler):
    server: _FingeringDebugServer

    def do_GET(self) -> None:  # noqa: N802 - inherited HTTP method name
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            self._send_bytes(self.server.html, "text/html; charset=utf-8")
            return
        if path == "/comparison.json":
            self._send_bytes(
                self.server.comparison_json,
                "application/json; charset=utf-8",
            )
            return
        if path == "/audio":
            self._send_audio()
            return
        if path == "/health":
            self._send_bytes(b"ok\n", "text/plain; charset=utf-8")
            return
        self.send_error(404)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_bytes(self, payload: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_audio(self) -> None:
        audio = self.server.audio_path
        content_type = mimetypes.guess_type(audio.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(audio.stat().st_size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with audio.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                self.wfile.write(chunk)


class FingeringDebugSession:
    """A running local debugger that can be reopened or stopped by a GUI."""

    def __init__(
        self,
        server: _FingeringDebugServer,
        thread: threading.Thread,
        url: str,
    ) -> None:
        self._server = server
        self._thread = thread
        self.url = url
        self._stopped = False

    def open_browser(self) -> None:
        """Open another browser tab for this session."""
        webbrowser.open(self.url)

    def wait(self) -> None:
        """Wait until the server stops, remaining interruptible by Ctrl+C."""
        while self._thread.is_alive():
            self._thread.join(timeout=0.5)

    def stop(self) -> None:
        """Stop the local server and release its listening socket."""
        if self._stopped:
            return
        self._stopped = True
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
