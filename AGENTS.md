# Project guidance for agents

## Project and entry points

This project converts recordings into synchronized four- or five-string bass
tablature and self-contained ToneLib Jam projects. Read `README.md` for current
behavior and `KNOWN_ISSUES.md` for observed limitations and deferred fixes.
`v1plan.md` is historical design context; check the current implementation before
treating its scope as a requirement.

- `src/bass_transcriber/pipeline.py`: processing stages and artifact generation.
- `src/bass_transcriber/gui.py`: desktop interface.
- `src/bass_transcriber/shell_job.py`: Explorer jobs and recursive folder runs.
- `src/bass_transcriber/context_menu.py`: Explorer menu definitions and registry writes.
- `scripts/windows-context-menu.ps1`: interactive-user menu installer.
- `src/bass_transcriber/tab/`: fingering, rhythm quantization, and note placement.
- `src/bass_transcriber/export/`: GP5, MIDI, and ToneLib export.
- `tests/`: automated checks.

## Windows Explorer registration: important isolation trap

**Always use the PowerShell wrapper when installing or removing the Explorer
menu from Codex or another desktop automation host:**

```powershell
powershell -NoProfile -File .\scripts\windows-context-menu.ps1 install
powershell -NoProfile -File .\scripts\windows-context-menu.ps1 uninstall
```

The wrapper runs the Python installer through a temporary Task Scheduler task in
the signed-in user's session, then removes that task. This avoids the isolated
registry view inherited from the automation host.

**Do not call `python -m bass_transcriber.context_menu install` directly from
the automation host and assume it changes the menu Explorer uses.** That command
can report success while writing to an isolated registry. Reading HKCU or HKCR
from the same host can confirm those writes and still give a misleading result.
An unsandboxed command alone did not resolve this isolation in the observed case.
Restarting Explorer cannot make isolated registry changes visible.

This failure was reproduced on 2026-10-14: direct installs and registry checks
passed, but the visible menu did not change even after Explorer restarts. Running
the existing PowerShell wrapper made the revised menu visible.

For menu changes:

1. Update `context_menu.py` and its tests.
2. Install through the PowerShell wrapper.
3. Check the actual Explorer menu on a supported music file or music folder.
   Verify both ToneLib Only and Debug, both string counts, and all three styles.
   Inspect the choices without accidentally starting transcription.
4. Report a visible fix only after checking the UI or receiving user confirmation.
   Treat unit tests and host registry checks as supporting evidence.

The current menu has six labelled choices directly under each output mode, such
as `5-string (BEADG) - Combined`. Its cascade contains 14 entries including
submenu headings. The previous deeper hierarchy contained 18 and displayed an
incomplete five-string branch. Keep the menu-budget regression test and verify
any deeper layout in Explorer. Modify only this project's owned menu keys; do
not replace unrelated file associations. Do not use Explorer restarts as a
substitute for checking the registration context.

## Audio, fingering, and export invariants

- Preserve existing output runs. Repeated jobs reserve new folders rather than
  overwriting previous results. Keep local recordings, model weights, generated
  outputs, and diagnostics out of Git.
- Retain raw transcription data and provenance so fingering and export changes
  can be evaluated without repeating model inference unnecessarily.
- Use the original recording for rhythm, synchronized playback, and ToneLib
  backing audio. The separated bass stem serves transcription and spectral cleanup.
- Guitar Pro string numbers run highest to lowest: G is 1; low B is 5 in BEADG.
  Four-string EADG must not produce B-string placements.
- Balanced, Punish open strings (`avoid_open`), and Combined are supported for
  both tunings. Combined retains the five-string penalty for B-string positions
  at fret 5 and above. It is a soft penalty, not a ban on necessary placements.
- ToneLib audio archive IDs must match `_audio_entry_name(audio.name)`, including
  filename extension and case. Do not substitute random IDs or content hashes.
  The proprietary format is inferred; verify compatibility-sensitive changes in
  the installed app as well as in archive tests.
- Names such as `01-SUM~1` in ToneLib's recent list are Windows 8.3 aliases, not
  song IDs. Check their resolution before attributing failures to ID collisions.
  ToneLib's internal File → Open → Browse worked when external opening did not.
- Respect the deferred near-silence false-note issue in `KNOWN_ISSUES.md`; do not
  change carryover defaults or add a complex detector as an unrelated fix.

## Development and verification

Use the existing Python 3.12 environment and `uv` project configuration. Standard
checks are:

```powershell
uv run pytest
uv run ruff check .
uv run mypy src
```

Run checks appropriate to the change. Fingering and Explorer changes should cover
`tests/test_fingering.py` and `tests/test_context_menu.py`; ToneLib export changes
should cover `tests/test_tonelib.py`. Prefer existing fixtures and mocked model
inference for routine verification. Run expensive transcription only when needed
to establish behavior that those checks cannot cover.

If the sandbox cannot create pytest temporary directories, use a new, unused
`--basetemp` directory under `outputs/` and point `TEMP` and `TMP` at an ignored
workspace directory for that command. Pytest may clear an existing basetemp:
never point it at a real output run or other user data.

Inspect the working tree before editing, preserve unrelated changes, and keep
documentation aligned with the final behavior. State what was tested and
distinguish verified results from hypotheses.
