# Known issues

## Missing five-string choices in the Explorer menu (resolved)

On 2026-10-14, Explorer displayed only **Punish open strings** under
**ToneLib Only → 5-string bass**, despite all three strategy commands being
present in the automation host's registry view. Direct Python installs and their
registry verification ran in an isolated view, so even restarting Explorer did
not apply those updates. The existing `scripts/windows-context-menu.ps1` wrapper
avoids this by registering in a temporary signed-in-user task. Always use that
wrapper when installing from a desktop automation host.

The old cascade contained 18 entries including
submenu headings, and Explorer omitted the last two, consistent with a
16-entry budget for this static cascade.

The installer now puts six choices directly under each output mode, labelled
with their string count, tuning, and strategy. This reduces the cascade to 14
entries; for example, use **ToneLib Only → 5-string (BEADG) - Combined**.
The command arguments and the five-string B-string penalty are unchanged.
After installation through the PowerShell wrapper, Explorer was visually verified
to display all six choices in both ToneLib Only and Debug. All 40 menu and
fingering tests pass.

## ToneLib stays on an earlier song when opened through Windows

**Status: observed in the installed ToneLib Jam; use its internal file picker.**

On 2026-10-14, opening another generated album project through Windows did not
switch to the requested song. ToneLib continued to show the startup filename
`01-SUM~1`. The same problem occurred when passing the requested project path
to the installed `ToneLib-Jam.exe` launcher. The user also reports a confirmation
about losing changes followed by the previous song appearing again; that exact
confirmation sequence was not reproduced during the check with no pending edits.

Opening `02 - Kusuri wo Takusan.bass.song` using ToneLib's **Open → Browse**
successfully changed the filename, backing waveform, and tempo (105 to 126 BPM).
The same internal picker also successfully loaded `03 - Nani mo Iranai.bass.song`
with its different waveform and 103 BPM tempo.
The album archives contain distinct titles, score data, and embedded-audio
identifiers. No audio filename-hash collisions between different recording names
were found in the inspected local projects, and the bundled template has no
project-wide song ID. All 11 ToneLib exporter tests passed.

This points to ToneLib's handling of files opened from Windows while an instance
is running. The precise internal cause remains unconfirmed; it is not evidence
for changing the export's audio IDs or inventing a new song ID.

The short names shown in ToneLib's recent-song list are Windows 8.3 aliases.
Calling `GetShortPathNameW` on the album files reproduced the displayed folder
`1977-S~1.COM` and filenames such as `01-SUM~1.SON` and `04-TOK~1.SON`.
`GetLongPathNameW` resolved each of the first four aliases back to its correct,
distinct original file. `.COM` here abbreviates the folder's `.combined` suffix;
it is not a website. These aliases explain the displayed names, but their valid
resolution does not by itself explain the failure to switch songs.

**Workaround:** use **File → Open → Browse** inside ToneLib to switch songs.

## Repeated false notes during near-silent bass passages

**Priority: low. Status: documented; remediation deferred.**

MuScriptor Large with electric-bass conditioning can generate repeated false
notes during near-silent passages in an otherwise accurate separated bass stem.
The user considers this tolerable for the playability of the current output.
Do not add a complex silence detector or change carryover defaults as part of
the stem-separation integration.

Observed on 2026-10-02 in `10. NH-CA's Struttin' (Crossing At Airport)(Sanae)`:

- The BS-RoFormer SW bass stem is nearly silent after approximately 3:22.
- The raw MuScriptor event stream contains 99 G-sharp2 / MIDI 44 note starts
  between 3:20 and 3:35, corresponding to the first fret on the G string.
- Independent inference on the 3:25–3:30 chunk produces zero notes with both
  automatic classification and electric-bass conditioning.
- A diagnostic rerun of the first 215 seconds with `prelude_forcing=False`
  produces zero note starts between 3:20 and 3:35. Earlier predictions also
  change, so this does not establish that disabling carryover globally is safe.

These observations support forced cross-chunk note carryover as a contributor
to the failure. The events are present before quantization and fingering; this
is not an export-only glitch. The public MuScriptor note events do not expose
per-note confidence scores.

Detailed local evidence is retained in the song's output folder under
`stem-transcription-large-electric-bass/bridge-diagnostics/`, including
`findings.md`, `original-raw-events.csv`, `bridge-analysis.json`, and
`carryover-disabled.json`. Output artifacts are ignored by Git.

Revisit if the problem becomes common or materially affects usability.
