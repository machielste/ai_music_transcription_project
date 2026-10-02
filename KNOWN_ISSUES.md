# Known issues

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
