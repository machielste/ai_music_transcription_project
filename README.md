# Bass Transcriber

A local Python project for converting songs into synchronized bass tablature that can be opened in ToneLib Jam.

The refined design and V1 scope are documented in [v1plan.md](v1plan.md).

## Disclosure

THIS REPO INCLUDES AI ASSISTED CODE

## Development setup

Install [uv](https://docs.astral.sh/uv/), then run:

```powershell
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
```

The repository pins its uv-managed Python version in `.python-version`.

On Windows and Linux, uv resolves PyTorch from the official CUDA 12.8 wheel index.
Model weights are not stored in this repository; MuScriptor downloads them into the
standard Hugging Face cache on first use.

## MuScriptor access

MuScriptor checkpoints are gated on Hugging Face. Before fetching one:

1. Accept the conditions on the desired `MuScriptor` model page on Hugging Face.
2. Authenticate locally with `uvx hf auth login`.
3. Check the local environment with `uv run bass-transcriber doctor`.

We begin with the small checkpoint to validate the integration. The medium and large
checkpoints can be evaluated after the smoke test succeeds.

```powershell
uv run bass-transcriber model fetch small
```

## First transcription

Run the small checkpoint with hard electric-bass conditioning and write raw,
unquantized notes to a JSON sidecar:

```powershell
uv run bass-transcriber transcribe path\to\audio.wav
```

The default output is `path\to\audio.notes.json`. Use `--instrument acoustic_bass`
for an acoustic-bass source. At this stage JSON is intentionally a diagnostic
artifact; rhythm inference, fingering, and GP5 export come later.

For a full mix, let MuScriptor classify instruments and retain only events it
labels as electric or acoustic bass:

```powershell
uv run bass-transcriber transcribe song.wav --model large --instrument auto
```

Export the raw notes to MIDI for an auditory comparison with the source:

```powershell
uv run bass-transcriber export midi outputs\smoke.notes.json
```

This preserves note times but uses 120 BPM only as a MIDI timing carrier. It does
not claim that 120 BPM is the detected musical tempo.

Detect the song's beat grid without changing the raw note timing:

```powershell
uv run bass-transcriber rhythm detect outputs\song.large.auto.notes.json
```

The command prefers Beat This when its official checkpoint is cached and otherwise
uses librosa's beat tracker. The fallback reports meter as unknown instead of
guessing downbeats.

Write a first-pass five-string BEADG Guitar Pro 5 score:

```powershell
uv run bass-transcriber export gp5 outputs\song.large.auto.notes.json `
  --rhythm outputs\song.large.auto.rhythm.json
```

Until a downbeat detector is available, GP5 export assumes 4/4 and treats the
detected beat phase as an arbitrary bar phase. Note timing and pitch remain usable;
barlines can be shifted later.

GP5 stores integer tempo values. Fractional detected tempos are approximated with
hidden, error-diffused tempo changes at measure boundaries—for example, alternating
117 and 118 BPM to track a detected 117.454 BPM without accumulating drift.

## CLI

```powershell
uv run bass-transcriber --version
```

## Desktop interface

Open the native file-picker interface with:

```powershell
uv run bass-transcriber-gui
```

Select a music file and destination folder, then choose **Process to GP5**. The UI
runs the large-model automatic-instrument pipeline in the background and writes
`<song>.bass.gp5` into the selected folder. Temporary WAV and JSON artifacts are
removed after a successful or failed run. Enable **Also copy the original music
file to the output folder** when you want the source audio placed beside the GP5.
