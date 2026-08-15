# Bass Transcriber

A local Python project for converting songs into synchronized bass tablature that can be opened in ToneLib Jam.

The refined design and V1 scope are documented in [v1plan.md](v1plan.md).

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

## CLI

```powershell
uv run bass-transcriber --version
```
