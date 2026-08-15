# Refined V1 Plan: Song to Synchronized ToneLib Bass Tab

## Summary

Build a Windows-first, local Python 3.12 project for personal/noncommercial use:

`MP3/WAV/FLAC -> rhythm + bass-note transcription -> quantized EADG tab -> GP5 + synchronized WAV`

Each run produces:

- `song_bass.gp5` — editable bass score for ToneLib Jam.
- `song_backing.wav` — original audio with silence prepended so its first downbeat aligns with a GP5 barline.
- `song_analysis.json` — lossless raw notes, beat grid, rejected notes, warnings, model versions, hashes, and settings.
- Optional diagnostic MIDI.

V1 targets an editable draft for mostly monophonic rock/pop/metal bass, standard four-string E1-A1-D2-G2 tuning, and up to 24 frets.

## Critique and Architectural Corrections

- Keep the neutral internal representation, benchmark harness, dynamic-programming fingering, and GP5 target; these are the strongest parts of the original proposal.
- Make source separation optional. Benchmark full-mix MuScriptor against `htdemucs` and `htdemucs_ft` stems; select by downstream bass-note F1, not separation SDR or perceived stem quality. Demucs is a sound baseline, but not an assumed permanent winner. [Demucs](https://pypi.org/project/demucs/)
- Treat MuScriptor as a new general AMT model, not a specialized bass model. Use the actual constraints `electric_bass` and `acoustic_bass`; it does not emit calibrated confidence or velocity. Use greedy decoding, CFG 2, and chunk prelude forcing initially; benchmark medium versus large instead of assuming 1.3B is worth its extra cost. Its gated weights are CC BY-NC and must not be bundled. [MuScriptor model card](https://huggingface.co/MuScriptor/muscriptor-large), [event API](https://github.com/muscriptor/muscriptor/blob/main/muscriptor/events.py)
- Defer the proposed ensemble. Low bass is not intrinsically easy: fundamentals may be weak, kick bleed is correlated with bass onsets, and long pitch-analysis windows reduce onset precision. Basic Pitch, CREPE/pYIN, onset detection, and harmonic evidence become V2 candidates only after ablations show improvement; do not majority-vote correlated detectors.
- Remove key, tuning inference, velocity inference, and articulation recognition from V1. Tuning is explicit; key is unnecessary for tablature; unsupported dynamics use a fixed GP5 velocity; only structural ties are generated.
- Do not force strict monophony. Preserve feasible double-stops or ringing notes, while flagging impossible overlaps.
- Run only Demucs and MuScriptor on CUDA, sequentially or in isolated worker processes. Audio conversion, quantization, fingering, and GP5 generation remain CPU tasks. Configure an explicit CUDA 12.8-compatible PyTorch build for the RTX 5080.
- Treat timing as a first-class subsystem. A single BPM cannot reliably keep a score synchronized to a real recording. Track beats/downbeats on the full mix, create a compressed hidden tempo map, and refuse export when rhythm confidence or resulting synchronization is inadequate.
- GP5 contains the score, not the recording. ToneLib documents backing audio as a separate drag-and-drop track, so a synchronized companion WAV is part of the deliverable. [ToneLib Jam manual](https://tonelib.net/doc/tonelib-jam-manual.pdf)

## Interfaces and Implementation

### Project foundation

- Replace the PyCharm stub with a `uv`-managed package under `src/bass_transcriber`, with separate `tests` and `benchmarks`.
- Pin Python 3.12, CUDA-enabled PyTorch, MuScriptor, Demucs 4.1, Beat This, PyGuitarPro 0.11, and evaluation dependencies in a lockfile.
- Add `bass-transcriber doctor` to verify FFmpeg, Hugging Face authentication/license acceptance, CUDA availability, RTX compute capability, dependency versions, model access, and a small CUDA forward pass.
- Cache expensive stage outputs using the input hash, complete configuration, package versions, and model/checkpoint hashes. Never store credentials.

### Public API

```python
result = transcribe_bass(
    "song.mp3",
    TranscriptionOptions(
        tuning=("E1", "A1", "D2", "G2"),
        frets=24,
        model="auto",
        audio_source="auto",
        meter="auto",
    ),
)

result.export("output/song")
```

CLI:

```text
bass-transcriber transcribe INPUT --out-dir OUTPUT
    [--model auto|medium|large]
    [--audio-source auto|mix|htdemucs|htdemucs-ft]
    [--tuning E1,A1,D2,G2]
    [--frets 24]
    [--meter auto|3/4|4/4|6/8]
    [--bpm BPM]
    [--first-downbeat SECONDS]
    [--resume]

bass-transcriber benchmark MANIFEST
bass-transcriber doctor
```

Core backend-independent types:

- `AudioNote`: MIDI pitch, onset/offset seconds, source, optional evidence values and warning flags.
- `BeatMap`: tracked beat/downbeat times, meter, tempo events, confidence diagnostics, and backing-audio padding.
- `ScoreNote`: rational beat position/duration, pitch, string/fret, and tie information.
- `BassScore`: measures, tempo map, instrument specification, notes, and metadata.
- `TranscriptionResult`: raw notes, score, diagnostics, and exported artifact paths.

No `guitarpro.models` objects may escape the GP5 adapter.

### Processing stages

1. Decode once through FFmpeg while preserving timing and stereo content. Do not loudness-normalize the source.
2. Run Beat This on the full mix. Auto-accept stable 3/4 or 4/4; require `--meter`, `--bpm`, or `--first-downbeat` when ambiguous. [Beat This](https://pypi.org/project/beat-this/)
3. Run the benchmark-selected audio branch and MuScriptor configuration. Pair streamed start/end events and retain immutable raw output.
4. Apply conservative cleanup: validate duration/order, merge only demonstrable duplicates, retain playable overlaps, and flag rather than octave-fold notes outside E1-G4.
5. Map seconds to continuous beat positions, then quantize using straight and triplet grids through 32nd-note resolution. Penalize unnecessary rhythmic complexity and large movements from the detected timing.
6. Derive hidden GP5 tempo changes from tracked beats, compressing them while keeping predicted playback within 50 ms of tracked beats.
7. Add enough leading silence to the companion WAV for the first detected downbeat to land on a barline. Shift score timing by the identical amount.
8. Enumerate every pitch-valid string/fret position and use dynamic programming to minimize position movement, large fret/string jumps, high-fret use, and awkward chord stretches. Phrase-length rests reduce transition penalties. The result is ergonomic, not claimed to reproduce the performer's original fingering.
9. Spell complete measures with rests, split notes at barlines or tempo boundaries, and add ties. Export GP5 5.1 using PyGuitarPro with a four-string bass track and fixed default velocity. Preserve full Unicode metadata in JSON; transliterate only metadata that GP5's eight-bit encoding cannot represent. [PyGuitarPro](https://pyguitarpro.readthedocs.io/en/stable/pyguitarpro/quickstart.html)

## Benchmark and Test Plan

- Build ten rights-cleared, manually corrected 30-60-second rock/pop/metal excerpts covering clean and distorted bass, picked/fingerstyle playing, syncopation, sustained notes, and strong kick bleed.
- Use IDMT-SMT-Bass for isolated-bass regression and fingering fixtures. Use Slakh only for engineering smoke tests and oracle-stem diagnosis, not final model ranking because its Lakh-derived material may overlap MuScriptor's training sources. [IDMT-SMT-Bass](https://www.idmt.fraunhofer.de/en/publications/datasets/bass_lines.html), [Slakh2100](https://www.slakh.com/)
- Benchmark full mix, `htdemucs`, and `htdemucs_ft`, each with MuScriptor medium and large. Rank by macro note-with-offset F1 using 50 ms onset tolerance and the standard offset tolerance. Break results within one absolute F1 point by lower runtime, then lower peak VRAM. Pin that winner as the `auto` configuration rather than running an ensemble for every song.
- Report onset-only F1, note-with-offset F1, frame F1, octave-error rate, duration error, quantization displacement, beat/downbeat accuracy, runtime, and peak VRAM.
- Unit-test tuning/range mapping, ambiguous positions such as A2, deterministic fingering, double-stops, rests, triplets, pickups, barline ties, meter changes, tempo compression, audio padding, and failure diagnostics.
- Round-trip every generated GP5 through PyGuitarPro and compare semantic score content.
- Maintain a tiny GP5 compatibility corpus and manually verify it in the installed ToneLib Jam 4.8.7: tuning, notes, rests, ties, tuplets, tempo changes, playback cursor, and zero-offset companion WAV alignment.
- Release acceptance requires valid measures, exact pitch preservation through string/fret assignment, GP5 import in ToneLib, no more than 50 ms synchronization error at tracked beats, reproducible cached runs, and an actionable override command whenever rhythm analysis refuses export.

## Assumptions and Deferred Work

- Input recordings and MuScriptor outputs are used only where the user has the required rights; V1 is noncommercial.
- Manual correction in ToneLib is expected.
- Standard EADG is the only validated tuning, although the internal instrument model remains configurable.
- V1 excludes a GUI, automatic tuning/key detection, original-fingering claims, articulations, calibrated confidence, commercial model support, GPX/GP7, MusicXML delivery, and ToneLib's proprietary project format.
- V2 may add Basic Pitch or CREPE/pYIN evidence, onset refinement, articulation classification, alternate tunings, and newer separators only when controlled ablations improve the end-to-end tab.
