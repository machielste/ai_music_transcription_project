"""Make controlled fingering examples from an existing generated ToneLib project.

This deliberately supports our simple 4/4 exports only. It changes placements
and the displayed title, preserving beat segmentation, ties, tempos, settings,
and embedded backing audio. No transcription or audio encoding is repeated.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, replace
from itertools import groupby
from pathlib import Path
from typing import TypedDict
from zipfile import ZipFile

from bass_transcriber.tab.fingering import (
    FINGERING_PROFILES,
    FingeringEvent,
    FingeringProfile,
    optimize_fingering,
)
from bass_transcriber.tab.timeline import ResolvedTabNote, _assign_distinct_chord_strings


@dataclass
class _Note:
    event: FingeringEvent
    string: int
    fret: int
    parts: list[int]


class _Example(TypedDict):
    label: str
    file: str
    settings: dict[str, float | int] | None
    changed_notes: int
    g1_attacks: int
    focus_g1_attacks: int
    focus: list[dict[str, int]]
    large_fretted_jumps: int
    large_jumps_without_rest: int
    compact_root_fifth_octaves: int
    split_root_fifth_octaves: int
    review_bars: dict[str, list[str]]


def _read_score(root: ET.Element) -> tuple[list[_Note], tuple[tuple[int, int], ...], float]:
    tracks = root.findall("Tracks/Track")
    signatures = root.findall("BarIndex/Bar/time_sign")
    if len(tracks) != 1 or not signatures or any(
        (s.get("numerator"), s.get("duration")) != ("4", "4") for s in signatures
    ):
        raise ValueError("experiment requires a single generated 4/4 bass track")
    track = tracks[0]
    strings = tuple(
        (int(s.attrib["id"]), int(s.attrib["tuning"]))
        for s in track.findall("Strings/String")
    )
    tuning = dict(strings)
    notes: list[_Note] = []
    last_by_string: dict[int, _Note] = {}
    part_index = 0
    slot = 0
    for bar in track.findall("Bars/Bar"):
        bar_start = slot
        for beat in bar.findall("Beat"):
            duration = int(beat.attrib["duration"])
            if duration not in (1, 2, 4, 8, 16, 32):
                raise ValueError(f"unsupported beat duration {duration}")
            length = 32 // duration
            for element in beat.findall("Note"):
                string, fret = int(element.attrib["string"]), int(element.attrib["fret"])
                pitch = tuning[string] + fret
                if element.get("tied") == "yes":
                    previous = last_by_string.get(string)
                    if previous is None or previous.event.end_slot != slot or (
                        fret != 0 and previous.event.pitch != pitch
                    ):
                        raise ValueError("tie does not continue the preceding note")
                    # Some generated GP5 tie fragments carry value zero, which
                    # the ToneLib exporter retains. Their pitch is inherited
                    # from the preceding attack, rather than a fresh open note.
                    previous.event = replace(previous.event, end_slot=slot + length)
                    previous.parts.append(part_index)
                else:
                    note = _Note(FingeringEvent(pitch, slot, slot + length), string, fret,
                                 [part_index])
                    notes.append(note)
                    last_by_string[string] = note
                part_index += 1
            slot += length
        if slot - bar_start != 32:
            raise ValueError("experiment requires complete generated 4/4 bars")
    # Tempo dithering represents one underlying tempo. Use its mean duration
    # for ergonomic timing, while preserving every original tempo in the output.
    headers = root.findall("BarIndex/Bar")
    seconds_per_slot = sum(240 / int(b.attrib["tempo"]) for b in headers) / (32 * len(headers))
    return notes, strings, seconds_per_slot


def _resolve(
    notes: list[_Note], strings: tuple[tuple[int, int], ...],
    seconds_per_slot: float, profile: FingeringProfile,
) -> list[ResolvedTabNote]:
    groups = [list(g) for _, g in groupby(enumerate(notes), lambda item: item[1].event.start_slot)]
    backbone = [min(g, key=lambda item: item[1].event.pitch)[0] for g in groups]
    choices = optimize_fingering(
        [notes[i].event for i in backbone], "avoid_open", strings=strings,
        seconds_per_slot=seconds_per_slot, profile_override=profile,
    )
    by_index = dict(zip(backbone, choices, strict=True))
    resolved = [
        ResolvedTabNote(
            n.event.pitch, n.event.start_slot, n.event.end_slot,
            by_index[i].string if i in by_index else n.string,
            by_index[i].fret if i in by_index else n.fret,
        ) for i, n in enumerate(notes)
    ]
    chord_profile = profile if (5, 23) in strings else replace(profile, upper_b_string_penalty=0)
    return _assign_distinct_chord_strings(resolved, strings, chord_profile)


def _invariant_score(root: ET.Element) -> bytes:
    root = copy.deepcopy(root)
    name = root.find("info/name")
    if name is not None:
        name.text = ""
    for note in root.findall("Tracks/Track/Bars/Bar/Beat/Note"):
        note.attrib.pop("string")
        note.attrib.pop("fret")
    return ET.tostring(root)


def _write_variant(source: Path, output: Path, root: ET.Element) -> None:
    with (
        ZipFile(source) as original,
        output.open("xb") as destination,
        ZipFile(destination, "w") as archive,
    ):
        for info in original.infolist():
            if info.filename == "the_song.dat":
                archive.writestr(
                    info, ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\0",
                )
            else:
                with original.open(info) as reader, archive.open(info, "w") as writer:
                    shutil.copyfileobj(reader, writer)


def generate_examples(
    source: Path, output: Path, *, suite: str = "position",
) -> dict[str, object]:
    """Create five examples in a new directory, never overwrite existing files."""
    with ZipFile(source) as archive:
        root = ET.fromstring(archive.read("the_song.dat").rstrip(b"\0"))
        original_entries = {i.filename: (i.CRC, i.file_size) for i in archive.infolist()}
    notes, strings, seconds_per_slot = _read_score(root)
    base = FINGERING_PROFILES["avoid_open"]
    variants = [
        ("00-original", None),
        ("01-gentle", replace(base, hand_position_weight=0.15)),
        ("02-medium", replace(base, hand_position_weight=0.40)),
        ("03-strong", replace(base, hand_position_weight=0.80)),
        ("04-medium-easier-shifts", replace(
            base, hand_position_weight=0.40, position_shift_weight=base.position_shift_weight / 2,
        )),
    ]
    if suite == "shapes":
        medium = replace(base, hand_position_weight=0.40)
        variants = [
            ("00-previous-medium", medium),
            ("01-jump-aware", replace(medium, same_string_jump_weight=0.50)),
            ("02-shape-only", replace(medium, root_fifth_octave_reward=4.0)),
            ("03-compact-shapes", replace(
                medium, same_string_jump_weight=0.50, root_fifth_octave_reward=4.0,
            )),
            ("04-gentle-compact-shapes", replace(
                base, hand_position_weight=0.15, same_string_jump_weight=0.50,
                root_fifth_octave_reward=4.0,
            )),
        ]
    elif suite != "position":
        raise ValueError(f"unknown experiment suite: {suite}")
    original_score = _invariant_score(root)
    original_events = [asdict(n.event) for n in notes]
    # Reconstructing from an export can expose different tie/chord handling.
    # Report control agreement explicitly rather than attributing that to gravity.
    control = _resolve(notes, strings, seconds_per_slot, base)
    control_changes = sum((a.string, a.fret) != (b.string, b.fret)
                          for a, b in zip(notes, control, strict=True))
    output.mkdir(parents=True, exist_ok=False)
    results: list[_Example] = []
    for label, profile in variants:
        variant_root = copy.deepcopy(root)
        if profile is None:
            resolved = [ResolvedTabNote(n.event.pitch, n.event.start_slot, n.event.end_slot,
                                       n.string, n.fret) for n in notes]
        else:
            resolved = _resolve(notes, strings, seconds_per_slot, profile)
            elements = variant_root.findall("Tracks/Track/Bars/Bar/Beat/Note")
            for original, choice in zip(notes, resolved, strict=True):
                for part in original.parts:
                    elements[part].set("string", str(choice.string))
                    elements[part].set("fret", str(choice.fret))
            name = variant_root.find("info/name")
            if name is not None:
                name.text = f"{name.text} [{label}]"
        target = output / f"{label}.song"
        _write_variant(source, target, variant_root)
        with ZipFile(target) as archive:
            checked_root = ET.fromstring(archive.read("the_song.dat").rstrip(b"\0"))
            assert archive.testzip() is None
            assert _invariant_score(checked_root) == original_score
            assert {i.filename: (i.CRC, i.file_size) for i in archive.infolist()
                    if i.filename != "the_song.dat"} == {
                        name: entry for name, entry in original_entries.items()
                        if name != "the_song.dat"
                    }
        checked_notes, checked_strings, _ = _read_score(checked_root)
        assert checked_strings == strings
        assert [asdict(n.event) for n in checked_notes] == original_events
        assert [(n.string, n.fret) for n in checked_notes] == [
            (n.string, n.fret) for n in resolved
        ]
        focus = [n for n in resolved if 10 * 32 <= n.start_slot < 15 * 32]
        jumps = [(a, b) for a, b in zip(resolved, resolved[1:], strict=False)
                 if a.fret > 0 and b.fret > 0 and a.string == b.string
                 and abs(a.fret - b.fret) >= 5]
        compact = split = 0
        for a, b, c in zip(resolved, resolved[1:], resolved[2:], strict=False):
            if (b.pitch - a.pitch, c.pitch - b.pitch) != (7, 5):
                continue
            compact += int(a.fret > 0 and b.string == a.string - 1
                           and c.string == a.string - 2 and b.fret == c.fret == a.fret + 2)
            split += int(a.string == b.string and abs(a.fret - b.fret) >= 5)
        result: _Example = {
            "label": label, "file": target.name,
            "settings": asdict(profile) if profile else None,
            "changed_notes": sum((a.string, a.fret) != (b.string, b.fret)
                                 for a, b in zip(notes, resolved, strict=True)),
            "g1_attacks": sum(n.string == 1 and n.fret == 1 for n in resolved),
            "focus_g1_attacks": sum(n.string == 1 and n.fret == 1 for n in focus),
            "focus": [asdict(n) for n in focus],
            "large_fretted_jumps": len(jumps),
            "large_jumps_without_rest": sum(b.start_slot == a.end_slot for a, b in jumps),
            "compact_root_fifth_octaves": compact,
            "split_root_fifth_octaves": split,
            "review_bars": {
                str(bar): [f"{'GDAEB'[n.string - 1]}{n.fret}" for n in resolved
                           if n.start_slot // 32 + 1 == bar]
                for bar in (11, 13, 23, 50)
            },
        }
        results.append(result)
        print(f"{label}: {result['changed_notes']} changed notes, "
              f"G1 attacks {result['g1_attacks']} (bars 11-15: {result['focus_g1_attacks']})",
              flush=True)
        print(f"  large fretted jumps: {len(jumps)}, compact triples: {compact}, "
              f"split triples: {split}", flush=True)
    summary = {
        "source": str(source.resolve()), "base_profile": "avoid_open",
        "hand_position_range": [5, 9], "control_changed_notes": control_changes,
        "notes": len(notes), "seconds_per_slot": seconds_per_slot, "variants": results,
        "suite": suite,
    }
    (output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    headers = root.findall("BarIndex/Bar")
    focus_start = sum(240 / int(b.attrib["tempo"]) for b in headers[:10])
    focus_end = sum(240 / int(b.attrib["tempo"]) for b in headers[:15])
    lines = [
        f"# {root.findtext('info/name', source.stem)} — fingering experiment", "",
        f"Open each .song file in ToneLib. Start with bars 11–15 "
        f"(about {focus_start:.0f}–{focus_end:.0f} seconds).",
        "Each file contains the full song and the original embedded backing audio.", "",
        "All optimized variants use Avoid open strings, preferring index-finger positions 5–9;",
        "played notes can extend above that zone. Cost is weight × distance outside it,",
        "once per note attack, including open-string attacks.", "",
        "| File | Attraction weight | Base shift weight | G1 attacks, bars 11–15 |",
        "| --- | ---: | --- | ---: |",
    ]
    for result in results:
        settings = result["settings"]
        weight = settings["hand_position_weight"] if settings else 0
        shifts = settings["position_shift_weight"] if settings else base.position_shift_weight
        lines.append(f"| [{result['label']}]({result['file']}) | {weight} | {shifts} | "
                     f"{result['focus_g1_attacks']} |")
    lines += [
        "", f"Control rerun: {control_changes} placements differ from the saved original.",
        "Every output was checked for identical pitches, onsets, durations, ties, beat",
        "segmentation, tempo changes, track settings, and embedded audio contents.", "",
        "Compare comfort of the whole riff, the return to low notes, and string crossings.",
        "Fewer G1 notes alone does not establish that a version is easier to play.", "",
        "Full settings and note placements for bars 11–15 are in comparison.json.",
    ]
    if suite == "shapes":
        lines += [
            "", "## Second experiment: jumps and complete shapes", "",
            "The reference is the previous medium setting, regenerated from the original.",
            "Jump cost = weight × max(0, fret distance − 3)² × urgency / rest discount,",
            "only when both notes are fretted on the same string. Urgency uses a",
            "0.12-second articulation window plus actual rest, capped by onset spacing.",
            "This is an experimental heuristic; open strings and slides are not modeled",
            "as fretted jumps. A prescribed slide could legitimately use a penalized move.",
            "The shape reward scores consecutive root–fifth–octave triples, ascending or",
            "descending, on three adjacent strings at frets f, f+2, f+2. It does not yet",
            "recognize shapes interrupted by repeated notes or fills. Gaps over one beat",
            "disable the reward. It uses an exact search retaining the previous two states.",
            "No new penalty for using D/G in low positions is included.", "",
            "| File | Jump weight | Shape reward | Large jumps | No-rest jumps | "
            "G1 attacks | Compact triples | Split triples |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for example in results:
            settings = example["settings"]
            assert settings is not None
            lines.append(
                f"| [{example['label']}]({example['file']}) | "
                f"{settings['same_string_jump_weight']} | {settings['root_fifth_octave_reward']} | "
                f"{example['large_fretted_jumps']} | {example['large_jumps_without_rest']} | "
                f"{example['g1_attacks']} | {example['compact_root_fifth_octaves']} | "
                f"{example['split_root_fifth_octaves']} |",
            )
        lines += ["", "Large jumps count fretted same-string moves of five or more frets.",
                  "Compact/split triple counts describe consecutive ascending figures.",
                  "Also compare bars 23 (about 0:57) and 50 (about 2:09).", ""]
        for example in results:
            lines += [f"### {example['label']}", ""]
            for bar, placements in example["review_bars"].items():
                lines.append(f"- Bar {bar}: {' → '.join(placements)}")
            lines.append("")
    else:
        lines += ["", "The easier-shifts example halves the base position-shift weight;",
                  "the penalty for very rapid shifts remains unchanged."]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="existing generated ToneLib .song")
    parser.add_argument("output", type=Path, help="new directory for examples")
    parser.add_argument("--suite", choices=("position", "shapes"), default="position")
    args = parser.parse_args()
    generate_examples(args.source, args.output, suite=args.suite)


if __name__ == "__main__":
    main()
