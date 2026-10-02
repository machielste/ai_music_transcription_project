"""Experimental ToneLib Jam 3.1 export using a user-supplied project template.

The archive/XML structure is inferred from saved projects, not a published spec.
Only the simple single-track GP5 scores produced by this application are supported.
"""

from __future__ import annotations

import argparse
import copy
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

import guitarpro  # type: ignore[import-untyped]


def _text(parent: ET.Element, tag: str, value: object) -> None:
    ET.SubElement(parent, tag).text = str(value)


def _audio_entry_name(filename: str) -> str:
    """Match the filename hash used for audio IDs in native ToneLib projects.

    This is JUCE String::hashCode64: multiply by 101 for each Unicode code point,
    wrapping to 64 bits. ToneLib stores its unsigned hexadecimal representation.
    The hash must match the audio/name field, including extension and case.
    """
    value = 0
    for character in filename:
        value = (value * 101 + ord(character)) & ((1 << 64) - 1)
    return f"audio/{value:x}.snd"


def _score_from_gp5(song: Any, template: ET.Element) -> ET.Element:
    """Preserve template settings while rebuilding score and backing timing."""
    if len(song.tracks) != 1:
        raise ValueError("Prototype supports exactly one GP5 track")
    track = song.tracks[0]
    if len(track.strings) not in (4, 5):
        raise ValueError("Prototype supports four- or five-string bass")
    root = copy.deepcopy(template)
    info = root.find("info")
    if info is None:
        raise ValueError("Template has no score information")
    for child in info:
        child.text = "no" if child.tag == "show_remarks" else None
    name = info.find("name")
    if name is None:
        name = ET.SubElement(info, "name")
    name.text = song.title

    index = root.find("BarIndex")
    tracks = root.find("Tracks")
    backing = root.find("Backing_track1")
    if index is None or tracks is None or backing is None:
        raise ValueError("Template must contain BarIndex, Tracks, and Backing_track1")
    old_track = tracks.find("Track")
    if old_track is None:
        raise ValueError("Template has no track settings")
    settings = dict(old_track.attrib)
    settings.update(name=track.name, id="1", mute="1", solo="0")
    index.clear()
    tracks.clear()
    new_track = ET.SubElement(tracks, "Track", settings)
    strings = ET.SubElement(new_track, "Strings")
    for string in track.strings:
        ET.SubElement(strings, "String", id=str(string.number), tuning=str(string.value))
    bars = ET.SubElement(new_track, "Bars")
    for child in list(backing):
        backing.remove(child)
    backing.set("mute", "0")
    backing.set("solo", "0")
    audio = ET.SubElement(backing, "audio")
    markers = ET.SubElement(audio, "bars", num="0", nst="0")

    elapsed = 0.0
    tempo = song.tempo
    marker_count = 0
    for number, measure in enumerate(track.measures, start=1):
        if len(measure.voices) > 1 and measure.voices[1].beats:
            raise ValueError("Prototype does not support a second GP5 voice")
        signature = measure.header.timeSignature
        if signature.numerator != 4 or signature.denominator.value != 4:
            raise ValueError("Prototype supports 4/4 scores only")
        for beat_index, beat in enumerate(measure.voices[0].beats):
            change = beat.effect.mixTableChange
            if change is not None and change.tempo is not None:
                if beat_index != 0:
                    raise ValueError("Prototype supports tempo changes at bar starts only")
                tempo = change.tempo.value
        header = ET.SubElement(index, "Bar", id=str(number), jam_set="0")
        # Explicit tempos on every bar avoid depending on undocumented inheritance.
        header.set("tempo", str(tempo))
        if number == 1:
            ET.SubElement(header, "time_sign", numerator="4", duration="4")
        bar = ET.SubElement(bars, "Bar", id=str(number))
        if number == 1:
            ET.SubElement(bar, "Clef", value="2")
            ET.SubElement(bar, "KeySign", value="0")
        for beat in measure.voices[0].beats:
            duration = beat.duration
            if duration.isDotted or duration.tuplet.enters != duration.tuplet.times:
                raise ValueError("Prototype does not support dotted notes or tuplets")
            xb = ET.SubElement(bar, "Beat", duration=str(duration.value), dyn="mf")
            for note in beat.notes:
                if note.type.name not in ("normal", "tie"):
                    raise ValueError(f"Unsupported note type: {note.type.name}")
                attributes = {"fret": str(note.value), "string": str(note.string)}
                if note.type.name == "tie":
                    attributes["tied"] = "yes"
                ET.SubElement(xb, "Note", attributes)
        ET.SubElement(bar, "Beats")
        for quarter in range(4):
            ET.SubElement(
                markers, f"beat{marker_count}", n=str(quarter),
                t=format(elapsed + quarter * 60.0 / tempo, ".16g"),
            )
            marker_count += 1
        elapsed += 240.0 / tempo
    markers.set("num", str(marker_count))
    return root


def write_tonelib_song(
    gp5: Path, audio: Path, output: Path, *, template: Path | None = None,
) -> Path:
    """Create a new self-contained project without overwriting any existing file."""
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg must be available on PATH")
    if template is None:
        template = Path(__file__).with_name("tonelib_template.song")
    with zipfile.ZipFile(template) as archive:
        version = archive.read("version.info")
        if version.rstrip(b"\0") != b"3.1":
            raise ValueError("Prototype requires a ToneLib 3.1 project template")
        settings = archive.read("plg_set_list.dat")
        root = ET.fromstring(archive.read("the_song.dat").rstrip(b"\0"))
    root = _score_from_gp5(guitarpro.parse(str(gp5)), root)
    with tempfile.TemporaryDirectory(prefix="bass-tonelib-") as directory:
        encoded = Path(directory) / "backing.flac"
        # ToneLib's .snd entries are ordinary FLAC streams. Preserve source rate/channels.
        subprocess.run(
            [ffmpeg, "-nostdin", "-v", "error", "-i", str(audio.resolve()),
             "-map", "0:a:0", "-c:a", "flac", str(encoded)],
            check=True, capture_output=True,
        )
        # ToneLib resolves backing audio by its filename hash, not its content hash.
        audio_entry = _audio_entry_name(audio.name)
        backing = root.find("Backing_track1/audio")
        assert backing is not None
        markers = backing.find("bars")
        assert markers is not None
        backing.remove(markers)
        _text(backing, "name", audio.name)
        _text(backing, "data_file", audio_entry)
        _text(backing, "data_len", encoded.stat().st_size)
        _text(backing, "time_offset", 0)
        _text(backing, "gain", 1)
        _text(backing, "channel_mode", 0)
        backing.append(markers)
        ET.indent(root, space="  ")
        xml = ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\0"
        staged = Path(directory) / "project.song"
        with zipfile.ZipFile(staged, "w") as archive:
            archive.writestr("version.info", version, compress_type=zipfile.ZIP_DEFLATED)
            archive.write(encoded, audio_entry, compress_type=zipfile.ZIP_STORED)
            archive.writestr("plg_set_list.dat", settings, compress_type=zipfile.ZIP_DEFLATED)
            archive.writestr("the_song.dat", xml, compress_type=zipfile.ZIP_DEFLATED)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also prevents overwriting a file created during conversion.
        with output.open("xb") as destination, staged.open("rb") as source:
            shutil.copyfileobj(source, destination)
    return output


def main() -> int:
    """Run the prototype independently of the transcription pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gp5", type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--template", type=Path, help="optional project settings template")
    parser.add_argument("--output", "-o", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = write_tonelib_song(args.gp5, args.audio, args.output, template=args.template)
    except subprocess.CalledProcessError as error:
        parser.exit(1, f"FFmpeg failed: {error.stderr.decode(errors='replace')}\n")
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, KeyError) as error:
        parser.exit(1, f"Export failed: {error}\n")
    print(f"Created {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
