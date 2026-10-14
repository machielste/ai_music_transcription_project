import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import pytest

from bass_transcriber.fingering_experiment import _read_score, generate_examples


def _source(path: Path) -> None:
    # First two fragments are one tied root, followed by a fifth, octave,
    # and a double stop. Keep one full bar to exercise archive preservation.
    xml = b'''<Score><info><name>Example</name></info>
      <BarIndex><Bar id="1" tempo="120"><time_sign numerator="4" duration="4"/>
      </Bar></BarIndex><Tracks><Track mute="1"><Strings>
      <String id="1" tuning="43"/><String id="2" tuning="38"/>
      <String id="3" tuning="33"/><String id="4" tuning="28"/>
      </Strings><Bars><Bar id="1">
      <Beat duration="8"><Note string="3" fret="1"/></Beat>
      <Beat duration="8"><Note string="3" fret="0" tied="yes"/></Beat>
      <Beat duration="4"><Note string="2" fret="3"/></Beat>
      <Beat duration="4"><Note string="1" fret="3"/></Beat>
      <Beat duration="4"><Note string="3" fret="1"/><Note string="2" fret="3"/></Beat>
      </Bar></Bars></Track></Tracks><Backing_track1 mute="0"><audio>
      <data_file>audio/example.snd</data_file></audio></Backing_track1></Score>'''
    with ZipFile(path, "w") as archive:
        archive.writestr("the_song.dat", xml + b"\0")
        archive.writestr("audio/example.snd", b"unchanged audio bytes")
        archive.writestr("version.info", b"3.1\0")
        archive.writestr("plg_set_list.dat", b"unchanged settings")


def test_examples_preserve_ties_chords_timing_and_backing(tmp_path: Path) -> None:
    source = tmp_path / "source.song"
    _source(source)
    before = source.read_bytes()
    output = tmp_path / "examples"

    summary = generate_examples(source, output)

    assert source.read_bytes() == before
    assert summary["notes"] == 5
    files = sorted(output.glob("*.song"))
    assert len(files) == 5
    for path in files:
        with ZipFile(path) as archive:
            assert archive.read("audio/example.snd") == b"unchanged audio bytes"
            root = ET.fromstring(archive.read("the_song.dat").rstrip(b"\0"))
        notes, _, _ = _read_score(root)
        assert notes[0].event.start_slot == 0
        assert notes[0].event.end_slot == 8
        assert notes[0].event.pitch == 34
        assert len(notes[0].parts) == 2
        assert notes[-1].string != notes[-2].string
    with pytest.raises(FileExistsError):
        generate_examples(source, output)


def test_experiment_rejects_broken_ties(tmp_path: Path) -> None:
    source = tmp_path / "source.song"
    _source(source)
    with ZipFile(source) as archive:
        root = ET.fromstring(archive.read("the_song.dat").rstrip(b"\0"))
    root.findall("Tracks/Track/Bars/Bar/Beat/Note")[1].set("fret", "2")
    with pytest.raises(ValueError, match="tie does not continue"):
        _read_score(root)


def test_shape_suite_keeps_an_exact_medium_reference(tmp_path: Path) -> None:
    source = tmp_path / "source.song"
    _source(source)
    output = tmp_path / "shape-examples"

    summary = generate_examples(source, output, suite="shapes")

    assert summary["suite"] == "shapes"
    assert len(list(output.glob("*.song"))) == 5
    assert (output / "00-previous-medium.song").exists()
    assert (output / "02-shape-only.song").exists()
    assert "Second experiment" in (output / "README.md").read_text(encoding="utf-8")
