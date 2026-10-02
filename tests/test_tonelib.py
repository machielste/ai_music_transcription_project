import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import guitarpro
import numpy as np
import pytest
import soundfile as sf

from bass_transcriber.export.gp5 import write_gp5
from bass_transcriber.export.tonelib import _score_from_gp5, write_tonelib_song
from bass_transcriber.models import BassNote, RhythmGrid


def _template() -> ET.Element:
    return ET.fromstring(
        '<Score><info><name>Old song</name><artist>Old artist</artist></info>'
        '<BarIndex/><Tracks><Track name="Old bass" offset="24"/></Tracks>'
        '<Backing_track1 mute="1"><audio><name>old.mp3</name></audio></Backing_track1>'
        '</Score>'
    )


def _gp5(path: Path) -> None:
    grid = RhythmGrid(
        detector="test", bpm=117.454, beats_per_bar=4,
        first_downbeat_seconds=0.0, beat_times_seconds=(0.0, 0.5),
        onset_delay_seconds=0.0,
    )
    write_gp5(
        path, [BassNote(25, 0.0, 0.7, "electric_bass"),
               BassNote(28, 8.0, 9.0, "electric_bass")],
        grid, title="Test bass", five_string=True,
    )


def test_score_preserves_gp5_notes_ties_durations_and_tempo(tmp_path: Path) -> None:
    gp = tmp_path / "test.gp5"
    _gp5(gp)
    song = guitarpro.parse(str(gp))
    template = _template()
    root = _score_from_gp5(song, template)
    assert template.findtext("info/name") == "Old song"
    assert root.findtext("info/name") == song.title
    assert root.find("Tracks/Track").get("mute") == "1"
    assert root.find("Backing_track1").get("mute") == "0"
    assert [int(s.get("tuning")) for s in root.findall("Tracks/Track/Strings/String")] == [
        43, 38, 33, 28, 23,
    ]
    tempo = song.tempo
    elapsed = 0.0
    markers = root.findall("Backing_track1/audio/bars/*")
    for index, (bar, measure) in enumerate(
        zip(root.findall("Tracks/Track/Bars/Bar"), song.tracks[0].measures, strict=True)
    ):
        for xb, beat in zip(bar.findall("Beat"), measure.voices[0].beats, strict=True):
            assert int(xb.get("duration")) == beat.duration.value
            assert [(int(n.get("string")), int(n.get("fret")), n.get("tied") == "yes")
                    for n in xb.findall("Note")] == [
                (n.string, n.value, n.type.name == "tie") for n in beat.notes
            ]
            change = beat.effect.mixTableChange
            if change and change.tempo:
                tempo = change.tempo.value
        assert float(markers[index * 4].get("t")) == pytest.approx(elapsed)
        elapsed += 240 / tempo
    assert len(markers) == len(song.tracks[0].measures) * 4


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg required")
def test_archive_embeds_audio_and_refuses_overwrite(tmp_path: Path) -> None:
    gp = tmp_path / "test.gp5"
    _gp5(gp)
    source = tmp_path / "backing.wav"
    samples = np.zeros((4410, 2))
    sf.write(source, samples, 44100)
    template = tmp_path / "template.song"
    with zipfile.ZipFile(template, "w") as archive:
        archive.writestr("version.info", b"3.1\0")
        archive.writestr("the_song.dat", ET.tostring(_template()) + b"\0")
        archive.writestr("plg_set_list.dat", b"<plg_set_list/>\0")
    original = template.read_bytes()
    output = tmp_path / "new.song"
    write_tonelib_song(gp, source, output, template=template)
    assert template.read_bytes() == original
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        root = ET.fromstring(archive.read("the_song.dat").rstrip(b"\0"))
        assert root.findtext("Backing_track1/audio/name") == source.name
        audio_entry = root.findtext("Backing_track1/audio/data_file")
        assert re.fullmatch(r"audio/[0-9a-f]{16}\.snd", audio_entry)
        payload = archive.read(audio_entry)
        assert payload.startswith(b"fLaC")
        assert len(payload) == int(root.findtext("Backing_track1/audio/data_len"))
        import io

        decoded, rate = sf.read(io.BytesIO(payload))
        assert rate == 44100
        assert decoded.shape == samples.shape
        assert np.array_equal(decoded, samples)
    with pytest.raises(FileExistsError):
        write_tonelib_song(gp, source, output, template=template)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg required")
@pytest.mark.parametrize("draft_first", [False, True])
def test_desktop_pipeline_creates_and_refreshes_companion_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, draft_first: bool,
) -> None:
    import bass_transcriber.pipeline as pipeline

    source = tmp_path / "music.wav"
    sf.write(source, np.zeros((44100, 2)), 44100)
    grid = RhythmGrid(
        detector="test", bpm=120.0, beats_per_bar=4, first_downbeat_seconds=0.0,
        beat_times_seconds=(0.0, 0.5), onset_delay_seconds=0.0,
    )
    monkeypatch.setattr(pipeline, "transcribe_bass", lambda *a, **k: [
        BassNote(28, 0.0, 0.25, "electric_bass"),
    ])
    monkeypatch.setattr(pipeline, "detect_rhythm", lambda *a, **k: grid)
    result = pipeline.process_song(
        source, tmp_path / "run", copy_source=True,
        generate_tonelib=True, generate_gp5=not draft_first,
    )
    assert result.tonelib_song == result.output.with_suffix(".song")
    assert result.copied_source.read_bytes() == source.read_bytes()
    if draft_first:
        assert not result.tonelib_song.exists()
        pipeline.export_gp5_from_fingering_draft(result)
    assert result.output.is_file()
    assert result.fingering_draft.is_file()
    assert result.raw_notes.is_file()
    with zipfile.ZipFile(result.tonelib_song) as archive:
        score = ET.fromstring(archive.read("the_song.dat").rstrip(b"\0"))
        assert score.findtext("Backing_track1/audio/name") == "music.wav"
        assert score.find("Tracks/Track").get("mute") == "1"
        assert archive.testzip() is None
    # Re-exporting an edited draft replaces its companion, rather than refusing
    # to export because the first .song already exists or leaving stale content.
    result.tonelib_song.write_bytes(b"stale project")
    pipeline.export_gp5_from_fingering_draft(result)
    assert zipfile.is_zipfile(result.tonelib_song)
