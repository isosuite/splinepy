from argparse import Namespace

import splinepy
from splinepy import cli


def test_convert_handles_what_each_reader_returns(tmp_path):
    box = splinepy.helpme.create.box(1, 1).nurbs
    names = ("a.json", "b.xml", "c.mesh")
    single, gismo, mesh = (str(tmp_path / name) for name in names)
    splinepy.io.export(single, [box])
    cli.convert(Namespace(input_file=single, output_file=mesh))
    assert splinepy.io.load(mesh).whatami == box.whatami
    cli.convert(Namespace(input_file=single, output_file=gismo))
    cli.convert(Namespace(input_file=gismo, output_file=single))
    assert len(splinepy.io.load(single)) == 1


def test_entry_prints_the_version(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["splinepy", "--version"])
    cli.entry()
    assert f"splinepy version: {splinepy.__version__}" in capsys.readouterr().out


def test_entry_show_writes_a_viewer_file(monkeypatch, tmp_path):
    source, page = str(tmp_path / "box.json"), tmp_path / "box.html"
    splinepy.io.export(source, [splinepy.helpme.create.box(1, 1).bspline])
    monkeypatch.setattr(
        "sys.argv", ["splinepy", "show", "-i", source, "-o", str(page)]
    )
    cli.entry()
    assert "<html" in page.read_text()[:200].lower()
