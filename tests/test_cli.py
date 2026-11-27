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
