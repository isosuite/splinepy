import numpy as np

import splinepy

if __name__ == "__main__":
    line = splinepy.helpme.create.line(
        np.array([[0, 0, 0], [2, 5, 0], [4, 4, 2]])
    )
    rect = splinepy.helpme.create.box(5, 3)
    box = splinepy.helpme.create.box(3, 2, 4)
    pyramid = splinepy.helpme.create.pyramid(1, 1, 2)

    splinepy.show(
        ["Line", line],
        ["Rectangle", rect],
        ["Box", box],
        ["Pyramid", pyramid],
        title="Rectangular objects",
        resolution=50,
    )

    circ = splinepy.helpme.create.circle(5)
    disk1 = splinepy.helpme.create.disk(3, angle=100, n_knot_spans=4)
    disk2 = splinepy.helpme.create.disk(
        5, inner_radius=1.0, angle=360, n_knot_spans=10
    )

    splinepy.show(["Line Circle", circ], ["Disk section", disk1], ["Disk", disk2])

    cone = splinepy.helpme.create.cone(5, 10, angle=180)
    splinepy.show(["Cone", cone])

    torus = splinepy.helpme.create.torus(10, 2)
    torus2 = splinepy.helpme.create.torus(
        5, 2, section_inner_radius=0.5, torus_angle=100, section_angle=210
    )

    splinepy.show(["Torus", torus], ["Torus section", torus2], resolution=50)

    sphere = splinepy.helpme.create.sphere(3)
    splinepy.show(["Sphere", sphere], resolution=50)
