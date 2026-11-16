API References
==============

Overview
--------

- The standalone C++ core implements spline evaluation, manipulation, Bézier
  algebra, and proximity queries. The :mod:`splinepy.splinepy_core` module
  exposes native operations to Python through nanobind.
- :class:`splinepy.spline.Spline` provides the common tensor-product API.
  Use :class:`splinepy.bezier.Bezier`,
  :class:`splinepy.rational_bezier.RationalBezier`,
  :class:`splinepy.bspline.BSpline`, or :class:`splinepy.nurbs.NURBS`
  to construct a spline. NumPy arrays provide access to spline data.
- :class:`splinepy.hierarchy.HBSpline` and :class:`splinepy.hierarchy.THBSpline`
  support local refinement. Both have rational variants. THB splines also
  support coarsening.
- :class:`splinepy.multipatch.Multipatch` groups patches and manages
  interfaces, boundaries, and fields.
- :mod:`splinepy.helpme` provides tools for creating, fitting, and processing
  splines. :mod:`splinepy.io` handles file import and export.

Python API
----------
.. toctree::

   python_api

C++ API
-------
.. raw:: html

   <a href='doxy/html/index.html'>Full documentation (doxygen)</a>
