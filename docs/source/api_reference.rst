API References
==============

Overview
--------
- :class:`splinepy.splinepy_core` is a module that includes all the c++ bindings. It provides the kernels (queries, manipulations, Bezier algebra) that work on spline arrays. It only contains minimal documentation, as the functions are integrated throughout the library with proper documentation on python side.
- :class:`.Spline` is a plain python class that stores degrees, knot vectors, control points and weights as numpy arrays. Common features for all splines are implemented here. We recommend using :class:`.Bezier`, :class:`.RationalBezier`, :class:`.BSpline`, or :class:`.NURBS` in front-end applications.
- :class:`.BezierBase` extends :class:`.Spline` with features available for Bezier-families. It becomes a basis for :class:`.Bezier` and :class:`.RationalBezier`, where the derived classes implement appropriate initializations.
- :class:`.BSplineBase` extends :class:`.Spline` with features available for BSpline-famlilies. It becomes a basis for :class:`.BSpline` and :class:`.NURBS`, where derived classes implements appropriate initializations.
- :class:`.Multipatch` is a plain python class that holds a list of splines together with their topology (interfaces, boundaries, fields).
- :class:`splinepy.helpme` module contains useful application-driven tools / utility functions.
- :class:`splinepy.io` module provides all methods concerning exports to other mesh/file formats, but also exporting vector graphics using `svg`


Python API
----------
.. toctree::

   python_api

C++ API
-------
.. raw:: html

   <a href='doxy/html/index.html'>Full documentation (doxygen)</a>
