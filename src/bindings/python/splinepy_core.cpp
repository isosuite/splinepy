/*
MIT License

Copyright (c) 2021 Jaewook Lee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <nanobind/nanobind.h>

#include "nb_knot_insertion_matrix.hpp"
#include "nb_multipatch.hpp"
#include "nb_spline.hpp"
#include "nb_spline_extensions.hpp"

namespace nb = nanobind;

namespace splinepy::py {
void init_nb_hierarchy(nb::module_&);
}

NB_MODULE(splinepy_core, m) {
  splinepy::py::init_nb_spline(m);
  splinepy::py::init_nb_multipatch(m);
  splinepy::py::init_nb_knot_insertion_matrix(m);
  splinepy::py::init_nb_spline_extensions(m);
  splinepy::py::init_nb_hierarchy(m);

  m.def("build_type", []() {
#ifndef NDEBUG
    return "debug";
#else
    return "release";
#endif
  });
}
