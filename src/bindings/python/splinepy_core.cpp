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

#include <exception>

#include "nb_multipatch.hpp"
#include "nb_spline.hpp"
#include "nb_spline_extensions.hpp"
#include "splinepy/core/common.hpp"

namespace nb = nanobind;

namespace splinepy::py {
void init_nb_hierarchy(nb::module_&);
}

namespace {

/// Exposes core::Status and translates every core::SplinepyException into
/// splinepy.SplinepyError. The type derives from ValueError and RuntimeError,
/// so code catching either builtin keeps working; its status attribute holds
/// the core::Status of the error.
void init_errors(nb::module_& m) {
  using splinepy::core::Status;
  nb::enum_<Status>(m, "Status")
      .value("OK", Status::kOk)
      .value("INVALID_ARGUMENT", Status::kInvalidArgument)
      .value("DIMENSION_MISMATCH", Status::kDimensionMismatch)
      .value("OUT_OF_BOUNDS", Status::kOutOfBounds)
      .value("NOT_CONVERGED", Status::kNotConverged)
      .value("DEGENERATE", Status::kDegenerate);

  nb::object bases = nb::make_tuple(nb::handle(PyExc_ValueError),
                                    nb::handle(PyExc_RuntimeError));
  nb::object error = nb::steal(PyErr_NewExceptionWithDoc(
      "splinepy.splinepy_core.SplinepyError",
      "Error raised by the splinepy core. Subclasses ValueError and "
      "RuntimeError; `status` holds the splinepy_core.Status of the error.",
      bases.ptr(),
      nullptr));
  if (!error.is_valid()) {
    throw nb::python_error();
  }
  m.attr("SplinepyError") = error;

  // The module keeps the type alive for the lifetime of the interpreter.
  nb::register_exception_translator(
      [](const std::exception_ptr& p, void* payload) {
        try {
          std::rethrow_exception(p);
        } catch (const splinepy::core::SplinepyException& e) {
          nb::handle type(static_cast<PyObject*>(payload));
          nb::object instance = type(nb::str(e.what()));
          instance.attr("status") = nb::cast(e.status());
          PyErr_SetObject(type.ptr(), instance.ptr());
        }
      },
      error.ptr());
}

} // namespace

NB_MODULE(splinepy_core, m) {
  init_errors(m);
  splinepy::py::init_nb_spline(m);
  splinepy::py::init_nb_multipatch(m);
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
