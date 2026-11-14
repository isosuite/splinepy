#pragma once

#include <nanobind/nanobind.h>

namespace splinepy::py {

namespace nb = nanobind;

void init_nb_knot_insertion_matrix(nb::module_& m);

} // namespace splinepy::py
