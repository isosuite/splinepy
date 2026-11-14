find_package(
  Python 3.8
  COMPONENTS Interpreter Development.Module
  REQUIRED)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(splinepy_core
  splinepy_core.cpp
  nb_spline.cpp
  nb_multipatch.cpp
  nb_knot_insertion_matrix.cpp
  nb_spline_extensions.cpp
  nb_hierarchy.cpp
)

target_link_libraries(splinepy_core PRIVATE splinepy::core)

install(
  TARGETS splinepy_core
  DESTINATION splinepy
  COMPONENT PythonModule)
