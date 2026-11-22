# Core library target (zero Python dependencies)
add_library(
  splinepy_core_lib STATIC
  splinepy/core/concurrency/thread_pool.cpp
  splinepy/core/validate/validator.cpp
  splinepy/core/hierarchy/key.cpp
  splinepy/core/hierarchy/hierarchy.cpp
  splinepy/core/hierarchy/queries.cpp
  splinepy/core/hierarchy/refine.cpp
  splinepy/core/hierarchy/coarsen.cpp
  splinepy/core/hierarchy/export.cpp
  splinepy/core/hierarchy/proximity.cpp
  splinepy/core/math/binomial.cpp
  splinepy/core/math/bernstein.cpp
  splinepy/core/math/bspline_basis.cpp
  splinepy/core/queries/evaluate.cpp
  splinepy/core/queries/derivative.cpp
  splinepy/core/queries/jacobian.cpp
  splinepy/core/queries/basis.cpp
  splinepy/core/manipulation/axis_map.cpp
  splinepy/core/manipulation/insert_knots.cpp
  splinepy/core/manipulation/remove_knots.cpp
  splinepy/core/manipulation/elevate_degree.cpp
  splinepy/core/manipulation/greville.cpp
  splinepy/core/manipulation/extract.cpp
  splinepy/core/bezier/split.cpp
  splinepy/core/bezier/add.cpp
  splinepy/core/bezier/multiply.cpp
  splinepy/core/bezier/derivative_spline.cpp
  splinepy/core/bezier/compose.cpp
  splinepy/core/microstructure/plan.cpp
  splinepy/core/multipatch/interfaces.cpp
  splinepy/core/multipatch/orientations.cpp
  splinepy/core/multipatch/boundary.cpp
  splinepy/core/multipatch/continuity.cpp
  splinepy/core/spatial/element_bvh.cpp
  splinepy/proximity/proximity.cpp
  splinepy/proximity/search.cpp
  splinepy/proximity/starts.cpp
  splinepy/proximity/dogleg.cpp
  splinepy/proximity/slsqp_solver.cpp
  splinepy/proximity/newton_lm.cpp
  splinepy/proximity/grid_guess.cpp
  splinepy/proximity/slsqp.c)

target_include_directories(
  splinepy_core_lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                           $<INSTALL_INTERFACE:include>)

target_compile_features(splinepy_core_lib PUBLIC cxx_std_17)
set_target_properties(splinepy_core_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
  target_compile_options(splinepy_core_lib PRIVATE /W4)
else()
  target_compile_options(splinepy_core_lib PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Modern alias
add_library(splinepy::core ALIAS splinepy_core_lib)

if(DEFINED TARGETS_EXPORT_NAME)
  install(
    TARGETS splinepy_core_lib
    EXPORT "${TARGETS_EXPORT_NAME}"
    LIBRARY DESTINATION ${lib_dest}
    ARCHIVE DESTINATION ${lib_dest}
    INCLUDES
    DESTINATION "${incl_dest}")
endif()

# Tests
if(SPLINEPY_CORE_BUILD_TESTS)
  add_subdirectory(tests)
endif()

# Benchmarks
option(SPLINEPY_CORE_BUILD_BENCHMARKS "Build splinepy core benchmarks" ON)
if(SPLINEPY_CORE_BUILD_BENCHMARKS)
  foreach(name queries compose)
    add_executable(${name}_benchmarks benchmarks/benchmark_${name}.cpp)
    target_link_libraries(${name}_benchmarks PRIVATE splinepy::core)
    if(MSVC)
      target_compile_options(${name}_benchmarks PRIVATE /W4)
    else()
      target_compile_options(${name}_benchmarks PRIVATE -Wall -Wextra)
    endif()
  endforeach()
endif()

# python?
if(SPLINEPY_COMPILE_PYTHON)
  add_subdirectory(bindings/python)
endif()

# cmake config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${version_config}"
                                 COMPATIBILITY SameMajorVersion)
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/config.cmake.in" "${project_config}"
  INSTALL_DESTINATION "${cfg_dest}")

# config files
install(FILES "${project_config}" "${version_config}" DESTINATION "${cfg_dest}")
install(
  EXPORT "${TARGETS_EXPORT_NAME}"
  NAMESPACE "${namespace}"
  DESTINATION "${cfg_dest}")

# Install public headers under their existing include paths.
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/splinepy"
  DESTINATION "${incl_dest}"
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN "*.h"
  PATTERN "*.inl"
  PATTERN "detail" EXCLUDE)
