cmake_minimum_required(VERSION 3.19)
project(splinepy_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SPLINEPY_CORE_BUILD_TESTS "Build splinepy core tests" ON)

# Core library target (zero Python dependencies)
add_library(splinepy_core_lib STATIC
    src/validate/validator.cpp
    src/math/binomial.cpp
    src/math/bernstein.cpp
    src/math/bspline_basis.cpp
    src/queries/evaluate.cpp
    src/queries/derivative.cpp
    src/queries/jacobian.cpp
    src/queries/basis.cpp
    src/manipulation/insert_knots.cpp
    src/manipulation/remove_knots.cpp
    src/manipulation/elevate_degree.cpp
    src/manipulation/greville.cpp
    src/manipulation/extract.cpp
    src/bezier/split.cpp
    src/bezier/add.cpp
    src/bezier/multiply.cpp
    src/bezier/derivative_spline.cpp
    src/bezier/compose.cpp
    src/multipatch/interfaces.cpp
    src/multipatch/orientations.cpp
    src/multipatch/boundary.cpp
    src/multipatch/continuity.cpp
)

target_include_directories(splinepy_core_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(splinepy_core_lib PUBLIC cxx_std_17)
target_compile_options(splinepy_core_lib PRIVATE -fPIC -Wall -Wextra -Wpedantic)

# Modern alias
add_library(splinepy::core ALIAS splinepy_core_lib)

# Tests
if(SPLINEPY_CORE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
option(SPLINEPY_CORE_BUILD_BENCHMARKS "Build splinepy core benchmarks" ON)
if(SPLINEPY_CORE_BUILD_BENCHMARKS)
    add_executable(core_benchmarks benchmarks/benchmark_queries.cpp)
    target_link_libraries(core_benchmarks PRIVATE splinepy::core)
    target_compile_options(core_benchmarks PRIVATE -Wall -Wextra -O3)
endif()
