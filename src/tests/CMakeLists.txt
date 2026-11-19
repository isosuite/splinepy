include(FetchContent)

find_package(GTest QUIET)

if(NOT GTest_FOUND)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    GIT_SHALLOW TRUE)
  set(gtest_force_shared_crt
      ON
      CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

add_executable(
  core_tests
  test_main.cpp
  test_buffer.cpp
  test_span.cpp
  test_validator.cpp
  test_hierarchy.cpp
  test_parallel_for.cpp
  test_team.cpp
  test_binomial.cpp
  test_bernstein.cpp
  test_bspline_basis.cpp
  test_queries.cpp
  test_manipulation.cpp
  test_bezier_algebra.cpp
  test_multipatch.cpp
  test_workspace.cpp
  test_element_bvh.cpp
  test_proximity.cpp
  test_query_kernels.cpp)

target_link_libraries(core_tests PRIVATE splinepy::core GTest::gtest_main)

if(MSVC)
  target_compile_options(core_tests PRIVATE /W4)
else()
  target_compile_options(core_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
target_compile_definitions(
  core_tests
  PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/data/golden/bin")

include(GoogleTest)
gtest_discover_tests(core_tests)
