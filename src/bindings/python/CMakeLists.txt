find_package(
  Python 3.10
  COMPONENTS Interpreter Development.Module
  REQUIRED)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(splinepy_core splinepy_core.cpp nb_spline.cpp
                    nb_multipatch.cpp nb_spline_extensions.cpp nb_hierarchy.cpp)

target_link_libraries(splinepy_core PRIVATE splinepy::core)

# as for the core: query loops instantiated here keep their place
if(NOT MSVC)
  target_compile_options(splinepy_core PRIVATE -falign-functions=64)
endif()

install(
  TARGETS splinepy_core
  DESTINATION splinepy
  COMPONENT PythonModule)
