cmake_minimum_required(VERSION 3.19.0)

project(
  splinepy
  VERSION 0.1.1
  LANGUAGES C CXX)

# build options
option(SPLINEPY_VERBOSE_MAKE
       "Verbose `make` output. Alias to CMAKE_VERBOSE_MAKEFILE" OFF)
option(SPLINEPY_COMPILE_PYTHON "Compile python module." ON)

# config
set(exe_dest "bin")
set(incl_dest "include")
set(lib_dest "lib")
set(cfg_dest "${lib_dest}/cmake/${PROJECT_NAME}")
set(gen_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(version_config "${gen_dir}/${PROJECT_NAME}ConfigVersion.cmake")
set(project_config "${gen_dir}/${PROJECT_NAME}Config.cmake")
set(TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")
set(namespace "${PROJECT_NAME}::")

# default build type id debug
if(NOT CMAKE_BUILD_TYPE)
  message("CMAKE_BUILD_TYPE undefined. Setting Debug.")
  set(CMAKE_BUILD_TYPE Debug)
endif()

# global flags
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")

if(SPLINEPY_VERBOSE_MAKE)
  set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

message("")
message(
  "                    %%\\ %%\\                                         ")
message("                    %% |\\__|                                        ")
message(
  " %%%%%%%\\  %%%%%%\\  %% |%%\\ %%%%%%%\\   %%%%%%\\   %%%%%%\\  %%\\   %%\\ "
)
message(
  "%%  _____|%%  __%%\\ %% |%% |%%  __%%\\ %%  __%%\\ %%  __%%\\ %% |  %% |")
message(
  "\\%%%%%%\\  %% /  %% |%% |%% |%% |  %% |%%%%%%%% |%% /  %% |%% |  %% |")
message(
  " \\____%%\\ %% |  %% |%% |%% |%% |  %% |%%   ____|%% |  %% |%% |  %% |")
message(
  "%%%%%%%  |%%%%%%%  |%% |%% |%% |  %% |\\%%%%%%%\\ %%%%%%%  |\\%%%%%%% |")
message(
  "\\_______/ %%  ____/ \\__|\\__|\\__|  \\__| \\_______|%%  ____/  \\____%% |")
message("          %% |                                  %% |      %%\\   %% |")
message("          %% |                                  %% |      \\%%%%%%  |")
message(
  "          \\__|                                  \\__|       \\______/")
message("")

set(SPLINEPY_CORE_BUILD_TESTS
    OFF
    CACHE BOOL "Build splinepy core tests")
set(SPLINEPY_CORE_BUILD_BENCHMARKS
    OFF
    CACHE BOOL "Build splinepy core benchmarks")

add_subdirectory(core)
add_subdirectory(src)
