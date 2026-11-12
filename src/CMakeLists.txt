# create splinepy target - enables cpp standalone use define srcs
set(SPLINEPY_SRCS ${PROJECT_SOURCE_DIR}/src/proximity/proximity.cpp)

set(SPLINEPY_SLSQP_SRCS ${PROJECT_SOURCE_DIR}/src/proximity/slsqp/slsqp.c)

add_library(slsqp ${SPLINEPY_SLSQP_SRCS})
add_library(splinepy::slsqp ALIAS slsqp)
target_compile_options(slsqp PRIVATE -fPIC ${SPLINEPY_OPTIMIZATION_FLAGS})
target_compile_definitions(slsqp PRIVATE ${SPLINEPY_DEFS})
target_include_directories(
  slsqp PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
               $<INSTALL_INTERFACE:${incl_dest}>)

# target
add_library(splinepy ${SPLINEPY_LIB_TYPE} ${SPLINEPY_SRCS})
# alias for convenience
add_library(splinepy::splinepy ALIAS splinepy)

# include
target_include_directories(
  splinepy PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                  $<INSTALL_INTERFACE:${incl_dest}>)

# link
target_link_libraries(splinepy PUBLIC splinepy::core napf slsqp)

# flags
target_compile_options(splinepy PRIVATE ${SPLINEPY_FLAGS})

# defs
target_compile_definitions(splinepy PRIVATE ${SPLINEPY_DEFS})

# features
target_compile_features(splinepy PUBLIC cxx_std_17)

# python?
if(SPLINEPY_COMPILE_PYTHON)
  add_subdirectory(py)
endif()

# cmake config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${version_config}"
                                 COMPATIBILITY SameMajorVersion)
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/config.cmake.in" "${project_config}"
  INSTALL_DESTINATION "${cfg_dest}")

# install main target
install(
  TARGETS splinepy
  EXPORT "${TARGETS_EXPORT_NAME}"
  LIBRARY DESTINATION ${lib_dest}
  ARCHIVE DESTINATION ${lib_dest}
  INCLUDES
  DESTINATION "${incl_dest}")

install(
  TARGETS slsqp
  EXPORT "${TARGETS_EXPORT_NAME}"
  LIBRARY DESTINATION ${lib_dest}
  ARCHIVE DESTINATION ${lib_dest}
  INCLUDES
  DESTINATION "${incl_dest}")

# config files
install(FILES "${project_config}" "${version_config}" DESTINATION "${cfg_dest}")
install(
  EXPORT "${TARGETS_EXPORT_NAME}"
  NAMESPACE "${namespace}"
  DESTINATION "${cfg_dest}")

# header files - keep tree
install(
  DIRECTORY "${PROJECT_SOURCE_DIR}/include/splinepy"
  DESTINATION ${incl_dest} # target directory
  FILES_MATCHING
  PATTERN "*.hpp")

install(
  DIRECTORY "${PROJECT_SOURCE_DIR}/include/splinepy"
  DESTINATION ${incl_dest} # target directory
  FILES_MATCHING
  PATTERN "*.h")

# inline files - keep tree
install(
  DIRECTORY "${PROJECT_SOURCE_DIR}/include/splinepy"
  DESTINATION ${incl_dest} # target directory
  FILES_MATCHING
  PATTERN "*.inl")
