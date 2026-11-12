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

# config files
install(FILES "${project_config}" "${version_config}" DESTINATION "${cfg_dest}")
install(
  EXPORT "${TARGETS_EXPORT_NAME}"
  NAMESPACE "${namespace}"
  DESTINATION "${cfg_dest}")

# header files - keep tree
install(
  DIRECTORY "${PROJECT_SOURCE_DIR}/core/include/splinepy"
  DESTINATION ${incl_dest} # target directory
  FILES_MATCHING
  PATTERN "*.hpp")

install(
  DIRECTORY "${PROJECT_SOURCE_DIR}/core/include/splinepy"
  DESTINATION ${incl_dest} # target directory
  FILES_MATCHING
  PATTERN "*.h")

# inline files - keep tree
install(
  DIRECTORY "${PROJECT_SOURCE_DIR}/core/include/splinepy"
  DESTINATION ${incl_dest} # target directory
  FILES_MATCHING
  PATTERN "*.inl")
