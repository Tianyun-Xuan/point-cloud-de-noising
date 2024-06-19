# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_mistnet_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED mistnet_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(mistnet_FOUND FALSE)
  elseif(NOT mistnet_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(mistnet_FOUND FALSE)
  endif()
  return()
endif()
set(_mistnet_CONFIG_INCLUDED TRUE)

# output package information
if(NOT mistnet_FIND_QUIETLY)
  message(STATUS "Found mistnet: 0.0.0 (${mistnet_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'mistnet' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${mistnet_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(mistnet_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${mistnet_DIR}/${_extra}")
endforeach()
