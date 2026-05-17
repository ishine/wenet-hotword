# cpp-pinyin: a lightweight Chinese-to-Pinyin G2P used by hotword corrector
# https://github.com/wolfgitpr/cpp-pinyin
#
# This module supports two paths:
#   1. CPP_PINYIN_SOURCE_DIR=<path>  -> use a local checkout (avoids network)
#   2. (default) FetchContent from a mirror that's reachable from mainland CN.
#      Override the upstream URL with CPP_PINYIN_URL if needed.

set(CPP_PINYIN_VERSION "1.0.2")

if(DEFINED CPP_PINYIN_SOURCE_DIR AND EXISTS "${CPP_PINYIN_SOURCE_DIR}/CMakeLists.txt")
  message(STATUS "Using local cpp-pinyin source: ${CPP_PINYIN_SOURCE_DIR}")
  add_subdirectory("${CPP_PINYIN_SOURCE_DIR}" cpp-pinyin-build EXCLUDE_FROM_ALL)
else()
  if(NOT DEFINED CPP_PINYIN_URL)
    # gh-proxy.com works from mainland China without VPN (the older ghproxy.com
    # endpoint is no longer reachable as of 2025).
    if(NOT DEFINED WENET_GH_MIRROR)
      set(WENET_GH_MIRROR "https://gh-proxy.com/https://github.com")
    endif()
    set(CPP_PINYIN_URL
        "${WENET_GH_MIRROR}/wolfgitpr/cpp-pinyin/archive/refs/tags/${CPP_PINYIN_VERSION}.tar.gz")
  endif()
  message(STATUS "Fetching cpp-pinyin from ${CPP_PINYIN_URL}")
  FetchContent_Declare(cpp_pinyin
    URL ${CPP_PINYIN_URL}
  )
  FetchContent_MakeAvailable(cpp_pinyin)
endif()

# Make cpp-pinyin headers available to the rest of the project. cpp-pinyin
# already exposes them via its INTERFACE include directories on the target, but
# we set a global include directory for downstream targets that have not yet
# been migrated to target-based includes.
if(TARGET cpp-pinyin)
  get_target_property(_cpp_pinyin_include cpp-pinyin INTERFACE_INCLUDE_DIRECTORIES)
  if(_cpp_pinyin_include)
    include_directories(${_cpp_pinyin_include})
  endif()
endif()

# Add the namespaced alias if the upstream did not provide one.
if(NOT TARGET cpp-pinyin::cpp-pinyin AND TARGET cpp-pinyin)
  add_library(cpp-pinyin::cpp-pinyin ALIAS cpp-pinyin)
endif()
