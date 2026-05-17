if(NOT DEFINED WENET_GH_MIRROR)
  set(WENET_GH_MIRROR "https://gh-proxy.com/https://github.com")
endif()
FetchContent_Declare(glog
  URL      ${WENET_GH_MIRROR}/google/glog/archive/v0.4.0.zip
  URL_HASH SHA256=9e1b54eb2782f53cd8af107ecf08d2ab64b8d0dc2b7f5594472f3bd63ca85cdc
)
FetchContent_MakeAvailable(glog)
include_directories(${glog_SOURCE_DIR}/src ${glog_BINARY_DIR})