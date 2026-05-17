set(decoder_srcs
  asr_decoder.cc
  asr_model.cc
  context_graph.cc
  ctc_prefix_beam_search.cc
  ctc_wfst_beam_search.cc
  ctc_endpoint.cc
  corrector.cc
  hotword_cache.cc
)

if(NOT TORCH AND NOT ONNX AND NOT XPU AND NOT IOS AND NOT BPU AND NOT OPENVINO)
  message(FATAL_ERROR "Please build with TORCH or ONNX or OPENVINO or XPU or IOS or BPU!!!")
endif()
if(TORCH OR IOS)
  list(APPEND decoder_srcs torch_asr_model.cc)
endif()
if(ONNX)
  list(APPEND decoder_srcs onnx_asr_model.cc)
endif()

add_library(decoder STATIC ${decoder_srcs})
target_link_libraries(decoder PUBLIC kaldi-decoder frontend
                      post_processor utils cpp-pinyin::cpp-pinyin)

if(JEMALLOC)
  set(jemalloc_SOURCE_DIR ${fc_base}/jemalloc-src CACHE PATH "jemalloc source directory")
  set(jemalloc_BINARY_DIR ${fc_base}/jemalloc-build CACHE PATH "jemalloc build directory")
  set(jemalloc_SUBBUILD_DIR ${fc_base}/jemalloc-subbuild CACHE PATH "jemalloc subbuild directory")
  set(jemalloc_PREFIX_DIR ${jemalloc_SUBBUILD_DIR}/jemalloc-populate-prefix CACHE PATH "jemalloc prefix directory")
  ExternalProject_Add(jemalloc
	  #URL               https://github.com/jemalloc/jemalloc/releases/download/5.2.1/jemalloc-5.2.1.tar.bz2
	  #URL_HASH          SHA256=34330e5ce276099e2e8950d9335db5a875689a4c6a56751ef3b1d8c537f887f6
    TMP_DIR           ${jemalloc_SUBBUILD_DIR}/tmp
    STAMP_DIR         ${jemalloc_SUBBUILD_DIR}/src/jemalloc-stamp
    DOWNLOAD_DIR      ${jemalloc_SUBBUILD_DIR}/src
    SOURCE_DIR        ${jemalloc_SOURCE_DIR}
    BINARY_DIR        ${jemalloc_BINARY_DIR}
    INSTALL_DIR       ${jemalloc_PREFIX_DIR}
    CONFIGURE_COMMAND ${jemalloc_SOURCE_DIR}/configure --prefix=${jemalloc_PREFIX_DIR}
    BUILD_COMMAND     make -j 4
  )
  target_link_libraries(decoder PUBLIC ${jemalloc_PREFIX_DIR}/lib/libjemalloc.a)
endif(JEMALLOC)

if(ANDROID)
  target_link_libraries(decoder PUBLIC ${PYTORCH_LIBRARY} ${FBJNI_LIBRARY})
else()
  if(TORCH)
    target_link_libraries(decoder PUBLIC ${TORCH_LIBRARIES})
  endif()
  if(ONNX)
    target_link_libraries(decoder PUBLIC onnxruntime)
  endif()
  if(BPU)
    target_link_libraries(decoder PUBLIC bpu_asr_model)
  endif()
  if(XPU)
    target_link_libraries(decoder PUBLIC xpu_conformer)
  endif()
  if(OPENVINO)
    target_link_libraries(decoder PUBLIC ov_asr_model)
  endif()
endif()
