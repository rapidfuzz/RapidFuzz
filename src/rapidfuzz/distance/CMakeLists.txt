function(create_cython_target _name)
  if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/${_name}.cxx)
    set(${_name}
        ${CMAKE_CURRENT_LIST_DIR}/${_name}.cxx
        PARENT_SCOPE)
  else()
    find_package(Cython REQUIRED)
    # should use target_include_directories once this is supported by
    # scikit-build
    include_directories(${RF_BASE_DIR}/rapidfuzz
                        ${RF_BASE_DIR}/rapidfuzz/distance)
    add_cython_target(${_name} CXX)
    set(${_name}
        ${_name}
        PARENT_SCOPE)
  endif()
endfunction(create_cython_target)

function(rf_add_library name)
  if(CMAKE_VERSION VERSION_LESS 3.17)
    python_add_library(${name} MODULE ${ARGV})
    get_property(
      suffix
      TARGET ${name}
      PROPERTY SUFFIX)
    if(NOT suffix)
      set(suffix "${CMAKE_SHARED_MODULE_SUFFIX}")
    endif()
    set_property(TARGET ${name} PROPERTY SUFFIX ".${Python_SOABI}${suffix}")
  else()
    python_add_library(${name} MODULE WITH_SOABI ${ARGV})
  endif()
endfunction(rf_add_library)

create_cython_target(_initialize_cpp)
rf_add_library(_initialize_cpp ${_initialize_cpp})
target_compile_features(_initialize_cpp PUBLIC cxx_std_17)
target_include_directories(
  _initialize_cpp PRIVATE ${RF_BASE_DIR}/rapidfuzz
                          ${RF_BASE_DIR}/rapidfuzz/distance)
target_link_libraries(_initialize_cpp PRIVATE rapidfuzz::rapidfuzz)
install(TARGETS _initialize_cpp LIBRARY DESTINATION src/rapidfuzz/distance)

create_cython_target(metrics_cpp)
rf_add_library(metrics_cpp ${metrics_cpp})
target_compile_features(metrics_cpp PUBLIC cxx_std_17)
target_include_directories(
  metrics_cpp PRIVATE ${RF_BASE_DIR}/rapidfuzz
                      ${RF_BASE_DIR}/rapidfuzz/distance)
target_link_libraries(metrics_cpp PRIVATE rapidfuzz::rapidfuzz)
install(TARGETS metrics_cpp LIBRARY DESTINATION src/rapidfuzz/distance)

if(RAPIDFUZZ_ARCH_X64 OR RAPIDFUZZ_ARCH_X86)
  # on macOs this generates illegal instructions https://github.com/rapidfuzz/RapidFuzz/issues/383
  if(NOT APPLE)
    create_cython_target(metrics_cpp_avx2)
    rf_add_library(metrics_cpp_avx2 ${metrics_cpp_avx2})
    target_compile_features(metrics_cpp_avx2 PUBLIC cxx_std_17)

    if(MSVC)
      set_target_properties(metrics_cpp_avx2 PROPERTIES COMPILE_FLAGS
                                                        "/arch:AVX2")
    else()
      set_target_properties(metrics_cpp_avx2 PROPERTIES COMPILE_FLAGS "-mavx2")
    endif()

    target_include_directories(
      metrics_cpp_avx2 PRIVATE ${RF_BASE_DIR}/rapidfuzz
                              ${RF_BASE_DIR}/rapidfuzz/distance)
    target_link_libraries(metrics_cpp_avx2 PRIVATE rapidfuzz::rapidfuzz)
    install(TARGETS metrics_cpp_avx2 LIBRARY DESTINATION src/rapidfuzz/distance)
  endif()
endif()

if(RAPIDFUZZ_ARCH_X86)
  create_cython_target(metrics_cpp_sse2)
  rf_add_library(metrics_cpp_sse2 ${metrics_cpp_sse2})
  target_compile_features(metrics_cpp_sse2 PUBLIC cxx_std_17)

  if(MSVC)
    set_target_properties(metrics_cpp_sse2 PROPERTIES COMPILE_FLAGS
                                                      "/arch:SSE2")
  else()
    set_target_properties(metrics_cpp_sse2 PROPERTIES COMPILE_FLAGS "-msse2")
  endif()

  target_include_directories(
    metrics_cpp_sse2 PRIVATE ${RF_BASE_DIR}/rapidfuzz
                             ${RF_BASE_DIR}/rapidfuzz/distance)
  target_link_libraries(metrics_cpp_sse2 PRIVATE rapidfuzz::rapidfuzz)
  install(TARGETS metrics_cpp_sse2 LIBRARY DESTINATION src/rapidfuzz/distance)
endif()
