function(create_cython_target _name)
  if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/${_name}.cxx)
    set(${_name}
        ${CMAKE_CURRENT_LIST_DIR}/${_name}.cxx
        PARENT_SCOPE)
  else()
    find_package(Cython REQUIRED)
    # should use target_include_directories once this is supported by
    # scikit-build
    include_directories(${RF_BASE_DIR}/rapidfuzz)
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

if(RAPIDFUZZ_ARCH_X64 OR RAPIDFUZZ_ARCH_X86)
  create_cython_target(_feature_detector_cpp)
  rf_add_library(_feature_detector_cpp ${_feature_detector_cpp}
                 ${CMAKE_CURRENT_LIST_DIR}/FeatureDetector/CpuInfo.cpp)
  target_compile_features(_feature_detector_cpp PUBLIC cxx_std_17)
  target_include_directories(_feature_detector_cpp
                             PRIVATE ${RF_BASE_DIR}/rapidfuzz)
  target_link_libraries(_feature_detector_cpp PRIVATE rapidfuzz::rapidfuzz)
  install(TARGETS _feature_detector_cpp LIBRARY DESTINATION src/rapidfuzz)
endif()

create_cython_target(utils_cpp)
rf_add_library(utils_cpp ${utils_cpp} ${CMAKE_CURRENT_LIST_DIR}/utils.cpp)
target_compile_features(utils_cpp PUBLIC cxx_std_17)
target_include_directories(utils_cpp PRIVATE ${RF_BASE_DIR}/rapidfuzz)
target_link_libraries(utils_cpp PRIVATE rapidfuzz::rapidfuzz)
install(TARGETS utils_cpp LIBRARY DESTINATION src/rapidfuzz)

create_cython_target(fuzz_cpp)
rf_add_library(fuzz_cpp ${fuzz_cpp})
target_compile_features(fuzz_cpp PUBLIC cxx_std_17)
target_include_directories(fuzz_cpp PRIVATE ${RF_BASE_DIR}/rapidfuzz)
target_link_libraries(fuzz_cpp PRIVATE rapidfuzz::rapidfuzz)
install(TARGETS fuzz_cpp LIBRARY DESTINATION src/rapidfuzz)

if(RAPIDFUZZ_ARCH_X64 OR RAPIDFUZZ_ARCH_X86)
  # on macOs this generates illegal instructions https://github.com/rapidfuzz/RapidFuzz/issues/383
  if(NOT APPLE)
    create_cython_target(fuzz_cpp_avx2)
    rf_add_library(fuzz_cpp_avx2 ${fuzz_cpp_avx2})
    target_compile_features(fuzz_cpp_avx2 PUBLIC cxx_std_17)

    if(MSVC)
      set_target_properties(fuzz_cpp_avx2 PROPERTIES COMPILE_FLAGS "/arch:AVX2")
    else()
      set_target_properties(fuzz_cpp_avx2 PROPERTIES COMPILE_FLAGS "-mavx2")
    endif()

    target_include_directories(fuzz_cpp_avx2 PRIVATE ${RF_BASE_DIR}/rapidfuzz)
    target_link_libraries(fuzz_cpp_avx2 PRIVATE rapidfuzz::rapidfuzz)
    install(TARGETS fuzz_cpp_avx2 LIBRARY DESTINATION src/rapidfuzz)
  endif()
endif()

if(RAPIDFUZZ_ARCH_X86)
  create_cython_target(fuzz_cpp_sse2)
  rf_add_library(fuzz_cpp_sse2 ${fuzz_cpp_sse2})
  target_compile_features(fuzz_cpp_sse2 PUBLIC cxx_std_17)

  if(MSVC)
    set_target_properties(fuzz_cpp_sse2 PROPERTIES COMPILE_FLAGS "/arch:SSE2")
  else()
    set_target_properties(fuzz_cpp_sse2 PROPERTIES COMPILE_FLAGS "-msse2")
  endif()

  target_include_directories(fuzz_cpp_sse2 PRIVATE ${RF_BASE_DIR}/rapidfuzz)
  target_link_libraries(fuzz_cpp_sse2 PRIVATE rapidfuzz::rapidfuzz)
  install(TARGETS fuzz_cpp_sse2 LIBRARY DESTINATION src/rapidfuzz)
endif()

create_cython_target(process_cpp_impl)
rf_add_library(process_cpp_impl ${process_cpp_impl})
target_compile_features(process_cpp_impl PUBLIC cxx_std_17)
target_include_directories(process_cpp_impl PRIVATE ${RF_BASE_DIR}/rapidfuzz)
target_link_libraries(process_cpp_impl PRIVATE Taskflow::Taskflow
                                               rapidfuzz::rapidfuzz)

# check if we need to link with libatomic (not needed on Windows)
if(NOT Windows)
  # TODO: migrate to CheckSourceCompiles in CMake >= 3.19
  include(CheckCXXSourceCompiles)

  set(ATOMICS_TEST_SOURCE
      [=[
        #include <atomic>
        #include <cstdint>
        #include <cstddef>
        std::atomic<int> x{0};
        int main() {
            auto c = x.load(std::memory_order_relaxed);
            return 0;
        }
    ]=])
  string(REPLACE "std::atomic<int>" "std::atomic<std::size_t>"
                 ATOMICS_SIZE_T_TEST_SOURCE "${ATOMICS_TEST_SOURCE}")
  string(REPLACE "std::atomic<int>" "std::atomic<void*>"
                 ATOMICS_VOID_PTR_TEST_SOURCE "${ATOMICS_TEST_SOURCE}")
  string(REPLACE "std::atomic<int>" "std::atomic<unsigned>"
                 ATOMICS_UNSIGNED_TEST_SOURCE "${ATOMICS_TEST_SOURCE}")
  string(REPLACE "std::atomic<int>" "std::atomic<uint64_t>"
                 ATOMICS_UINT64_TEST_SOURCE "${ATOMICS_TEST_SOURCE}")

  if(APPLE)
    set(CMAKE_REQUIRED_FLAGS "-std=c++11")
  endif()
  check_cxx_source_compiles("${ATOMICS_TEST_SOURCE}"
                            HAVE_CXX_ATOMICS_INT_WITHOUT_LIB)
  check_cxx_source_compiles("${ATOMICS_SIZE_T_TEST_SOURCE}"
                            HAVE_CXX_ATOMICS_SIZE_T_WITHOUT_LIB)
  check_cxx_source_compiles("${ATOMICS_VOID_PTR_TEST_SOURCE}"
                            HAVE_CXX_ATOMICS_VOID_PTR_WITHOUT_LIB)
  check_cxx_source_compiles("${ATOMICS_UNSIGNED_TEST_SOURCE}"
                            HAVE_CXX_ATOMICS_UNSIGNED_WITHOUT_LIB)
  check_cxx_source_compiles("${ATOMICS_UINT64_TEST_SOURCE}"
                            HAVE_CXX_ATOMICS_UINT64_WITHOUT_LIB)
  if((NOT HAVE_CXX_ATOMICS_INT_WITHOUT_LIB)
     OR (NOT HAVE_CXX_ATOMICS_SIZE_T_WITHOUT_LIB)
     OR (NOT HAVE_CXX_ATOMICS_VOID_PTR_WITHOUT_LIB)
     OR (NOT HAVE_CXX_ATOMICS_UNSIGNED_WITHOUT_LIB)
     OR (NOT HAVE_CXX_ATOMICS_UINT64_WITHOUT_LIB))
    set(CMAKE_REQUIRED_LIBRARIES "atomic")
    check_cxx_source_compiles("${ATOMICS_TEST_SOURCE}"
                              HAVE_CXX_ATOMICS_INT_WITH_LIB)
    check_cxx_source_compiles("${ATOMICS_SIZE_T_TEST_SOURCE}"
                              HAVE_CXX_ATOMICS_SIZE_T_WITH_LIB)
    check_cxx_source_compiles("${ATOMICS_VOID_PTR_TEST_SOURCE}"
                              HAVE_CXX_ATOMICS_VOID_PTR_WITH_LIB)
    check_cxx_source_compiles("${ATOMICS_UNSIGNED_TEST_SOURCE}"
                              HAVE_CXX_ATOMICS_UNSIGNED_WITH_LIB)
    check_cxx_source_compiles("${ATOMICS_UINT64_TEST_SOURCE}"
                              HAVE_CXX_ATOMICS_UINT64_WITH_LIB)
    if(NOT HAVE_CXX_ATOMICS_INT_WITH_LIB)
      message(
        FATAL_ERROR
          "No native support for std::atomic<int>, or libatomic not found!")
    elseif(NOT HAVE_CXX_ATOMICS_SIZE_T_WITH_LIB)
      message(
        FATAL_ERROR
          "No native support for std::atomic<size_t>, or libatomic not found!")
    elseif(NOT HAVE_CXX_ATOMICS_VOID_PTR_WITH_LIB)
      message(
        FATAL_ERROR
          "No native support for std::atomic<void*>, or libatomic not found!")
    elseif(NOT HAVE_CXX_ATOMICS_UNSIGNED_WITH_LIB)
      message(
        FATAL_ERROR
          "No native support for std::atomic<unsigned>, or libatomic not found!"
      )
    elseif(NOT HAVE_CXX_ATOMICS_UINT64_WITH_LIB)
      message(
        FATAL_ERROR
          "No native support for std::atomic<uint64>, or libatomic not found!")
    else()
      message(STATUS "Linking with libatomic for atomics support")
      unset(CMAKE_REQUIRED_LIBRARIES)
      target_link_libraries(process_cpp_impl PUBLIC atomic)
    endif()
  endif()
  if(APPLE)
    unset(CMAKE_REQUIRED_FLAGS)
  endif()
endif()

install(TARGETS process_cpp_impl LIBRARY DESTINATION src/rapidfuzz)
