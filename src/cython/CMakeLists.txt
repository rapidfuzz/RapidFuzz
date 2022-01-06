
add_cython_target(cpp_utils CXX)
add_library(cpp_utils MODULE
    ${cpp_utils}
    ${RAPIDFUZZ_BASE_DIR}/extern/rapidfuzz-cpp/rapidfuzz/details/unicode.cpp)
target_compile_features(cpp_utils PUBLIC cxx_std_14)
target_include_directories(cpp_utils PRIVATE
    ${RAPIDFUZZ_BASE_DIR}/extern/rapidfuzz-cpp/
    ${RAPIDFUZZ_CAPI_PATH}
    ${RAPIDFUZZ_BASE_DIR}/src/cython)
python_extension_module(cpp_utils)
install(TARGETS cpp_utils LIBRARY DESTINATION src/rapidfuzz)

add_cython_target(cpp_fuzz CXX)
add_library(cpp_fuzz MODULE
    ${cpp_fuzz}
    ${RAPIDFUZZ_BASE_DIR}/extern/rapidfuzz-cpp/rapidfuzz/details/unicode.cpp)
target_compile_features(cpp_fuzz PUBLIC cxx_std_14)
target_include_directories(cpp_fuzz PRIVATE
    ${RAPIDFUZZ_BASE_DIR}/extern/rapidfuzz-cpp/
    ${RAPIDFUZZ_CAPI_PATH}
    ${RAPIDFUZZ_BASE_DIR}/src/cython)
python_extension_module(cpp_fuzz)
install(TARGETS cpp_fuzz LIBRARY DESTINATION src/rapidfuzz)

add_cython_target(cpp_string_metric CXX)
add_library(cpp_string_metric MODULE ${cpp_string_metric})
target_compile_features(cpp_string_metric PUBLIC cxx_std_14)
target_include_directories(cpp_string_metric PRIVATE
    ${RAPIDFUZZ_BASE_DIR}/extern/rapidfuzz-cpp/
    ${RAPIDFUZZ_CAPI_PATH}
    ${RAPIDFUZZ_BASE_DIR}/src/cython)
python_extension_module(cpp_string_metric)
install(TARGETS cpp_string_metric LIBRARY DESTINATION src/rapidfuzz)

add_cython_target(cpp_process CXX)
add_library(cpp_process MODULE ${cpp_process})
target_compile_features(cpp_process PUBLIC cxx_std_14)
target_include_directories(cpp_process PRIVATE
    ${RAPIDFUZZ_BASE_DIR}/extern/rapidfuzz-cpp/
    ${RAPIDFUZZ_BASE_DIR}/extern/optional-lite/include
    ${RAPIDFUZZ_CAPI_PATH}
    ${RAPIDFUZZ_BASE_DIR}/src/cython)
python_extension_module(cpp_process)
install(TARGETS cpp_process LIBRARY DESTINATION src/rapidfuzz)

add_cython_target(cpp_process_cdist CXX)
add_library(cpp_process_cdist MODULE ${cpp_process_cdist})
target_compile_features(cpp_process_cdist PUBLIC cxx_std_14)
target_include_directories(cpp_process_cdist PRIVATE
    ${RAPIDFUZZ_BASE_DIR}/extern/rapidfuzz-cpp/
    ${RAPIDFUZZ_BASE_DIR}/extern/optional-lite/include
    ${RAPIDFUZZ_CAPI_PATH}
    ${RAPIDFUZZ_BASE_DIR}/src/cython
    ${RAPIDFUZZ_BASE_DIR}/extern/taskflow/
    ${NumPy_INCLUDE_DIR})
# this fails with All uses of target_link_libraries with a target must be either all-keyword or all-plain.
# target_link_libraries(cpp_process_cdist PRIVATE Threads::Threads)
target_link_libraries(cpp_process_cdist Threads::Threads)
python_extension_module(cpp_process_cdist)
install(TARGETS cpp_process_cdist LIBRARY DESTINATION src/rapidfuzz)
