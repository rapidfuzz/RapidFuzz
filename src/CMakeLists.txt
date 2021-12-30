
# should use target_include_directories once this is supported by scikit-build
include_directories(../capi/)

add_cython_target(cpp_utils CXX)
add_library(cpp_utils MODULE ${cpp_utils} ../extern/rapidfuzz-cpp/rapidfuzz/details/unicode.cpp)
target_compile_features(cpp_utils PUBLIC cxx_std_14)
target_include_directories(cpp_utils PRIVATE ../extern/rapidfuzz-cpp/ ../capi/ ../src)
python_extension_module(cpp_utils)
install(TARGETS cpp_utils LIBRARY DESTINATION src/rapidfuzz)

add_cython_target(cpp_fuzz CXX)
add_library(cpp_fuzz MODULE
    ${cpp_fuzz}
    ../extern/rapidfuzz-cpp/rapidfuzz/details/unicode.cpp)
target_compile_features(cpp_fuzz PUBLIC cxx_std_14)
target_include_directories(cpp_fuzz PRIVATE
    ../extern/rapidfuzz-cpp/
    ../capi/
    ../src)
python_extension_module(cpp_fuzz)
install(TARGETS cpp_fuzz LIBRARY DESTINATION src/rapidfuzz)

add_cython_target(cpp_string_metric CXX)
add_library(cpp_string_metric MODULE ${cpp_string_metric})
target_compile_features(cpp_string_metric PUBLIC cxx_std_14)
target_include_directories(cpp_string_metric PRIVATE
    ../extern/rapidfuzz-cpp/
    ../capi/
    ../src)
python_extension_module(cpp_string_metric)
install(TARGETS cpp_string_metric LIBRARY DESTINATION src/rapidfuzz)

add_cython_target(cpp_process CXX)
add_library(cpp_process MODULE ${cpp_process})
target_compile_features(cpp_process PUBLIC cxx_std_14)
target_include_directories(cpp_process PRIVATE
    ../extern/rapidfuzz-cpp/
    ../extern/optional-lite/include
    ../capi/
    ../src)
python_extension_module(cpp_process)
install(TARGETS cpp_process LIBRARY DESTINATION src/rapidfuzz)

add_cython_target(cpp_process_cdist CXX)
add_library(cpp_process_cdist MODULE ${cpp_process_cdist})
target_compile_features(cpp_process_cdist PUBLIC cxx_std_14)
target_include_directories(cpp_process_cdist PRIVATE
    ../extern/rapidfuzz-cpp/
    ../extern/optional-lite/include
    ../capi/
    ../src
    ../extern/taskflow/
    ${NumPy_INCLUDE_DIR})
# this fails with All uses of target_link_libraries with a target must be either all-keyword or all-plain.
# target_link_libraries(cpp_process_cdist PRIVATE Threads::Threads)
target_link_libraries(cpp_process_cdist Threads::Threads)
python_extension_module(cpp_process_cdist)
install(TARGETS cpp_process_cdist LIBRARY DESTINATION src/rapidfuzz)
