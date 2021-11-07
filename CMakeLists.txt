cmake_minimum_required(VERSION 3.11.0)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.9" CACHE STRING "Minimum OS X deployment version")
endif()

project(rapidfuzz LANGUAGES C CXX)

find_package(NumPy REQUIRED)
find_package(PythonExtensions REQUIRED)

add_library(cpp_utils MODULE src/cpp_utils.cpp extern/rapidfuzz-cpp/rapidfuzz/details/unicode.cpp)
target_compile_features(cpp_utils PUBLIC cxx_std_14)
target_include_directories(cpp_utils PRIVATE extern/rapidfuzz-cpp/ capi/)
python_extension_module(cpp_utils)
install(TARGETS cpp_utils LIBRARY DESTINATION src/rapidfuzz)

add_library(cpp_fuzz MODULE src/cpp_fuzz.cpp extern/rapidfuzz-cpp/rapidfuzz/details/unicode.cpp)
target_compile_features(cpp_fuzz PUBLIC cxx_std_14)
target_include_directories(cpp_fuzz PRIVATE extern/rapidfuzz-cpp/ capi/)
python_extension_module(cpp_fuzz)
install(TARGETS cpp_fuzz LIBRARY DESTINATION src/rapidfuzz)

add_library(cpp_string_metric MODULE src/cpp_string_metric.cpp)
target_compile_features(cpp_string_metric PUBLIC cxx_std_14)
target_include_directories(cpp_string_metric PRIVATE extern/rapidfuzz-cpp/ capi/)
python_extension_module(cpp_string_metric)
install(TARGETS cpp_string_metric LIBRARY DESTINATION src/rapidfuzz)

add_library(cpp_process MODULE src/cpp_process.cpp)
target_compile_features(cpp_process PUBLIC cxx_std_14)
target_include_directories(cpp_process PRIVATE extern/rapidfuzz-cpp/ capi/)
python_extension_module(cpp_process)
install(TARGETS cpp_process LIBRARY DESTINATION src/rapidfuzz)

add_library(cpp_process_cdist MODULE src/cpp_process_cdist.cpp)
target_compile_features(cpp_process_cdist PUBLIC cxx_std_14)
target_include_directories(cpp_process_cdist PRIVATE extern/rapidfuzz-cpp/ capi/ extern/taskflow/ ${NumPy_INCLUDE_DIR})
python_extension_module(cpp_process_cdist)
install(TARGETS cpp_process_cdist LIBRARY DESTINATION src/rapidfuzz)
