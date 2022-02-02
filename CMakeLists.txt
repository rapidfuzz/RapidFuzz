cmake_minimum_required(VERSION 3.12.0)

cmake_policy(SET CMP0054 NEW)
set(SKBUILD_LINK_LIBRARIES_KEYWORD PRIVATE)

set(THREADS_PREFER_PTHREAD_FLAG ON)
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.9" CACHE STRING "Minimum OS X deployment version")
endif()

project(rapidfuzz LANGUAGES C CXX)

if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -pedantic)
endif()

find_package(NumPy REQUIRED)
find_package(PythonExtensions REQUIRED)
find_package(Python COMPONENTS Interpreter Development)
find_package(Cython REQUIRED)

set(RF_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(TF_BUILD_CUDA OFF CACHE BOOL "Enables build of CUDA code")
set(TF_BUILD_TESTS OFF CACHE BOOL "Enables build of tests")
set(TF_BUILD_EXAMPLES OFF CACHE BOOL "Enables build of examples")
add_subdirectory(extern/taskflow)
add_subdirectory(extern/rapidfuzz-cpp)
add_subdirectory(extern/jarowinkler-cpp)

add_subdirectory(src/cython)
add_subdirectory(src/cython/distance)
