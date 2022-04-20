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
include(FetchContent)

set(RF_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Taskflow 3.0.0 QUIET)
if (NOT Taskflow_FOUND)
    find_package(Taskflow 2.7.0 QUIET)
endif()
if (Taskflow_FOUND)
    message("Using system supplied version of Taskflow")
else()
    message("Using FetchContent to load Taskflow")
    set(TF_BUILD_CUDA OFF CACHE BOOL "Enables build of CUDA code")
    set(TF_BUILD_TESTS OFF CACHE BOOL "Enables build of tests")
    set(TF_BUILD_EXAMPLES OFF CACHE BOOL "Enables build of examples")
    FetchContent_Declare(
        Taskflow
        GIT_REPOSITORY https://github.com/taskflow/taskflow.git
        # v2.7.0 is the last version supporting C++14. Newer versions
        # can be used once the project uses C++17 as well
        GIT_TAG f69ae7ce714baab30051b0edbc2f091ceba77916 # v2.7.0
        GIT_SHALLOW TRUE
    )
    # Taskflow installs itself even when included as subdirectory
    # workaround this by reimplementing FetchContent_MakeAvailable with
    # EXCLUDE_FROM_ALL
    FetchContent_GetProperties(Taskflow)
    if(NOT Taskflow_POPULATED)
        FetchContent_Populate(Taskflow)
        add_subdirectory(${taskflow_SOURCE_DIR} ${taskflow_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
    add_library(Taskflow::Taskflow ALIAS Taskflow)
endif()

find_package(rapidfuzz 1.0.1 QUIET)
if (rapidfuzz_FOUND)
    message("Using system supplied version of rapidfuzz-cpp")
else()
    message("Using FetchContent to load rapidfuzz-cpp")
    FetchContent_Declare(
        rapidfuzz
        GIT_REPOSITORY https://github.com/maxbachmann/rapidfuzz-cpp.git
        GIT_TAG 6642000b428b952de0fdf4d1f4be34e0b7b1eca3 # v1.0.1
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(rapidfuzz)
endif()

find_package(jaro_winkler 1.0.0 QUIET)
if (jaro_winkler_FOUND)
    message("Using system supplied version of jaro_winkler")
else()
    message("Using FetchContent to load jaro_winkler")
    FetchContent_Declare(
        jaro_winkler
        GIT_REPOSITORY https://github.com/maxbachmann/jarowinkler-cpp.git
        GIT_TAG 09f221c6b21d266823d5d9ea73b4673b6de2e80f # v1.0.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(jaro_winkler)
endif()

add_subdirectory(rapidfuzz)
add_subdirectory(rapidfuzz/distance)
