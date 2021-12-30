cmake_minimum_required(VERSION 3.12.0)

cmake_policy(SET CMP0054 NEW)

set(THREADS_PREFER_PTHREAD_FLAG ON)
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.9" CACHE STRING "Minimum OS X deployment version")
endif()

project(rapidfuzz LANGUAGES C CXX)

find_package(NumPy REQUIRED)
find_package(PythonExtensions REQUIRED)
find_package(Threads REQUIRED)
find_package(Python COMPONENTS Interpreter Development)
find_package(Cython REQUIRED)

add_subdirectory(src)