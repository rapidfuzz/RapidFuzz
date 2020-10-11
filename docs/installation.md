---
template: overrides/main.html
---

# Installation

=== "Python"
    While there are several ways of install RapidFuzz, the recommended methods
    are either by using `pip`(the Python package manager) or
    `conda` (an open-source, cross-platform, package manager)

    ## with pip <small>recommended</small>

    RapidFuzz can be installed with `pip`:

    ``` sh
    pip install rapidfuzz
    ```
    There are pre-built binaries (wheels) of RapidFuzz for MacOS (10.9 and later), Linux x86_64 and Windows.

    !!! failure "ImportError: DLL load failed"

        If you run into this error on Windows the reason is most likely, that the
        [Visual C++ 2019 redistributable](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads) is not installed, which is required to find C++ Libraries (The C++ 2019 version includes the 2015, 2017 and 2019 version).

    ## with conda <small>recommended</small>

    RapidFuzz can be installed with `conda`:

    ``` sh
    conda install -c conda-forge rapidfuzz
    ```

    ## from git

    RapidFuzz can be directly used from GitHub by cloning the
    repository which might be useful when you want to work on it:

    ``` sh
    git clone https://github.com/maxbachmann/rapidfuzz.git
    cd rapidfuzz
    pip install .
    ```

=== "C++"
    There are severals ways to integrate `rapidfuzz` in your CMake project.

    ## By Installing it
    ```bash
    git clone https://github.com/maxbachmann/rapidfuzz-cpp.git rapidfuzz-cpp
    cd rapidfuzz-cpp
    mkdir build && cd build
    cmake ..
    cmake --build .
    cmake --build . --target install
    ```

    Then in your CMakeLists.txt: 
    ```cmake
    find_package(rapidfuzz REQUIRED)
    add_executable(foo main.cpp)
    target_link_libraries(foo rapidfuzz::rapidfuzz)
    ```

    ## Add this repository as a submodule
    ```bash
    git submodule add https://github.com/maxbachmann/rapidfuzz-cpp.git 3rdparty/RapidFuzz
    ```
    Then you can either:

    1. include it as a subdirectory
        ```cmake
        add_subdirectory(3rdparty/RapidFuzz)
        add_executable(foo main.cpp)
        target_link_libraries(foo rapidfuzz::rapidfuzz)
        ```
    2. build it at configure time with `FetchContent`
        ```cmake
        FetchContent_Declare( 
          rapidfuzz
          SOURCE_DIR ${CMAKE_SOURCE_DIR}/3rdparty/RapidFuzz
          PREFIX ${CMAKE_CURRENT_BINARY_DIR}/rapidfuzz
          CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> "${CMAKE_OPT_ARGS}"
        )
        FetchContent_MakeAvailable(rapidfuzz)
        add_executable(foo main.cpp)
        target_link_libraries(foo PRIVATE rapidfuzz::rapidfuzz)
        ```
    ## Download it at configure time

    If you don't want to add `rapidfuzz-cpp` as a submodule, you can also download it with `FetchContent`:
    ```cmake
    FetchContent_Declare(rapidfuzz
      GIT_REPOSITORY https://github.com/maxbachmann/rapidfuzz-cpp.git
      GIT_TAG master)
    FetchContent_MakeAvailable(rapidfuzz)
    add_executable(foo main.cpp)
    target_link_libraries(foo PRIVATE rapidfuzz::rapidfuzz)
    ```
    It will be downloaded each time you run CMake in a blank folder.   

    ## CMake options

    There are the following CMake options:

    1. `BUILD_TESTS` : to build test (default OFF and requires [Catch2](https://github.com/catchorg/Catch2))

    2. `BUILD_BENCHMARKS` : to build benchmarks (default OFF and requires [Google Benchmark](https://github.com/google/benchmark))

