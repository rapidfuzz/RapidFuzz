---
template: overrides/main.html
---

# Installation

=== "Python"
    While there are several ways of install RapidFUzz, the recommended methods
    are either by using `pip`(the Python package manager) or
    `conda` (an open-source, cross-platform, package manager)

    ## with pip <small>recommended</small>

    RapidFuzz can be installed with `pip`:

    ``` sh
    pip install rapidfuzz
    ```
    There are pre-built binaries (wheels) of RapidFuzz for MacOS (10.9 and later), Linux x86_64 and Windows.

    !!! failure "ImportError: DLL load failed"

        If you run into this error the reason is most likely, that the
        [C++-Buildtools](https://visualstudio.microsoft.com/visual-cpp-build-tools)
        are not installed, which are required for Python to find the dynamic library on Windows.

    ## with conda <small>recommended</small>

    RapidFuzz can be installed with `conda`:

    ``` sh
    conda install -c conda-forge rapidfuzz
    ```

    ## from git

    RapidFuzz can be directly used from [GitHub][8] by cloning the
    repository which might be useful when you want to work on it:

    ``` sh
    git clone https://github.com/maxbachmann/rapidfuzz.git
    cd rapidfuzz
    pip install .
    ```

=== "C++"
    As of now it it only possible to use the sources directly by adding them to your
    project. There will be a version on `conan` in the future
