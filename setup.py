from __future__ import annotations

import os


def show_message(*lines):
    print("=" * 74)
    for line in lines:
        print(line)
    print("=" * 74)


with open("README.md", encoding="utf8") as f:
    readme = f.read()

setup_args = {
    "name": "rapidfuzz",
    "version": "3.9.1",
    "extras_require": {"full": ["numpy"]},
    "url": "https://github.com/rapidfuzz/RapidFuzz",
    "author": "Max Bachmann",
    "author_email": "pypi@maxbachmann.de",
    "description": "rapid fuzzy string matching",
    "long_description": readme,
    "long_description_content_type": "text/markdown",
    "license": "MIT",
    "classifiers": [
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
    ],
    "packages": ["rapidfuzz", "rapidfuzz.distance", "rapidfuzz.__pyinstaller"],
    "entry_points": {
        "pyinstaller40": [
            "hook-dirs = rapidfuzz.__pyinstaller:get_hook_dirs",
            "tests = rapidfuzz.__pyinstaller:get_PyInstaller_tests",
        ],
    },
    "package_dir": {
        "": "src",
    },
    "package_data": {
        "rapidfuzz": ["*.pyi", "py.typed", "__init__.pxd", "rapidfuzz.h"],
        "rapidfuzz.distance": ["*.pyi"],
    },
    "python_requires": ">=3.8",
}


def run_setup(with_binary):
    if with_binary:
        from skbuild import setup

        setup(**setup_args)
    else:
        from setuptools import setup

        setup(**setup_args)


# when packaging only build wheels which include the C extension
packaging = "1" in {
    os.environ.get("CIBUILDWHEEL", "0"),
    os.environ.get("CONDA_BUILD", "0"),
    os.environ.get("PIWHEELS_BUILD", "0"),
    os.environ.get("RAPIDFUZZ_BUILD_EXTENSION", "0"),
}
if packaging:
    run_setup(True)
else:
    try:
        run_setup(True)
    except BaseException:
        show_message(
            "WARNING: The C extension could not be compiled, speedups are not enabled.",
            "Failure information, if any, is above.",
            "Retrying the build without the C extension now.",
        )
        run_setup(False)
        show_message(
            "WARNING: The C extension could not be compiled, speedups are not enabled.",
            "Plain-Python build succeeded.",
        )
