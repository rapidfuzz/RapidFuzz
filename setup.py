import os

def show_message(*lines):
    print("=" * 74)
    for line in lines:
        print(line)
    print("=" * 74)

with open('README.md', 'rt', encoding="utf8") as f:
    readme = f.read()

setup_args = {
    "name": "rapidfuzz",
    "version": "2.10.1",
    "install_requires": ["jarowinkler >= 1.2.2, < 2.0.0"],
    "extras_require": {'full': ['numpy']},
    "url": "https://github.com/maxbachmann/RapidFuzz",
    "author": "Max Bachmann",
    "author_email": "pypi@maxbachmann.de",
    "description": "rapid fuzzy string matching",
    "long_description": readme,
    "long_description_content_type": "text/markdown",

    "license": "MIT",
    "classifiers": [
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License"
    ],

    "packages": ["rapidfuzz", "rapidfuzz.distance"],
    "package_dir": {
        '': 'src',
    },
    "package_data": {
        "rapidfuzz": ["*.pyi", "py.typed"],
        "rapidfuzz.distance": ["*.pyi"]
    },
    "python_requires": ">=3.6"
}

def run_setup(with_binary):
    if with_binary:
        from skbuild import setup
        import rapidfuzz_capi

        setup(
            **setup_args,
            cmake_args=[
                f'-DRF_CAPI_PATH:STRING={rapidfuzz_capi.get_include()}'
            ]
        )
    else:
        from setuptools import setup
        setup(**setup_args)

# when packaging only build wheels which include the C extension
packaging = "1" in {
    os.environ.get("CIBUILDWHEEL", "0"),
    os.environ.get("CONDA_BUILD", "0"),
    os.environ.get("RAPIDFUZZ_BUILD_EXTENSION", "0")
}
if packaging:
    run_setup(True)
else:
    try:
        run_setup(True)
    except:
        show_message(
            "WARNING: The C extension could not be compiled, speedups"
            " are not enabled.",
            "Failure information, if any, is above.",
            "Retrying the build without the C extension now.",
        )
        run_setup(False)
        show_message(
            "WARNING: The C extension could not be compiled, speedups"
            " are not enabled.",
            "Plain-Python build succeeded.",
        )
