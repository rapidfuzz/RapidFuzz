from skbuild import setup
import rapidfuzz_capi

with open('README.md', 'rt', encoding="utf8") as f:
    readme = f.read()

setup(
    name="rapidfuzz",
    version="2.0.5",
    extras_require={'full': ['numpy']},
    url="https://github.com/maxbachmann/RapidFuzz",
    author="Max Bachmann",
    author_email="contact@maxbachmann.de",
    description="rapid fuzzy string matching",
    long_description=readme,
    long_description_content_type="text/markdown",

    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License"
    ],

    packages=["rapidfuzz", "rapidfuzz/distance"],
    package_dir={'':'src'},
    zip_safe=True,
    include_package_data=True,
    python_requires=">=3.6",

    cmake_args=[f'-DRF_CAPI_PATH:STRING={rapidfuzz_capi.get_include()}']
)
