import io
from skbuild import setup

with io.open('README.md', 'rt', encoding="utf8") as f:
    readme = f.read()

setup(
    name="rapidfuzz",
    version="1.8.2",
    url="https://github.com/maxbachmann/RapidFuzz",
    author="Max Bachmann",
    author_email="contact@maxbachmann.de",
    description="rapid fuzzy string matching",
    long_description=readme,
    long_description_content_type="text/markdown",

    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License"
    ],

    packages=["rapidfuzz"],
    package_dir={'':'src'},
    zip_safe=True,
    include_package_data=True
)
