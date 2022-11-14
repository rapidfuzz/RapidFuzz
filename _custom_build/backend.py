import platform as _platform
import subprocess as _subprocess

from packaging import version as _version
from packaging.tags import sys_tags as _sys_tags
from setuptools import build_meta as _orig
from skbuild.cmaker import get_cmake_version as _get_cmake_version
from skbuild.exceptions import SKBuildError as _SKBuildError

prepare_metadata_for_build_wheel = _orig.prepare_metadata_for_build_wheel
build_wheel = _orig.build_wheel
build_sdist = _orig.build_sdist
get_requires_for_build_sdist = _orig.get_requires_for_build_sdist

cmake_wheels = {
    "win_amd64",
    "win32",
    "musllinux_1_1_x86_64",
    "musllinux_1_1_s390x",
    "musllinux_1_1_ppc64le",
    "musllinux_1_1_i686",
    "musllinux_1_1_aarch64",
    "manylinux_2_17_s390x",
    "manylinux_2_17_ppc64le",
    "manylinux_2_17_aarch64",
    "manylinux_2_17_x86_64",
    "manylinux_2_17_i686",
    "manylinux_2_5_x86_64",
    "manylinux_2_5_i686",
    "macosx_10_10_universal2",
}

ninja_wheels = {
    "win_amd64",
    "win32",
    "musllinux_1_1_x86_64",
    "musllinux_1_1_s390x",
    "musllinux_1_1_ppc64le",
    "musllinux_1_1_i686",
    "musllinux_1_1_aarch64",
    "manylinux_2_17_s390x",
    "manylinux_2_17_ppc64le",
    "manylinux_2_17_aarch64",
    "manylinux_2_5_x86_64",
    "manylinux_2_5_i686",
    "macosx_10_9_universal2",
}


def _cmake_required():
    try:
        if _version.parse(_get_cmake_version()) >= _version.parse("3.12"):
            print("Using System version of cmake")
            return False
    except _SKBuildError:
        pass

    for tag in _sys_tags():
        if tag.platform in cmake_wheels:
            return True

    print("No cmake wheel available on platform")
    return False


def _ninja_required():
    if _platform.system() == "Windows":
        print("Ninja is part of the MSVC installation on Windows")
        return False

    for generator in ("ninja", "make"):
        try:
            _subprocess.check_output([generator, "--version"])
            print(f"Using System version of {generator}")
            return False
        except (OSError, _subprocess.CalledProcessError):
            pass

    for tag in _sys_tags():
        if tag.platform in ninja_wheels:
            return True

    print("No Ninja wheel available on platform")
    return False


def get_requires_for_build_wheel(self, config_settings=None):
    packages = []
    if _cmake_required():
        packages.append("cmake")
    if _ninja_required():
        packages.append("ninja")

    return _orig.get_requires_for_build_wheel(config_settings) + packages
