from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import numpy as np

# use with export RAPIDFUZZ_TRACE=1
RAPIDFUZZ_TRACE = os.environ.get("RAPIDFUZZ_TRACE", False)

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc', '/O2', '/W4', '/DNDEBUG'],
        'unix': ['-O3', '-std=c++11', '-Wextra', '-Wall', '-Wconversion', '-g0', '-DNDEBUG'],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if RAPIDFUZZ_TRACE:
        c_opts['msvc'].append("/DCYTHON_TRACE_NOGIL=1")
        c_opts['unix'].append("-DCYTHON_TRACE_NOGIL=1")

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.9']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args += opts
            ext.extra_link_args += link_opts
        build_ext.build_extensions(self)

ext_modules = [
    Extension(
       name='rapidfuzz.cpp_process',
        sources=[
            'src/cpp_process.cpp',
            'src/rapidfuzz-cpp/rapidfuzz/details/unicode.cpp'
        ],
        include_dirs=["src/rapidfuzz-cpp/", np.get_include()],
        language='c++',
    ),
    Extension(
        name='rapidfuzz.cpp_fuzz',
        sources=[
            'src/cpp_fuzz.cpp',
            'src/rapidfuzz-cpp/rapidfuzz/details/unicode.cpp'
        ],
        include_dirs=["src/rapidfuzz-cpp/"],
        language='c++',
    ),
    Extension(
        name='rapidfuzz.cpp_string_metric',
        sources=[
            'src/cpp_string_metric.cpp',
            'src/rapidfuzz-cpp/rapidfuzz/details/unicode.cpp'
        ],
        include_dirs=["src/rapidfuzz-cpp/"],
        language='c++',
    ),
    Extension(
        name='rapidfuzz.cpp_utils',
        sources=[
            'src/cpp_utils.cpp',
            'src/rapidfuzz-cpp/rapidfuzz/details/unicode.cpp'
        ],
        include_dirs=["src/rapidfuzz-cpp/"],
        language='c++',
    )
]


if __name__ == "__main__":
    setup(
        cmdclass={'build_ext': BuildExt},
        ext_modules = ext_modules
    )
