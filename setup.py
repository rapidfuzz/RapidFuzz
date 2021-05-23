from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc', '/O2', '/std:c++11', '/W4', 
        # disable some warnings from the Cython code
            #'/wd4127', # conditional expression is constant
            #'/wd4100', # '__pyx_self': unreferenced formal parameter
            #'/wd4505', # unreferenced local function has been removed
            #'/wd4125', # decimal digit terminates octal escape sequence
            #'/wd4310', # cast truncates constant value
        ],
        'unix': ['-O3', '-std=c++11',
            '-Wextra', '-Wall', '-Wconversion', '-g0',
            #'-Wno-deprecated-declarations',
            # the xcode implementation used in the CI has a bug, which causes
            # this to be thrown even when it is ignored using brackets around the statement
            #'-Wno-unreachable-code',
            # this caused issues on the conda forge build
            #'-Wno-unused-command-line-argument'
            ],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

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
        include_dirs=["src/rapidfuzz-cpp/"],
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
