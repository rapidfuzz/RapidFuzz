#pragma once
#include "Python.h"
#define RAPIDFUZZ_PYTHON
#include <rapidfuzz/fuzz.hpp>
#include <rapidfuzz/utils.hpp>
#include <rapidfuzz/string_metric.hpp>
#include <exception>

#define PYTHON_VERSION(major, minor, micro) ((major << 24) | (minor << 16) | (micro << 8))

namespace string_metric = rapidfuzz::string_metric;
namespace fuzz = rapidfuzz::fuzz;
namespace utils = rapidfuzz::utils;

class PythonTypeError: public std::bad_typeid {
public:

    PythonTypeError(char const* error)
      : m_error(error) {}

    virtual char const* what() const noexcept {
        return m_error;
    }
private:
    char const* m_error;
};

struct proc_string {
    int kind;
    void* data;
    size_t length;
};

// this has to be separate from the string conversion, since it can not be called without
// the GIL
static inline void validate_string(PyObject* py_str, const char* err)
{
    if (!PyUnicode_Check(py_str)) {
        throw PythonTypeError(err);
    }

    // PEP 623 deprecates legacy strings and therefor
    // deprecates e.g. PyUnicode_READY in Python 3.10
#if PY_VERSION_HEX < PYTHON_VERSION(3, 10, 0)
    if (PyUnicode_READY(py_str)) {
      // cython will use the exception set by PyUnicode_READY
      throw std::runtime_error("");
    }
#endif
}

// Right now this can be called without the GIL, since the used Python API
// is implemented using macros, which directly access the PyObject both in
// CPython and PyPy. If this changes the multiprocessing module needs to be updated
static inline proc_string convert_string(PyObject* py_str)
{
    return {
        // see https://bugs.python.org/issue43565
        static_cast<int>(PyUnicode_KIND(py_str)),
        PyUnicode_DATA(py_str),
        static_cast<std::size_t>(PyUnicode_GET_LENGTH(py_str))
    };
}

