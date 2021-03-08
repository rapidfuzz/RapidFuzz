#include "Python.h"
#include <rapidfuzz/utils.hpp>
#include <exception>

#define PYTHON_VERSION(major, minor, micro) ((major << 24) | (minor << 16) | (micro << 8))

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

PyObject* default_process_impl(PyObject* sentence) {
    if (!PyUnicode_Check(sentence)) {
        throw PythonTypeError("sentence must be a String");
    }

    // PEP 623 deprecates legacy strings and therefor
    // deprecates e.g. PyUnicode_READY in Python 3.10
#if PY_VERSION_HEX < PYTHON_VERSION(3, 10, 0)
    if (PyUnicode_READY(sentence)) {
        Py_RETURN_NONE; // unitialized, but cython directly raises an exception anyways
    }
#endif

    Py_ssize_t len = PyUnicode_GET_LENGTH(sentence);
    void* str = PyUnicode_DATA(sentence);

    switch (PyUnicode_KIND(sentence)) {
    case PyUnicode_1BYTE_KIND:
    {
        auto proc_str = utils::default_process(
            rapidfuzz::basic_string_view<uint8_t>(static_cast<uint8_t*>(str), len));
        return PyUnicode_FromKindAndData(PyUnicode_1BYTE_KIND, proc_str.data(), proc_str.size());
    }
    case PyUnicode_2BYTE_KIND:
    {
        auto proc_str = utils::default_process(
            rapidfuzz::basic_string_view<uint16_t>(static_cast<uint16_t*>(str), len));
        return PyUnicode_FromKindAndData(PyUnicode_2BYTE_KIND, proc_str.data(), proc_str.size());
    }
    default:
    {
        auto proc_str = utils::default_process(
            rapidfuzz::basic_string_view<uint32_t>(static_cast<uint32_t*>(str), len));
        return PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, proc_str.data(), proc_str.size());
    }
    }
}
