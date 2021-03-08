#include "Python.h"
#include <rapidfuzz/string_metric.hpp>
#include <rapidfuzz/utils.hpp>
#include <exception>

#define PYTHON_VERSION(major, minor, micro) ((major << 24) | (minor << 16) | (micro << 8))

namespace utils = rapidfuzz::utils;
namespace string_metric = rapidfuzz::string_metric;

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

static proc_string convert_string(PyObject* py_str)
{
    proc_string str = {0, NULL, 0};

    if (!PyUnicode_Check(py_str)) {
        throw PythonTypeError("choice must be a String or None");
    }

    // PEP 623 deprecates legacy strings and therefor
    // deprecates e.g. PyUnicode_READY in Python 3.10
#if PY_VERSION_HEX < PYTHON_VERSION(3, 10, 0)
    if (PyUnicode_READY(py_str)) {
      return str; // unitialized, but cython directly raises an exception anyways
    }
#endif

    str.kind = PyUnicode_KIND(py_str);
    str.data = PyUnicode_DATA(py_str);
    str.length = PyUnicode_GET_LENGTH(py_str);

    return str;
}

/*
 * Levenshtein
 */

template<typename CharT>
size_t levenshtein_impl_inner(proc_string s1, proc_string s2,
    size_t insertion, size_t deletion, size_t substitution, size_t max)
{
    switch(s2.kind){
    case PyUnicode_1BYTE_KIND:
        return string_metric::levenshtein(
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),
            rapidfuzz::basic_string_view<uint8_t>((uint8_t*)s2.data, s2.length),
            {insertion, deletion, substitution}, max
        );
    case PyUnicode_2BYTE_KIND:
        return string_metric::levenshtein(
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),
            rapidfuzz::basic_string_view<uint16_t>((uint16_t*)s2.data, s2.length),
            {insertion, deletion, substitution}, max
        );
    default:
        return string_metric::levenshtein(
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),
            rapidfuzz::basic_string_view<uint32_t>((uint32_t*)s2.data, s2.length),
            {insertion, deletion, substitution}, max
        );
    }
}

PyObject* levenshtein_impl(PyObject* s1, PyObject* s2,
    size_t insertion, size_t deletion, size_t substitution, size_t max)
{
    size_t result = 0;
    proc_string c_s1 = convert_string(s1);
    if (c_s1.data == NULL) Py_RETURN_NONE;

    proc_string c_s2 = convert_string(s2);
    if (c_s2.data == NULL) Py_RETURN_NONE;

    switch(c_s1.kind){
    case PyUnicode_1BYTE_KIND:
        result = levenshtein_impl_inner<uint8_t>(
            c_s1, c_s2, insertion, deletion, substitution, max);
        break;
    case PyUnicode_2BYTE_KIND:
        result = levenshtein_impl_inner<uint16_t>(
            c_s1, c_s2, insertion, deletion, substitution, max);
        break;
    default:
        result = levenshtein_impl_inner<uint32_t>(
            c_s1, c_s2, insertion, deletion, substitution, max);
        break;
    }

    if (result == (std::size_t)-1) {
        return PyLong_FromLong(-1);
    }
    return PyLong_FromSize_t(result);
}


/*
 *  Normalized Levenshtein
 */

template<typename CharT>
inline double normalized_levenshtein_impl_inner(proc_string s1, proc_string s2,
    size_t insertion, size_t deletion, size_t substitution, double score_cutoff)
{
    switch(s2.kind){
    case PyUnicode_1BYTE_KIND:
        return string_metric::normalized_levenshtein(
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),
            rapidfuzz::basic_string_view<uint8_t>((uint8_t*)s2.data, s2.length),
            {insertion, deletion, substitution}, score_cutoff
        );
    case PyUnicode_2BYTE_KIND:
        return string_metric::normalized_levenshtein(
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),
            rapidfuzz::basic_string_view<uint16_t>((uint16_t*)s2.data, s2.length),
            {insertion, deletion, substitution}, score_cutoff
        );
    default:
        return string_metric::normalized_levenshtein(
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),
            rapidfuzz::basic_string_view<uint32_t>((uint32_t*)s2.data, s2.length),
            {insertion, deletion, substitution}, score_cutoff
        );
    }
}

double normalized_levenshtein_impl(PyObject* s1, PyObject* s2,
    size_t insertion, size_t deletion, size_t substitution, double score_cutoff)
{
    proc_string c_s1 = convert_string(s1);
    if (c_s1.data == NULL) return 0.0;

    proc_string c_s2 = convert_string(s2);
    if (c_s2.data == NULL) return 0.0;

    switch(c_s1.kind){
    case PyUnicode_1BYTE_KIND:
        return normalized_levenshtein_impl_inner<uint8_t>(
            c_s1, c_s2, insertion, deletion, substitution, score_cutoff);
    case PyUnicode_2BYTE_KIND:
        return normalized_levenshtein_impl_inner<uint16_t>(
            c_s1, c_s2, insertion, deletion, substitution, score_cutoff);
    default:
        return normalized_levenshtein_impl_inner<uint32_t>(
            c_s1, c_s2, insertion, deletion, substitution, score_cutoff);
    }
}


template<typename CharT>
inline double normalized_levenshtein_impl_inner_default_process(proc_string s1, proc_string s2,
    size_t insertion, size_t deletion, size_t substitution, double score_cutoff)
{
    switch(s2.kind){
    case PyUnicode_1BYTE_KIND:
        return string_metric::levenshtein(
            utils::default_process(
                rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length)
            ),
            rapidfuzz::basic_string_view<uint8_t>((uint8_t*)s2.data, s2.length),
            {insertion, deletion, substitution}, score_cutoff
        );
    case PyUnicode_2BYTE_KIND:
        return string_metric::levenshtein(
            utils::default_process(
                rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length)
            ),
            utils::default_process(
                rapidfuzz::basic_string_view<uint16_t>((uint16_t*)s1.data, s1.length)
            ),
            {insertion, deletion, substitution}, score_cutoff
        );
    default:
        return string_metric::levenshtein(
            utils::default_process(
                rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length)
            ),
            utils::default_process(
                rapidfuzz::basic_string_view<uint32_t>((uint32_t*)s1.data, s1.length)
            ),
            {insertion, deletion, substitution}, score_cutoff
        );
    }
}

double normalized_levenshtein_impl_default_process(PyObject* s1, PyObject* s2,
    size_t insertion, size_t deletion, size_t substitution, double score_cutoff)
{
    proc_string c_s1 = convert_string(s1);
    if (c_s1.data == NULL) return 0.0;

    proc_string c_s2 = convert_string(s2);
    if (c_s2.data == NULL) return 0.0;

    switch(c_s1.kind){
    case PyUnicode_1BYTE_KIND:
        return normalized_levenshtein_impl_inner_default_process<uint8_t>(
            c_s1, c_s2, insertion, deletion, substitution, score_cutoff);
    case PyUnicode_2BYTE_KIND:
        return normalized_levenshtein_impl_inner_default_process<uint16_t>(
            c_s1, c_s2, insertion, deletion, substitution, score_cutoff);
    default:
        return normalized_levenshtein_impl_inner_default_process<uint32_t>(
            c_s1, c_s2, insertion, deletion, substitution, score_cutoff);
    }
}

/*
 * Hamming
 */

template<typename CharT>
size_t hamming_impl_inner(proc_string s1, proc_string s2, size_t max)
{
    switch(s2.kind){
    case PyUnicode_1BYTE_KIND:
        return string_metric::hamming(
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),
            rapidfuzz::basic_string_view<uint8_t>((uint8_t*)s2.data, s2.length),
            max
        );
    case PyUnicode_2BYTE_KIND:
        return string_metric::hamming(
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),
            rapidfuzz::basic_string_view<uint16_t>((uint16_t*)s2.data, s2.length),
            max
        );
    default:
        return string_metric::hamming(
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),
            rapidfuzz::basic_string_view<uint32_t>((uint32_t*)s2.data, s2.length),
            max
        );
    }
}

PyObject* hamming_impl(PyObject* s1, PyObject* s2, size_t max)
{
    size_t result = 0;
    proc_string c_s1 = convert_string(s1);
    if (c_s1.data == NULL) Py_RETURN_NONE;

    proc_string c_s2 = convert_string(s2);
    if (c_s2.data == NULL) Py_RETURN_NONE;

    switch(c_s1.kind){
    case PyUnicode_1BYTE_KIND:
        result = hamming_impl_inner<uint8_t>(c_s1, c_s2, max);
        break;
    case PyUnicode_2BYTE_KIND:
        result = hamming_impl_inner<uint16_t>(c_s1, c_s2, max);
        break;
    default:
        result = hamming_impl_inner<uint32_t>(c_s1, c_s2, max);
        break;
    }

    if (result == (std::size_t)-1) {
        return PyLong_FromLong(-1);
    }
    return PyLong_FromSize_t(result);
}


/*
 * Normalized Hamming
 */

template<typename CharT>
inline double normalized_hamming_impl_inner(proc_string s1, proc_string s2, double score_cutoff)
{
    switch(s2.kind){
    case PyUnicode_1BYTE_KIND:
        return string_metric::normalized_hamming(
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),
            rapidfuzz::basic_string_view<uint8_t>((uint8_t*)s2.data, s2.length),
            score_cutoff
        );
    case PyUnicode_2BYTE_KIND:
        return string_metric::normalized_hamming(
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),
            rapidfuzz::basic_string_view<uint16_t>((uint16_t*)s2.data, s2.length),
            score_cutoff
        );
    default:
        return string_metric::normalized_hamming(
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),
            rapidfuzz::basic_string_view<uint32_t>((uint32_t*)s2.data, s2.length),
            score_cutoff
        );
    }
}

double normalized_hamming_impl(PyObject* s1, PyObject* s2, double score_cutoff)
{
    proc_string c_s1 = convert_string(s1);
    if (c_s1.data == NULL) return 0.0;

    proc_string c_s2 = convert_string(s2);
    if (c_s2.data == NULL) return 0.0;

    switch(c_s1.kind){
    case PyUnicode_1BYTE_KIND:
        return normalized_hamming_impl_inner<uint8_t>(c_s1, c_s2, score_cutoff);
    case PyUnicode_2BYTE_KIND:
        return normalized_hamming_impl_inner<uint16_t>(c_s1, c_s2, score_cutoff);
    default:
        return normalized_hamming_impl_inner<uint32_t>(c_s1, c_s2, score_cutoff);
    }
}

template<typename CharT>
inline double normalized_hamming_impl_inner_default_process(
    proc_string s1, proc_string s2, double score_cutoff)
{
    switch(s2.kind){
    case PyUnicode_1BYTE_KIND:
        return string_metric::hamming(
            utils::default_process(
                rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length)
            ),
            rapidfuzz::basic_string_view<uint8_t>((uint8_t*)s2.data, s2.length),
            score_cutoff
        );
    case PyUnicode_2BYTE_KIND:
        return string_metric::hamming(
            utils::default_process(
                rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length)
            ),
            utils::default_process(
                rapidfuzz::basic_string_view<uint16_t>((uint16_t*)s1.data, s1.length)
            ),
            score_cutoff
        );
    default:
        return string_metric::hamming(
            utils::default_process(
                rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length)
            ),
            utils::default_process(
                rapidfuzz::basic_string_view<uint32_t>((uint32_t*)s1.data, s1.length)
            ),
            score_cutoff
        );
    }
}

double normalized_hamming_impl_default_process(PyObject* s1, PyObject* s2, double score_cutoff)
{
    proc_string c_s1 = convert_string(s1);
    if (c_s1.data == NULL) return 0.0;

    proc_string c_s2 = convert_string(s2);
    if (c_s2.data == NULL) return 0.0;

    switch(c_s1.kind){
    case PyUnicode_1BYTE_KIND:
        return normalized_hamming_impl_inner_default_process<uint8_t>(c_s1, c_s2, score_cutoff);
    case PyUnicode_2BYTE_KIND:
        return normalized_hamming_impl_inner_default_process<uint16_t>(c_s1, c_s2, score_cutoff);
    default:
        return normalized_hamming_impl_inner_default_process<uint32_t>(c_s1, c_s2, score_cutoff);
    }
}
