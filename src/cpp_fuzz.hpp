#include "Python.h"
#include <rapidfuzz/fuzz.hpp>
#include <rapidfuzz/utils.hpp>
#include <exception>

#define PYTHON_VERSION(major, minor, micro) ((major << 24) | (minor << 16) | (micro << 8))

namespace utils = rapidfuzz::utils;
namespace fuzz = rapidfuzz::fuzz;

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

#if PY_VERSION_HEX < PYTHON_VERSION(3, 0, 0)
enum StringKind{
    CHAR_STRING,
    UNICODE_STRING
};
#endif

struct proc_string {
    int kind;
    void* data;
    size_t length;
};

static proc_string convert_string(PyObject* py_str)
{
    proc_string str = {0, NULL, 0};

#if PY_VERSION_HEX >= PYTHON_VERSION(3, 0, 0)
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
#else /* Python 2 */
    if (PyObject_TypeCheck(py_str, &PyString_Type)) {
        str.kind = CHAR_STRING;
        str.length = PyString_GET_SIZE(py_str);
        str.data = (void*)PyString_AS_STRING(py_str);
    }
    else if (PyObject_TypeCheck(py_str, &PyUnicode_Type)) {
        str.kind = UNICODE_STRING;
        str.length = PyUnicode_GET_SIZE(py_str);
        str.data = (void*)PyUnicode_AS_UNICODE(py_str);
    }
    else {
        throw PythonTypeError("choice must be a String, Unicode or None");
    }
#endif

    return str;
}

#if PY_VERSION_HEX >= PYTHON_VERSION(3, 0, 0)
#define RATIO_SINGLE(RATIO, RATIO_FUNC)                                            \
template<typename CharT>                                                           \
inline double RATIO##_single(proc_string s1, proc_string s2, double score_cutoff)  \
{                                                                                  \
    switch(s2.kind){                                                               \
    case PyUnicode_1BYTE_KIND:                                                     \
        return RATIO_FUNC(                                                         \
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),       \
            rapidfuzz::basic_string_view<uint8_t>((uint8_t*)s2.data, s2.length),   \
            score_cutoff                                                           \
        );                                                                         \
    case PyUnicode_2BYTE_KIND:                                                     \
        return RATIO_FUNC(                                                         \
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),       \
            rapidfuzz::basic_string_view<uint16_t>((uint16_t*)s2.data, s2.length), \
            score_cutoff                                                           \
        );                                                                         \
    default:                                                                       \
        return RATIO_FUNC(                                                         \
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),       \
            rapidfuzz::basic_string_view<uint32_t>((uint32_t*)s2.data, s2.length), \
            score_cutoff                                                           \
        );                                                                         \
    }                                                                              \
}
#else
#define RATIO_SINGLE(RATIO, RATIO_FUNC)                                                \
template<typename CharT>                                                               \
inline double RATIO##_single(proc_string s1, proc_string s2, double score_cutoff)      \
{                                                                                      \
    switch(s2.kind){                                                                   \
    case CHAR_STRING:                                                                  \
        return RATIO_FUNC(                                                             \
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),           \
            rapidfuzz::basic_string_view<uint8_t>((uint8_t*)s2.data, s2.length),       \
            score_cutoff                                                               \
        );                                                                             \
    default:                                                                           \
        return RATIO_FUNC(                                                             \
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),           \
            rapidfuzz::basic_string_view<Py_UNICODE>((Py_UNICODE*)s2.data, s2.length), \
            score_cutoff                                                               \
        );                                                                             \
    }                                                                                  \
}
#endif


#if PY_VERSION_HEX >= PYTHON_VERSION(3, 0, 0)
#define RATIO_IMPL(RATIO, RATIO_FUNC)                                \
double RATIO##_impl(PyObject* s1, PyObject* s2, double score_cutoff) \
{                                                                    \
    proc_string c_s1 = convert_string(s1);                           \
    if (c_s1.data == NULL) return 0.0;                               \
                                                                     \
    proc_string c_s2 = convert_string(s2);                           \
    if (c_s2.data == NULL) return 0.0;                               \
                                                                     \
    switch(c_s1.kind){                                               \
    case PyUnicode_1BYTE_KIND:                                       \
        return RATIO##_single<uint8_t>(c_s1, c_s2, score_cutoff);    \
    case PyUnicode_2BYTE_KIND:                                       \
        return RATIO##_single<uint16_t>(c_s1, c_s2, score_cutoff);   \
    default:                                                         \
        return RATIO##_single<uint32_t>(c_s1, c_s2, score_cutoff);   \
    }                                                                \
}
#else
#define RATIO_IMPL(RATIO, RATIO_FUNC)                                \
double RATIO##_impl(PyObject* s1, PyObject* s2, double score_cutoff) \
{                                                                    \
    proc_string c_s1 = convert_string(s1);                           \
    if (c_s1.data == NULL) return 0.0;                               \
                                                                     \
    proc_string c_s2 = convert_string(s2);                           \
    if (c_s2.data == NULL) return 0.0;                               \
                                                                     \
    switch(c_s1.kind){                                               \
    case CHAR_STRING:                                                \
        return RATIO##_single<uint8_t>(c_s1, c_s2, score_cutoff);    \
    default:                                                         \
        return RATIO##_single<Py_UNICODE>(c_s1, c_s2, score_cutoff); \
    }                                                                \
}
#endif


#if PY_VERSION_HEX >= PYTHON_VERSION(3, 0, 0)
#define RATIO_SINGLE_DEFAULT_PROCESS(RATIO, RATIO_FUNC)                               \
template<typename CharT>                                                              \
inline double RATIO##_single_default_process(                                         \
    proc_string s1, proc_string s2, double score_cutoff)                              \
{                                                                                     \
    switch(s2.kind){                                                                  \
    case PyUnicode_1BYTE_KIND:                                                        \
        return RATIO_FUNC(                                                            \
            utils::default_process(                                                   \
                rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length)       \
            ),                                                                        \
            utils::default_process(                                                   \
                rapidfuzz::basic_string_view<uint8_t>((uint8_t*)s2.data, s2.length)   \
            ),                                                                        \
            score_cutoff                                                              \
        );                                                                            \
    case PyUnicode_2BYTE_KIND:                                                        \
        return RATIO_FUNC(                                                            \
            utils::default_process(                                                   \
                rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length)       \
            ),                                                                        \
            utils::default_process(                                                   \
                rapidfuzz::basic_string_view<uint16_t>((uint16_t*)s2.data, s2.length) \
            ),                                                                        \
            score_cutoff                                                              \
        );                                                                            \
    default:                                                                          \
        return RATIO_FUNC(                                                            \
            utils::default_process(                                                   \
                rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length)       \
            ),                                                                        \
            utils::default_process(                                                   \
                rapidfuzz::basic_string_view<uint32_t>((uint32_t*)s2.data, s2.length) \
            ),                                                                        \
            score_cutoff                                                              \
        );                                                                            \
    }                                                                                 \
}
#else
#define RATIO_SINGLE_DEFAULT_PROCESS(RATIO, RATIO_FUNC)                                   \
template<typename CharT>                                                                  \
inline double RATIO##_single_default_process(                                             \
    proc_string s1, proc_string s2, double score_cutoff)                                  \
{                                                                                         \
    switch(s2.kind){                                                                      \
    case CHAR_STRING:                                                                     \
        return RATIO_FUNC(                                                                \
            utils::default_process(                                                       \
                rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length)           \
            ),                                                                            \
            utils::default_process(                                                       \
                rapidfuzz::basic_string_view<uint8_t>((uint8_t*)s2.data, s2.length)       \
            ),                                                                            \
            score_cutoff                                                                  \
        );                                                                                \
    default:                                                                              \
        return RATIO_FUNC(                                                                \
            utils::default_process(                                                       \
                rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length)           \
            ),                                                                            \
            utils::default_process(                                                       \
                rapidfuzz::basic_string_view<Py_UNICODE>((Py_UNICODE*)s2.data, s2.length) \
            ),                                                                            \
            score_cutoff                                                                  \
        );                                                                                \
    }                                                                                     \
}
#endif


#if PY_VERSION_HEX >= PYTHON_VERSION(3, 0, 0)
#define RATIO_IMPL_DEFAULT_PROCESS(RATIO, RATIO_FUNC)                                   \
double RATIO##_impl_default_process(PyObject* s1, PyObject* s2, double score_cutoff) {  \
    proc_string c_s1 = convert_string(s1);                                              \
    if (c_s1.data == NULL) return 0.0;                                                  \
                                                                                        \
    proc_string c_s2 = convert_string(s2);                                              \
    if (c_s2.data == NULL) return 0.0;                                                  \
                                                                                        \
    switch(c_s1.kind){                                                                  \
    case PyUnicode_1BYTE_KIND:                                                          \
        return RATIO##_single_default_process<uint8_t>(c_s1, c_s2, score_cutoff);       \
    case PyUnicode_2BYTE_KIND:                                                          \
        return RATIO##_single_default_process<uint16_t>(c_s1, c_s2, score_cutoff);      \
    default:                                                                            \
        return RATIO##_single_default_process<uint32_t>(c_s1, c_s2, score_cutoff);      \
    }                                                                                   \
}
#else
#define RATIO_IMPL_DEFAULT_PROCESS(RATIO, RATIO_FUNC)                                   \
double RATIO##_impl_default_process(PyObject* s1, PyObject* s2, double score_cutoff) {  \
    proc_string c_s1 = convert_string(s1);                                              \
    if (c_s1.data == NULL) return 0.0;                                                  \
                                                                                        \
    proc_string c_s2 = convert_string(s2);                                              \
    if (c_s2.data == NULL) return 0.0;                                                  \
                                                                                        \
    switch(c_s1.kind){                                                                  \
    case CHAR_STRING:                                                                   \
        return RATIO##_single_default_process<uint8_t>(c_s1, c_s2, score_cutoff);       \
    default:                                                                            \
        return RATIO##_single_default_process<Py_UNICODE>(c_s1, c_s2, score_cutoff);    \
    }                                                                                   \
}
#endif

#define RATIO_DEF(RATIO, RATIO_FUNC)            \
RATIO_SINGLE_DEFAULT_PROCESS(RATIO, RATIO_FUNC) \
RATIO_IMPL_DEFAULT_PROCESS(RATIO, RATIO_FUNC)   \
RATIO_SINGLE(RATIO, RATIO_FUNC)                 \
RATIO_IMPL(RATIO, RATIO_FUNC)

RATIO_DEF(ratio,                    fuzz::ratio)
RATIO_DEF(partial_ratio,            fuzz::partial_ratio)
RATIO_DEF(token_sort_ratio,         fuzz::token_sort_ratio)
RATIO_DEF(token_set_ratio,          fuzz::token_set_ratio)
RATIO_DEF(token_ratio,              fuzz::token_ratio)
RATIO_DEF(partial_token_sort_ratio, fuzz::partial_token_sort_ratio)
RATIO_DEF(partial_token_set_ratio,  fuzz::partial_token_set_ratio)
RATIO_DEF(partial_token_ratio,      fuzz::partial_token_ratio)
RATIO_DEF(WRatio,                   fuzz::WRatio)
RATIO_DEF(QRatio,                   fuzz::QRatio)