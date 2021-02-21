#include "Python.h"
#include <rapidfuzz/fuzz.hpp>
#include <rapidfuzz/utils.hpp>
#include <rapidfuzz/string_metric.hpp>
#include <exception>

#define PYTHON_VERSION(major, minor, micro) ((major << 24) | (minor << 16) | (micro << 8))

namespace string_metric = rapidfuzz::string_metric;
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

struct ListMatchElem {
    double score;
    size_t index;
};

struct DictMatchElem {
    double score;
    size_t index;
    PyObject* choice;
    PyObject* key;
};

void dummy() {

}

struct ExtractComp
{
    template<class T>
    bool operator()(T const &a, T const &b) const {
        if (a.score > b.score) {
            return true;
        } else if (a.score < b.score) {
            return false;
        } else {
            return a.index < b.index;
        }
    }
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
      return str;
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



typedef double (*scorer_func) (void* context, PyObject* str, double score_cutoff);
typedef void (*scorer_context_deinit) (void* context);

struct scorer_context {
    void* context;
    scorer_func scorer;
    scorer_context_deinit deinit;
};


#define CACHED_RATIO_DEINIT(RATIO, CACHED_SCORER)                                 \
template <typename CharT>                                                         \
static void cached_##RATIO##_deinit(void* context)                                \
{                                                                                 \
    delete (CACHED_SCORER<rapidfuzz::basic_string_view<CharT>>*)context;          \
}

#if PY_VERSION_HEX >= PYTHON_VERSION(3, 0, 0)
#define CACHED_RATIO_FUNC_DEFAULT_PROCESS(RATIO, CACHED_SCORER)                   \
template<typename CharT>                                                          \
static double cached_##RATIO##_func_default_process(                              \
    void* context, PyObject* py_str, double score_cutoff)                         \
{                                                                                 \
    proc_string str = convert_string(py_str);                                     \
    if (str.data == NULL) return 0.0;                                             \
                                                                                  \
    auto* ratio = (CACHED_SCORER<rapidfuzz::basic_string_view<CharT>>*)context;   \
                                                                                  \
    switch(str.kind){                                                             \
    case PyUnicode_1BYTE_KIND:                                                    \
        return ratio->ratio(                                                      \
            rapidfuzz::utils::default_process((uint8_t*)str.data, str.length),    \
            score_cutoff                                                          \
        );                                                                        \
    case PyUnicode_2BYTE_KIND:                                                    \
        return ratio->ratio(                                                      \
            rapidfuzz::utils::default_process((uint16_t*)str.data, str.length),   \
            score_cutoff                                                          \
        );                                                                        \
    default:                                                                      \
        return ratio->ratio(                                                      \
            rapidfuzz::utils::default_process((uint32_t*)str.data, str.length),   \
            score_cutoff                                                          \
        );                                                                        \
    }                                                                             \
}
#else
#define CACHED_RATIO_FUNC_DEFAULT_PROCESS(RATIO, CACHED_SCORER)                   \
template<typename CharT>                                                          \
static double cached_##RATIO##_func_default_process(                              \
    void* context, PyObject* py_str, double score_cutoff)                         \
{                                                                                 \
    proc_string str = convert_string(py_str);                                     \
    if (str.data == NULL) return 0.0;                                             \
                                                                                  \
    auto* ratio = (CACHED_SCORER<rapidfuzz::basic_string_view<CharT>>*)context;   \
                                                                                  \
    switch(str.kind){                                                             \
    case CHAR_STRING:                                                             \
        return ratio->ratio(                                                      \
            rapidfuzz::utils::default_process((uint8_t*)str.data, str.length),    \
            score_cutoff                                                          \
        );                                                                        \
    default:                                                                      \
        return ratio->ratio(                                                      \
            rapidfuzz::utils::default_process((Py_UNICODE*)str.data, str.length), \
            score_cutoff                                                          \
        );                                                                        \
    }                                                                             \
}
#endif

#if PY_VERSION_HEX >= PYTHON_VERSION(3, 0, 0)
#define CACHED_RATIO_FUNC(RATIO, CACHED_SCORER)                                      \
template<typename CharT>                                                             \
static double cached_##RATIO##_func(                                                 \
    void* context, PyObject* py_str, double score_cutoff)                            \
{                                                                                    \
    proc_string str = convert_string(py_str);                                        \
    if (str.data == NULL) return 0.0;                                                \
                                                                                     \
    auto* ratio = (CACHED_SCORER<rapidfuzz::basic_string_view<CharT>>*)context;      \
                                                                                     \
    switch(str.kind){                                                                \
    case PyUnicode_1BYTE_KIND:                                                       \
        return ratio->ratio(                                                         \
            rapidfuzz::basic_string_view<uint8_t>((uint8_t*)str.data, str.length),   \
            score_cutoff                                                             \
        );                                                                           \
    case PyUnicode_2BYTE_KIND:                                                       \
        return ratio->ratio(                                                         \
            rapidfuzz::basic_string_view<uint16_t>((uint16_t*)str.data, str.length), \
            score_cutoff                                                             \
        );                                                                           \
    default:                                                                         \
        return ratio->ratio(                                                         \
            rapidfuzz::basic_string_view<uint32_t>((uint32_t*)str.data, str.length), \
            score_cutoff                                                             \
        );                                                                           \
    }                                                                                \
}
#else
#define CACHED_RATIO_FUNC(RATIO, CACHED_SCORER)                                          \
template<typename CharT>                                                                 \
static double cached_##RATIO##_func(                                                     \
    void* context, PyObject* py_str, double score_cutoff)                                \
{                                                                                        \
    proc_string str = convert_string(py_str);                                            \
    if (str.data == NULL) return 0.0;                                                    \
                                                                                         \
    auto* ratio = (CACHED_SCORER<rapidfuzz::basic_string_view<CharT>>*)context;          \
                                                                                         \
    switch(str.kind){                                                                    \
    case CHAR_STRING:                                                                    \
        return ratio->ratio(                                                             \
            rapidfuzz::basic_string_view<uint8_t>((uint8_t*)str.data, str.length),       \
            score_cutoff                                                                 \
        );                                                                               \
    default:                                                                             \
        return ratio->ratio(                                                             \
            rapidfuzz::basic_string_view<Py_UNICODE>((Py_UNICODE*)str.data, str.length), \
            score_cutoff                                                                 \
        );                                                                               \
    }                                                                                    \
}
#endif


#if PY_VERSION_HEX >= PYTHON_VERSION(3, 0, 0)
#define CACHED_RATIO_INIT(RATIO, CACHED_SCORER)                                              \
static scorer_context cached_##RATIO##_init(PyObject* py_str, int def_process)               \
{                                                                                            \
    scorer_context context;                                                                  \
    proc_string str = convert_string(py_str);                                                \
    if (str.data == NULL) return context;                                                    \
                                                                                             \
    switch(str.kind){                                                                        \
    case PyUnicode_1BYTE_KIND:                                                               \
        context.context = (void*) new CACHED_SCORER<rapidfuzz::basic_string_view<uint8_t>>(  \
            rapidfuzz::basic_string_view<uint8_t>((uint8_t*)str.data, str.length)            \
        );                                                                                   \
                                                                                             \
        if (def_process) {                                                                   \
            context.scorer = cached_##RATIO##_func_default_process<uint8_t>;                 \
        } else {                                                                             \
            context.scorer = cached_##RATIO##_func<uint8_t>;                                 \
        }                                                                                    \
        context.deinit = cached_##RATIO##_deinit<uint8_t>;                                   \
        break;                                                                               \
    case PyUnicode_2BYTE_KIND:                                                               \
        context.context = (void*) new CACHED_SCORER<rapidfuzz::basic_string_view<uint16_t>>( \
            rapidfuzz::basic_string_view<uint16_t>((uint16_t*)str.data, str.length)          \
        );                                                                                   \
                                                                                             \
        if (def_process) {                                                                   \
            context.scorer = cached_##RATIO##_func_default_process<uint16_t>;                \
        } else {                                                                             \
            context.scorer = cached_##RATIO##_func<uint16_t>;                                \
        }                                                                                    \
        context.deinit = cached_##RATIO##_deinit<uint16_t>;                                  \
        break;                                                                               \
    default:                                                                                 \
        context.context = (void*) new CACHED_SCORER<rapidfuzz::basic_string_view<uint32_t>>( \
            rapidfuzz::basic_string_view<uint32_t>((uint32_t*)str.data, str.length)          \
        );                                                                                   \
                                                                                             \
        if (def_process) {                                                                   \
            context.scorer = cached_##RATIO##_func_default_process<uint32_t>;                \
        } else {                                                                             \
            context.scorer = cached_##RATIO##_func<uint32_t>;                                \
        }                                                                                    \
        context.deinit = cached_##RATIO##_deinit<uint32_t>;                                  \
        break;                                                                               \
    }                                                                                        \
                                                                                             \
    return context;                                                                          \
}
#else
#define CACHED_RATIO_INIT(RATIO, CACHED_SCORER)                                                \
static scorer_context cached_##RATIO##_init(PyObject* py_str, int def_process)                 \
{                                                                                              \
    scorer_context context;                                                                    \
    proc_string str = convert_string(py_str);                                                  \
    if (str.data == NULL) return context;                                                      \
                                                                                               \
                                                                                               \
    switch(str.kind){                                                                          \
    case CHAR_STRING:                                                                          \
        context.context = (void*) new CACHED_SCORER<rapidfuzz::basic_string_view<uint8_t>>(    \
            rapidfuzz::basic_string_view<uint8_t>((uint8_t*)str.data, str.length)              \
        );                                                                                     \
                                                                                               \
        if (def_process) {                                                                     \
            context.scorer = cached_##RATIO##_func_default_process<uint8_t>;                   \
        } else {                                                                               \
            context.scorer = cached_##RATIO##_func<uint8_t>;                                   \
        }                                                                                      \
        context.deinit = cached_##RATIO##_deinit<uint8_t>;                                     \
        break;                                                                                 \
    default:                                                                                   \
        context.context = (void*) new CACHED_SCORER<rapidfuzz::basic_string_view<Py_UNICODE>>( \
            rapidfuzz::basic_string_view<Py_UNICODE>((Py_UNICODE*)str.data, str.length)        \
        );                                                                                     \
                                                                                               \
        if (def_process) {                                                                     \
            context.scorer = cached_##RATIO##_func_default_process<Py_UNICODE>;                \
        } else {                                                                               \
            context.scorer = cached_##RATIO##_func<Py_UNICODE>;                                \
        }                                                                                      \
        context.deinit = cached_##RATIO##_deinit<Py_UNICODE>;                                  \
        break;                                                                                 \
    }                                                                                          \
                                                                                               \
    return context;                                                                            \
}

#endif

#define CACHED_RATIO(RATIO, CACHED_SCORER)              \
CACHED_RATIO_DEINIT(RATIO, CACHED_SCORER)               \
CACHED_RATIO_FUNC_DEFAULT_PROCESS(RATIO, CACHED_SCORER) \
CACHED_RATIO_FUNC(RATIO, CACHED_SCORER)                 \
CACHED_RATIO_INIT(RATIO, CACHED_SCORER)

/* fuzz */
CACHED_RATIO(ratio,                    fuzz::CachedRatio)
CACHED_RATIO(partial_ratio,            fuzz::CachedPartialRatio)
CACHED_RATIO(token_sort_ratio,         fuzz::CachedTokenSortRatio)
CACHED_RATIO(token_set_ratio,          fuzz::CachedTokenSetRatio)
CACHED_RATIO(token_ratio,              fuzz::CachedTokenRatio)
CACHED_RATIO(partial_token_sort_ratio, fuzz::CachedPartialTokenSortRatio)
CACHED_RATIO(partial_token_set_ratio,  fuzz::CachedPartialTokenSetRatio)
CACHED_RATIO(partial_token_ratio,      fuzz::CachedPartialTokenRatio)
CACHED_RATIO(WRatio,                   fuzz::CachedWRatio)
CACHED_RATIO(QRatio,                   fuzz::CachedQRatio)
/* string_metric */
CACHED_RATIO(normalized_hamming,       string_metric::CachedNormalizedHamming)