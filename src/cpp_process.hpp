#include "Python.h"
#include "fuzz.hpp"
#include "utils.hpp"
#include <exception>

#define PYTHON_VERSION(major, minor, micro) ((major << 24) | (minor << 16) | (micro << 8))

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

static proc_string process_string(PyObject* py_str)
{
    proc_string str;

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



typedef double (*scorer_func) (void* context, PyObject* str, double score_cutoff);
typedef void (*scorer_context_deinit) (void* context);

struct scorer_context {
    void* context;
    scorer_func scorer;
    scorer_context_deinit deinit;
};


#define CACHED_RATIO_DEINIT(RATIO, CLASS)                                         \
template <typename CharT>                                                         \
static void cached_##RATIO##_deinit(void* context)                                \
{                                                                                 \
    delete (rapidfuzz::fuzz::CLASS<rapidfuzz::basic_string_view<CharT>>*)context; \
}

CACHED_RATIO_DEINIT(ratio,                    CachedRatio)
CACHED_RATIO_DEINIT(partial_ratio,            CachedPartialRatio)
CACHED_RATIO_DEINIT(token_sort_ratio,         CachedTokenSortRatio)
CACHED_RATIO_DEINIT(token_set_ratio,          CachedTokenSetRatio)
CACHED_RATIO_DEINIT(token_ratio,              CachedTokenRatio)
CACHED_RATIO_DEINIT(partial_token_sort_ratio, CachedPartialTokenSortRatio)
CACHED_RATIO_DEINIT(partial_token_set_ratio,  CachedPartialTokenSetRatio)
CACHED_RATIO_DEINIT(partial_token_ratio,      CachedPartialTokenRatio)
CACHED_RATIO_DEINIT(WRatio,                   CachedWRatio)
CACHED_RATIO_DEINIT(QRatio,                   CachedQRatio)


#if PY_VERSION_HEX >= PYTHON_VERSION(3, 0, 0)
#define CACHED_RATIO_FUNC_DEFAULT_PROCESS(RATIO, CLASS)                           \
template<typename CharT>                                                          \
static double cached_##RATIO##_func_default_process(                              \
    void* context, PyObject* py_str, double score_cutoff)                         \
{                                                                                 \
    proc_string str = process_string(py_str);                                     \
    auto* ratio =                                                                 \
      (rapidfuzz::fuzz::CLASS<rapidfuzz::basic_string_view<CharT>>*)context;      \
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
#define CACHED_RATIO_FUNC_DEFAULT_PROCESS(RATIO, CLASS)                           \
template<typename CharT>                                                          \
static double cached_##RATIO##_func_default_process(                              \
    void* context, PyObject* py_str, double score_cutoff)                         \
{                                                                                 \
    proc_string str = process_string(py_str);                                     \
    auto* ratio =                                                                 \
      (rapidfuzz::fuzz::CLASS<rapidfuzz::basic_string_view<CharT>>*)context;      \
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

CACHED_RATIO_FUNC_DEFAULT_PROCESS(ratio,                    CachedRatio)
CACHED_RATIO_FUNC_DEFAULT_PROCESS(partial_ratio,            CachedPartialRatio)
CACHED_RATIO_FUNC_DEFAULT_PROCESS(token_sort_ratio,         CachedTokenSortRatio)
CACHED_RATIO_FUNC_DEFAULT_PROCESS(token_set_ratio,          CachedTokenSetRatio)
CACHED_RATIO_FUNC_DEFAULT_PROCESS(token_ratio,              CachedTokenRatio)
CACHED_RATIO_FUNC_DEFAULT_PROCESS(partial_token_sort_ratio, CachedPartialTokenSortRatio)
CACHED_RATIO_FUNC_DEFAULT_PROCESS(partial_token_set_ratio,  CachedPartialTokenSetRatio)
CACHED_RATIO_FUNC_DEFAULT_PROCESS(partial_token_ratio,      CachedPartialTokenRatio)
CACHED_RATIO_FUNC_DEFAULT_PROCESS(WRatio,                   CachedWRatio)
CACHED_RATIO_FUNC_DEFAULT_PROCESS(QRatio,                   CachedQRatio)


#if PY_VERSION_HEX >= PYTHON_VERSION(3, 0, 0)
#define CACHED_RATIO_FUNC(RATIO, CLASS)                                              \
template<typename CharT>                                                             \
static double cached_##RATIO##_func(                                                 \
    void* context, PyObject* py_str, double score_cutoff)                            \
{                                                                                    \
    proc_string str = process_string(py_str);                                        \
    auto* ratio =                                                                    \
        (rapidfuzz::fuzz::CLASS<rapidfuzz::basic_string_view<CharT>>*)context;       \
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
#define CACHED_RATIO_FUNC(RATIO, CLASS)                                                  \
template<typename CharT>                                                                 \
static double cached_##RATIO##_func(                                                     \
    void* context, PyObject* py_str, double score_cutoff)                                \
{                                                                                        \
    proc_string str = process_string(py_str);                                            \
    auto* ratio =                                                                        \
        (rapidfuzz::fuzz::CLASS<rapidfuzz::basic_string_view<CharT>>*)context;           \
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

CACHED_RATIO_FUNC(ratio,                    CachedRatio)
CACHED_RATIO_FUNC(partial_ratio,            CachedPartialRatio)
CACHED_RATIO_FUNC(token_sort_ratio,         CachedTokenSortRatio)
CACHED_RATIO_FUNC(token_set_ratio,          CachedTokenSetRatio)
CACHED_RATIO_FUNC(token_ratio,              CachedTokenRatio)
CACHED_RATIO_FUNC(partial_token_sort_ratio, CachedPartialTokenSortRatio)
CACHED_RATIO_FUNC(partial_token_set_ratio,  CachedPartialTokenSetRatio)
CACHED_RATIO_FUNC(partial_token_ratio,      CachedPartialTokenRatio)
CACHED_RATIO_FUNC(WRatio,                   CachedWRatio)
CACHED_RATIO_FUNC(QRatio,                   CachedQRatio)


#if PY_VERSION_HEX >= PYTHON_VERSION(3, 0, 0)
#define CACHED_RATIO_INIT(RATIO, CLASS)                                                               \
static scorer_context cached_##RATIO##_init(PyObject* py_str, int def_process)                        \
{                                                                                                     \
    scorer_context context;                                                                           \
    proc_string str = process_string(py_str);                                                         \
                                                                                                      \
                                                                                                      \
    switch(str.kind){                                                                                 \
    case PyUnicode_1BYTE_KIND:                                                                        \
        context.context = (void*) new rapidfuzz::fuzz::CLASS<rapidfuzz::basic_string_view<uint8_t>>(  \
            rapidfuzz::basic_string_view<uint8_t>((uint8_t*)str.data, str.length)                     \
        );                                                                                            \
                                                                                                      \
        if (def_process) {                                                                            \
            context.scorer = cached_##RATIO##_func_default_process<uint8_t>;                          \
        } else {                                                                                      \
            context.scorer = cached_##RATIO##_func<uint8_t>;                                          \
        }                                                                                             \
        context.deinit = cached_##RATIO##_deinit<uint8_t>;                                            \
        break;                                                                                        \
    case PyUnicode_2BYTE_KIND:                                                                        \
        context.context = (void*) new rapidfuzz::fuzz::CLASS<rapidfuzz::basic_string_view<uint16_t>>( \
            rapidfuzz::basic_string_view<uint16_t>((uint16_t*)str.data, str.length)                   \
        );                                                                                            \
                                                                                                      \
        if (def_process) {                                                                            \
            context.scorer = cached_##RATIO##_func_default_process<uint16_t>;                         \
        } else {                                                                                      \
            context.scorer = cached_##RATIO##_func<uint16_t>;                                         \
        }                                                                                             \
        context.deinit = cached_##RATIO##_deinit<uint16_t>;                                           \
        break;                                                                                        \
    default:                                                                                          \
        context.context = (void*) new rapidfuzz::fuzz::CLASS<rapidfuzz::basic_string_view<uint32_t>>( \
            rapidfuzz::basic_string_view<uint32_t>((uint32_t*)str.data, str.length)                   \
        );                                                                                            \
                                                                                                      \
        if (def_process) {                                                                            \
            context.scorer = cached_##RATIO##_func_default_process<uint32_t>;                         \
        } else {                                                                                      \
            context.scorer = cached_##RATIO##_func<uint32_t>;                                         \
        }                                                                                             \
        context.deinit = cached_##RATIO##_deinit<uint32_t>;                                           \
        break;                                                                                        \
    }                                                                                                 \
                                                                                                      \
    return context;                                                                                   \
}
#else
#define CACHED_RATIO_INIT(RATIO, CLASS)                                                               \
static scorer_context cached_##RATIO##_init(PyObject* py_str, int def_process)                        \
{                                                                                                     \
    scorer_context context;                                                                           \
    proc_string str = process_string(py_str);                                                         \
                                                                                                      \
                                                                                                      \
    switch(str.kind){                                                                                 \
    case CHAR_STRING:                                                                                 \
        context.context = (void*) new rapidfuzz::fuzz::CLASS<rapidfuzz::basic_string_view<uint8_t>>(  \
            rapidfuzz::basic_string_view<uint8_t>((uint8_t*)str.data, str.length)                     \
        );                                                                                            \
                                                                                                      \
        if (def_process) {                                                                            \
            context.scorer = cached_##RATIO##_func_default_process<uint8_t>;                          \
        } else {                                                                                      \
            context.scorer = cached_##RATIO##_func<uint8_t>;                                          \
        }                                                                                             \
        context.deinit = cached_##RATIO##_deinit<uint8_t>;                                            \
        break;                                                                                        \
    default:                                                                                          \
        context.context = (void*) new rapidfuzz::fuzz::CLASS<rapidfuzz::basic_string_view<Py_UNICODE>>( \
            rapidfuzz::basic_string_view<Py_UNICODE>((Py_UNICODE*)str.data, str.length)               \
        );                                                                                            \
                                                                                                      \
        if (def_process) {                                                                            \
            context.scorer = cached_##RATIO##_func_default_process<Py_UNICODE>;                       \
        } else {                                                                                      \
            context.scorer = cached_##RATIO##_func<Py_UNICODE>;                                       \
        }                                                                                             \
        context.deinit = cached_##RATIO##_deinit<Py_UNICODE>;                                         \
        break;                                                                                        \
    }                                                                                                 \
                                                                                                      \
    return context;                                                                                   \
}

#endif


CACHED_RATIO_INIT(ratio,                    CachedRatio)
CACHED_RATIO_INIT(partial_ratio,            CachedPartialRatio)
CACHED_RATIO_INIT(token_sort_ratio,         CachedTokenSortRatio)
CACHED_RATIO_INIT(token_set_ratio,          CachedTokenSetRatio)
CACHED_RATIO_INIT(token_ratio,              CachedTokenRatio)
CACHED_RATIO_INIT(partial_token_sort_ratio, CachedPartialTokenSortRatio)
CACHED_RATIO_INIT(partial_token_set_ratio,  CachedPartialTokenSetRatio)
CACHED_RATIO_INIT(partial_token_ratio,      CachedPartialTokenRatio)
CACHED_RATIO_INIT(WRatio,                   CachedWRatio)
CACHED_RATIO_INIT(QRatio,                   CachedQRatio)