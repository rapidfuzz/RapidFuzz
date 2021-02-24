#include "Python.h"
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

template <typename CachedScorer>
static void cached_deinit(void* context)
{
    delete (CachedScorer*)context;
}

template<typename CachedScorer>
static inline double cached_func_default_process(
    void* context, PyObject* py_str, double score_cutoff)
{
    proc_string str = convert_string(py_str);
    CachedScorer* ratio = (CachedScorer*)context;

    switch(str.kind){
#if PY_VERSION_HEX >= PYTHON_VERSION(3, 0, 0)
    case PyUnicode_1BYTE_KIND:
        return ratio->ratio(
            utils::default_process(
                rapidfuzz::basic_string_view<uint8_t>((uint8_t*)str.data, str.length)),
            score_cutoff
        );
    case PyUnicode_2BYTE_KIND:
        return ratio->ratio(
            utils::default_process(
                rapidfuzz::basic_string_view<uint16_t>((uint16_t*)str.data, str.length)),
            score_cutoff
        );
    case PyUnicode_4BYTE_KIND:
        return ratio->ratio(
            utils::default_process(
                rapidfuzz::basic_string_view<uint32_t>((uint32_t*)str.data, str.length)),
            score_cutoff
        );
#else
    case CHAR_STRING:
        return ratio->ratio(
            utils::default_process(
                rapidfuzz::basic_string_view<uint8_t>((uint8_t*)str.data, str.length)),
            score_cutoff
        );
    case UNICODE_STRING:
        return ratio->ratio(
            utils::default_process(
                rapidfuzz::basic_string_view<Py_UNICODE>((Py_UNICODE*)str.data, str.length)),
            score_cutoff
        );
#endif
    default:
       throw std::logic_error("Reached end of control flow in cached_func_default_process");
    }
}

template<typename CachedScorer>
static inline double cached_func(void* context, PyObject* py_str, double score_cutoff)
{
    proc_string str = convert_string(py_str);
    CachedScorer* ratio = (CachedScorer*)context;

    switch(str.kind){
#if PY_VERSION_HEX >= PYTHON_VERSION(3, 0, 0)
    case PyUnicode_1BYTE_KIND:
        return ratio->ratio(
            rapidfuzz::basic_string_view<uint8_t>((uint8_t*)str.data, str.length),
            score_cutoff
        );
    case PyUnicode_2BYTE_KIND:
        return ratio->ratio(
            rapidfuzz::basic_string_view<uint16_t>((uint16_t*)str.data, str.length),
            score_cutoff
        );
    case PyUnicode_4BYTE_KIND:
        return ratio->ratio(
            rapidfuzz::basic_string_view<uint32_t>((uint32_t*)str.data, str.length),
            score_cutoff
        );
#else
    case CHAR_STRING:
        return ratio->ratio(
            rapidfuzz::basic_string_view<uint8_t>((uint8_t*)str.data, str.length),
            score_cutoff
        );
    case UNICODE_STRING:
        return ratio->ratio(
            rapidfuzz::basic_string_view<Py_UNICODE>((Py_UNICODE*)str.data, str.length),
            score_cutoff
        );
#endif
    default:
       throw std::logic_error("Reached end of control flow in cached_func");
    }
}

template<template <typename> class CachedScorer, typename CharT>
static inline scorer_context get_scorer_context(const proc_string& str, int def_process)
{
    using Sentence = rapidfuzz::basic_string_view<CharT>;
    scorer_context context;
    context.context = (void*) new CachedScorer<Sentence>(Sentence((CharT*)str.data, str.length));

    if (def_process) {
        context.scorer = cached_func_default_process<CachedScorer<Sentence>>;
    } else {
        context.scorer = cached_func<CachedScorer<Sentence>>;
    }
    context.deinit = cached_deinit<CachedScorer<Sentence>>;
    return context;
}

template<template <typename> class CachedScorer>
static inline scorer_context cached_init(PyObject* py_str, int def_process)
{
    proc_string str = convert_string(py_str);
    if (str.data == NULL) return {NULL, NULL, NULL};

    switch(str.kind){
#if PY_VERSION_HEX >= PYTHON_VERSION(3, 0, 0)
    case PyUnicode_1BYTE_KIND:
        return get_scorer_context<CachedScorer, uint8_t>(str, def_process);
    case PyUnicode_2BYTE_KIND:
        return get_scorer_context<CachedScorer, uint16_t>(str, def_process);
    case PyUnicode_4BYTE_KIND:
        return get_scorer_context<CachedScorer, uint32_t>(str, def_process);
#else
    case CHAR_STRING:
        return get_scorer_context<CachedScorer, uint8_t>(str, def_process);
    case UNICODE_STRING:
        return get_scorer_context<CachedScorer, Py_UNICODE>(str, def_process);
#endif
    default:
       throw std::logic_error("Reached end of control flow in cached_init");
    }
}

/* fuzz */
static scorer_context cached_ratio_init(PyObject* py_str, int def_process)
{
    return cached_init<fuzz::CachedRatio>(py_str, def_process);
}

static scorer_context cached_partial_ratio_init(PyObject* py_str, int def_process)
{
    return cached_init<fuzz::CachedPartialRatio>(py_str, def_process);
}

static scorer_context cached_token_sort_ratio_init(PyObject* py_str, int def_process)
{
    return cached_init<fuzz::CachedTokenSortRatio>(py_str, def_process);
}

static scorer_context cached_token_set_ratio_init(PyObject* py_str, int def_process)
{
    return cached_init<fuzz::CachedTokenSetRatio>(py_str, def_process);
}

static scorer_context cached_token_ratio_init(PyObject* py_str, int def_process)
{
    return cached_init<fuzz::CachedTokenRatio>(py_str, def_process);
}

static scorer_context cached_partial_token_sort_ratio_init(PyObject* py_str, int def_process)
{
    return cached_init<fuzz::CachedPartialTokenSortRatio>(py_str, def_process);
}

static scorer_context cached_partial_token_set_ratio_init(PyObject* py_str, int def_process)
{
    return cached_init<fuzz::CachedPartialTokenSetRatio>(py_str, def_process);
}

static scorer_context cached_partial_token_ratio_init(PyObject* py_str, int def_process)
{
    return cached_init<fuzz::CachedPartialTokenRatio>(py_str, def_process);
}

static scorer_context cached_WRatio_init(PyObject* py_str, int def_process)
{
    return cached_init<fuzz::CachedWRatio>(py_str, def_process);
}

static scorer_context cached_QRatio_init(PyObject* py_str, int def_process)
{
    return cached_init<fuzz::CachedQRatio>(py_str, def_process);
}

/* string_metric */
static scorer_context cached_normalized_hamming_init(PyObject* py_str, int def_process)
{
    return cached_init<string_metric::CachedNormalizedHamming>(py_str, def_process);
}
