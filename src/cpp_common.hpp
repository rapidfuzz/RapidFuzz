#pragma once
#include "Python.h"
#define RAPIDFUZZ_PYTHON
#include <rapidfuzz/fuzz.hpp>
#include <rapidfuzz/utils.hpp>
#include <rapidfuzz/string_metric.hpp>
#include <exception>
#include <iostream>

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

#if PY_VERSION_HEX > PYTHON_VERSION(3, 0, 0)
#define LIST_OF_CASES()   \
    X_ENUM(RAPIDFUZZ_UINT8,  uint8_t ) \
    X_ENUM(RAPIDFUZZ_UINT16, uint16_t) \
    X_ENUM(RAPIDFUZZ_UINT32, uint32_t) \
    X_ENUM(RAPIDFUZZ_UINT64, uint64_t)
#else /* Python2 */
#define LIST_OF_CASES()   \
    X_ENUM(RAPIDFUZZ_CHAR,    char      ) \
    X_ENUM(RAPIDFUZZ_UNICODE, Py_UNICODE) \
    X_ENUM(RAPIDFUZZ_UINT64,  uint64_t  )
#endif


enum RapidfuzzType {
# define X_ENUM(kind, type) kind,
    LIST_OF_CASES()
# undef X_ENUM
};

struct proc_string {
    RapidfuzzType kind;
    bool allocated;
    void* data;
    size_t length;

    proc_string()
      : kind((RapidfuzzType)0),  allocated(false), data(nullptr), length(0) {}
    proc_string(RapidfuzzType _kind, uint8_t _allocated, void* _data, size_t _length)
      : kind(_kind), allocated(_allocated), data(_data), length(_length) {}

    proc_string(const proc_string&) = delete;
    proc_string& operator=(const proc_string&) = delete;

    proc_string(proc_string&& other)
     : kind(other.kind), allocated(other.allocated), data(other.data), length(other.length)
    {
        other.data = nullptr;
        other.allocated = false;
    }

    proc_string& operator=(proc_string&& other) {
        if (&other != this) {
            if (allocated) {
                free(data);
            }
            kind = other.kind;
            allocated = other.allocated;
            data = other.data;
            length = other.length;

            other.data = nullptr;
            other.allocated = false;
      }
      return *this;
    };

    ~proc_string() {
        if (allocated) {
            free(data);
        }
    }
};

template <typename T>
static inline rapidfuzz::basic_string_view<T> no_process(const proc_string& s)
{
    return rapidfuzz::basic_string_view<T>((T*)s.data, s.length);
}

template <typename T>
static inline std::basic_string<T> default_process(const proc_string& s)
{
    return utils::default_process(no_process<T>(s));
}

template <typename Func, typename... Args>
auto visit(const proc_string& str, Func&& f, Args&&... args)
{
    switch(str.kind){
# define X_ENUM(kind, type) case kind: return f(no_process<type>(str), std::forward<Args>(args)...);
    LIST_OF_CASES()
# undef X_ENUM
    default:                                           
        throw std::logic_error("Invalid string type");
    }
}

template <typename Func, typename... Args>
auto visit_default_process(const proc_string& str, Func&& f, Args&&... args)
{
    switch(str.kind){
# define X_ENUM(kind, type) case kind: return f(default_process<type>(str), std::forward<Args>(args)...);
    LIST_OF_CASES()
# undef X_ENUM
    default:                                           
        throw std::logic_error("Invalid string type");
    }
}

template <typename Func, typename... Args>
auto visitor(const proc_string& str1, const proc_string& str2, Func&& f, Args&&... args)
{
    return visit(str2,
        [&](auto str) {
            return visit(str1, std::forward<Func>(f), str, std::forward<Args>(args)...);
        }
    ); 
}

/* todo this should be refactored in the future since preprocessing does not really belong here */
template <typename Func, typename... Args>
auto visitor_default_process(const proc_string& str1, const proc_string& str2, Func&& f, Args&&... args)
{
    return visit_default_process(str2,
        [&](auto str) {
            return visit_default_process(str1, std::forward<Func>(f), str, std::forward<Args>(args)...);
        }
    ); 
}


static inline PyObject* dist_to_long(std::size_t dist)
{
    if (dist == (std::size_t)-1) {
        return PyLong_FromLong(-1);
    }
    return PyLong_FromSize_t(dist);
}

static inline bool is_valid_string(PyObject* py_str)
{
    bool is_string = false;

#if PY_VERSION_HEX > PYTHON_VERSION(3, 0, 0)
    if (PyBytes_Check(py_str)) {
        is_string = true;
    }
    else if (PyUnicode_Check(py_str)) {
        // PEP 623 deprecates legacy strings and therefor
        // deprecates e.g. PyUnicode_READY in Python 3.10
#if PY_VERSION_HEX < PYTHON_VERSION(3, 10, 0)
        if (PyUnicode_READY(py_str)) {
          // cython will use the exception set by PyUnicode_READY
          throw std::runtime_error("");
        }
#endif
        is_string = true;
    }
#else /* Python2 */
    if (PyObject_TypeCheck(py_str, &PyString_Type)) {
        is_string = true;
    }
    else if (PyObject_TypeCheck(py_str, &PyUnicode_Type)) {
        is_string = true;
    }
#endif

    return is_string;
}

static inline void validate_string(PyObject* py_str, const char* err)
{
#if PY_VERSION_HEX > PYTHON_VERSION(3, 0, 0)
    if (PyBytes_Check(py_str)) {
        return;
    }
    else if (PyUnicode_Check(py_str)) {
        // PEP 623 deprecates legacy strings and therefor
        // deprecates e.g. PyUnicode_READY in Python 3.10
#if PY_VERSION_HEX < PYTHON_VERSION(3, 10, 0)
        if (PyUnicode_READY(py_str)) {
          // cython will use the exception set by PyUnicode_READY
          throw std::runtime_error("");
        }
#endif
        return;
    }
#else /* Python2 */
    if (PyObject_TypeCheck(py_str, &PyString_Type)) {
        return;
    }
    else if (PyObject_TypeCheck(py_str, &PyUnicode_Type)) {
        return;
    }
#endif

    throw PythonTypeError(err);
}

static inline proc_string convert_string(PyObject* py_str)
{
#if PY_VERSION_HEX > PYTHON_VERSION(3, 0, 0)
    if (PyBytes_Check(py_str)) {
        return {
            RAPIDFUZZ_UINT8,
            false,
            PyBytes_AS_STRING(py_str),
            static_cast<std::size_t>(PyBytes_GET_SIZE(py_str))
        };
    } else {
        RapidfuzzType kind;
        switch(PyUnicode_KIND(py_str)) {
        case PyUnicode_1BYTE_KIND:
           kind = RAPIDFUZZ_UINT8;
           break;
        case PyUnicode_2BYTE_KIND:
           kind = RAPIDFUZZ_UINT16;
           break;
        default:
           kind = RAPIDFUZZ_UINT32;
           break;
        }
     
        return proc_string(
            kind,
            false,
            PyUnicode_DATA(py_str),
            static_cast<std::size_t>(PyUnicode_GET_LENGTH(py_str))
        );
    }
#else /* Python2 */
    if (PyObject_TypeCheck(py_str, &PyString_Type)) {
        return {
            RAPIDFUZZ_CHAR,
            false,
            PyString_AS_STRING(py_str),
            static_cast<std::size_t>(PyString_GET_SIZE(py_str))
        };
    }
    else if (PyObject_TypeCheck(py_str, &PyUnicode_Type)) {
        return {
            RAPIDFUZZ_UNICODE,
            false,
            PyUnicode_AS_UNICODE(py_str),
            static_cast<std::size_t>(PyUnicode_GET_SIZE(py_str))
        };
    }
    else {
        throw PythonTypeError("choice must be a String, Unicode or None");
    }
#endif
}

template <typename CharT>
proc_string default_process_func_impl(proc_string sentence) {
    CharT* str = static_cast<CharT*>(sentence.data);
    if (!sentence.allocated)
    {
      CharT* temp_str = (CharT*)malloc(sentence.length * sizeof(CharT));
      if (temp_str == NULL)
      {
          throw std::bad_alloc();
      }
      std::copy(str, str + sentence.length, temp_str);
      str = temp_str;
    }

    sentence.allocated = true;
    sentence.data = str;
    sentence.kind = sentence.kind;
    sentence.length = utils::default_process(str, sentence.length);

    return sentence;
}

proc_string default_process_func(proc_string sentence) {
    switch (sentence.kind) {
    # define X_ENUM(KIND, TYPE) case KIND: return default_process_func_impl<TYPE>(std::move(sentence));
    LIST_OF_CASES()
    default:
       throw std::logic_error("Reached end of control flow in default_process_func");
    # undef X_ENUM
    }
}