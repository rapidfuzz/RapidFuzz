#pragma once
#include "Python.h"
#define RAPIDFUZZ_PYTHON
#include <rapidfuzz/fuzz.hpp>
#include <rapidfuzz/utils.hpp>
#include <rapidfuzz/string_metric.hpp>
#include <exception>
#include <iostream>

#include "rapidfuzz_capi.h"

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

/* copy from cython */
static inline void CppExn2PyErr() {
  try {
    if (PyErr_Occurred())
      ; // let the latest Python exn pass through and ignore the current one
    else
      throw;
  } catch (const std::bad_alloc& exn) {
    PyErr_SetString(PyExc_MemoryError, exn.what());
  } catch (const std::bad_cast& exn) {
    PyErr_SetString(PyExc_TypeError, exn.what());
  } catch (const std::bad_typeid& exn) {
    PyErr_SetString(PyExc_TypeError, exn.what());
  } catch (const std::domain_error& exn) {
    PyErr_SetString(PyExc_ValueError, exn.what());
  } catch (const std::invalid_argument& exn) {
    PyErr_SetString(PyExc_ValueError, exn.what());
  } catch (const std::ios_base::failure& exn) {
    PyErr_SetString(PyExc_IOError, exn.what());
  } catch (const std::out_of_range& exn) {
    PyErr_SetString(PyExc_IndexError, exn.what());
  } catch (const std::overflow_error& exn) {
    PyErr_SetString(PyExc_OverflowError, exn.what());
  } catch (const std::range_error& exn) {
    PyErr_SetString(PyExc_ArithmeticError, exn.what());
  } catch (const std::underflow_error& exn) {
    PyErr_SetString(PyExc_ArithmeticError, exn.what());
  } catch (const std::exception& exn) {
    PyErr_SetString(PyExc_RuntimeError, exn.what());
  }
  catch (...)
  {
    PyErr_SetString(PyExc_RuntimeError, "Unknown exception");
  }
}

static void PyErr2RuntimeExn(int err) {
    if (err == -1)
    {
        // Python exceptions should be already set and will be retrieved by Cython
        throw std::runtime_error("");
    }
}

#if PY_VERSION_HEX > PYTHON_VERSION(3, 0, 0)
#define LIST_OF_CASES()   \
    X_ENUM(RF_UINT8,  uint8_t ) \
    X_ENUM(RF_UINT16, uint16_t) \
    X_ENUM(RF_UINT32, uint32_t) \
    X_ENUM(RF_UINT64, uint64_t)
#else /* Python2 */
#define LIST_OF_CASES()   \
    X_ENUM(RF_CHAR,    char      ) \
    X_ENUM(RF_UNICODE, Py_UNICODE) \
    X_ENUM(RF_UINT64,  uint64_t  )
#endif

/* RAII Wrapper for RfString */
struct RfStringWrapper {
    RfString string;

    RfStringWrapper()
        : string({(RfStringType)0, nullptr, 0, nullptr, nullptr}) {}

    RfStringWrapper(RfString string_)
        : string(string_) {}

    RfStringWrapper(const RfStringWrapper&) = delete;
    RfStringWrapper& operator=(const RfStringWrapper&) = delete;

    RfStringWrapper(RfStringWrapper&& other)
    {
        string = other.string;
        other.string = {(RfStringType)0, nullptr, 0, nullptr, nullptr};
    }

    RfStringWrapper& operator=(RfStringWrapper&& other) {
        if (&other != this) {
            if (string.deinit) {
                string.deinit(&string);
            }
            string = other.string;
            other.string = {(RfStringType)0, nullptr, 0, nullptr, nullptr};
      }
      return *this;
    };

    ~RfStringWrapper() {
        if (string.deinit) {
            string.deinit(&string);
        }
    }
};

/* RAII Wrapper for RfKwargsContext */
struct RfKwargsContextWrapper {
    RfKwargsContext kwargs;

    RfKwargsContextWrapper()
        : kwargs({NULL, NULL}) {}

    RfKwargsContextWrapper(RfKwargsContext kwargs_)
        : kwargs(kwargs_) {}

    RfKwargsContextWrapper(const RfKwargsContextWrapper&) = delete;
    RfKwargsContextWrapper& operator=(const RfKwargsContextWrapper&) = delete;

    RfKwargsContextWrapper(RfKwargsContextWrapper&& other)
    {
        kwargs = other.kwargs;
        other.kwargs = {NULL, NULL};
    }

    RfKwargsContextWrapper& operator=(RfKwargsContextWrapper&& other)
    {
        if (&other != this) {
            if (kwargs.deinit) {
                kwargs.deinit(&kwargs);
            }
            kwargs = other.kwargs;
            other.kwargs = {NULL, NULL};
      }
      return *this;
    };

    ~RfKwargsContextWrapper() {
        if (kwargs.deinit) {
            kwargs.deinit(&kwargs);
        }
    }
};

void default_string_deinit(RfString* string)
{
    free(string->data);
}

template <typename T>
static inline rapidfuzz::basic_string_view<T> no_process(const RfString& s)
{
    return rapidfuzz::basic_string_view<T>((T*)s.data, s.length);
}

template <typename T>
static inline std::basic_string<T> default_process(const RfString& s)
{
    return utils::default_process(no_process<T>(s));
}

template <typename Func, typename... Args>
auto visit(const RfString& str, Func&& f, Args&&... args)
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
auto visit_default_process(const RfString& str, Func&& f, Args&&... args)
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
auto visitor(const RfString& str1, const RfString& str2, Func&& f, Args&&... args)
{
    return visit(str2,
        [&](auto str) {
            return visit(str1, std::forward<Func>(f), str, std::forward<Args>(args)...);
        }
    ); 
}

/* todo this should be refactored in the future since preprocessing does not really belong here */
template <typename Func, typename... Args>
auto visitor_default_process(const RfString& str1, const RfString& str2, Func&& f, Args&&... args)
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

static inline RfString convert_string(PyObject* py_str)
{
#if PY_VERSION_HEX > PYTHON_VERSION(3, 0, 0)
    if (PyBytes_Check(py_str)) {
        return {
            RF_UINT8,
            PyBytes_AS_STRING(py_str),
            static_cast<std::size_t>(PyBytes_GET_SIZE(py_str)),
            nullptr,
            nullptr
        };
    } else {
        RfStringType kind;
        switch(PyUnicode_KIND(py_str)) {
        case PyUnicode_1BYTE_KIND:
           kind = RF_UINT8;
           break;
        case PyUnicode_2BYTE_KIND:
           kind = RF_UINT16;
           break;
        default:
           kind = RF_UINT32;
           break;
        }
     
        return {
            kind,
            PyUnicode_DATA(py_str),
            static_cast<std::size_t>(PyUnicode_GET_LENGTH(py_str)),
            nullptr,
            nullptr
        };
    }
#else /* Python2 */
    if (PyObject_TypeCheck(py_str, &PyString_Type)) {
        return {
            RF_CHAR,
            PyString_AS_STRING(py_str),
            static_cast<std::size_t>(PyString_GET_SIZE(py_str)),
            nullptr,
            nullptr
        };
    }
    else if (PyObject_TypeCheck(py_str, &PyUnicode_Type)) {
        return {
            RF_UNICODE,
            PyUnicode_AS_UNICODE(py_str),
            static_cast<std::size_t>(PyUnicode_GET_SIZE(py_str)),
            nullptr,
            nullptr
        };
    }
    else {
        throw PythonTypeError("choice must be a String, Unicode or None");
    }
#endif
}

template <typename CharT>
RfString default_process_func_impl(RfString sentence) {
    CharT* str = static_cast<CharT*>(sentence.data);

    if (!sentence.deinit)
    {
      CharT* temp_str = (CharT*)malloc(sentence.length * sizeof(CharT));
      if (temp_str == NULL)
      {
          throw std::bad_alloc();
      }
      std::copy(str, str + sentence.length, temp_str);
      str = temp_str;
    }

    sentence.deinit = default_string_deinit;
    sentence.data = str;
    sentence.kind = sentence.kind;
    sentence.length = utils::default_process(str, sentence.length);

    return sentence;
}

RfString default_process_func(RfString sentence) {
    switch (sentence.kind) {
    # define X_ENUM(KIND, TYPE) case KIND: return default_process_func_impl<TYPE>(std::move(sentence));
    LIST_OF_CASES()
    default:
       throw std::logic_error("Reached end of control flow in default_process_func");
    # undef X_ENUM
    }
}