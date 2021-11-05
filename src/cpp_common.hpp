#pragma once
#include "Python.h"
#define RAPIDFUZZ_PYTHON
#include <rapidfuzz/fuzz.hpp>
#include <rapidfuzz/string_metric.hpp>
#include <exception>

#include "rapidfuzz_capi.h"

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

static inline void PyErr2RuntimeExn(bool success) {
    if (!success)
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

/* RAII Wrapper for RF_String */
struct RF_StringWrapper {
    RF_String string;
    PyObject* obj;

    RF_StringWrapper()
        : string({nullptr, (RF_StringType)0, nullptr, 0, nullptr}), obj(nullptr) {}

    RF_StringWrapper(RF_String string_)
        : string(string_), obj(nullptr) {}

    RF_StringWrapper(RF_String string_, PyObject* o)
        : string(string_), obj(o)
    {
        Py_XINCREF(obj);
    }

    RF_StringWrapper(const RF_StringWrapper&) = delete;
    RF_StringWrapper& operator=(const RF_StringWrapper&) = delete;

    RF_StringWrapper(RF_StringWrapper&& other)
    {
        string = other.string;
        obj = other.obj;
        other.string = {nullptr, (RF_StringType)0, nullptr, 0, nullptr};
        other.obj = nullptr;
    }

    RF_StringWrapper& operator=(RF_StringWrapper&& other) {
        if (&other != this) {
            if (string.dtor) {
                string.dtor(&string);
            }
            Py_XDECREF(obj);
            string = other.string;
            obj = other.obj;
            other.string = {nullptr, (RF_StringType)0, nullptr, 0, nullptr};
            other.obj = nullptr;
      }
      return *this;
    };

    ~RF_StringWrapper() {
        if (string.dtor) {
            string.dtor(&string);
        }
        Py_XDECREF(obj);
    }
};

/* RAII Wrapper for RF_Kwargs */
struct RF_KwargsWrapper {
    RF_Kwargs kwargs;

    RF_KwargsWrapper()
        : kwargs({NULL, NULL}) {}

    RF_KwargsWrapper(RF_Kwargs kwargs_)
        : kwargs(kwargs_) {}

    RF_KwargsWrapper(const RF_KwargsWrapper&) = delete;
    RF_KwargsWrapper& operator=(const RF_KwargsWrapper&) = delete;

    RF_KwargsWrapper(RF_KwargsWrapper&& other)
    {
        kwargs = other.kwargs;
        other.kwargs = {NULL, NULL};
    }

    RF_KwargsWrapper& operator=(RF_KwargsWrapper&& other)
    {
        if (&other != this) {
            if (kwargs.dtor) {
                kwargs.dtor(&kwargs);
            }
            kwargs = other.kwargs;
            other.kwargs = {NULL, NULL};
        }
        return *this;
    };

    ~RF_KwargsWrapper() {
        if (kwargs.dtor) {
            kwargs.dtor(&kwargs);
        }
    }
};

/* RAII Wrapper for PyObject* */
struct PyObjectWrapper {
    PyObject* obj;

    PyObjectWrapper()
        : obj(nullptr) {}

    PyObjectWrapper(PyObject* o)
        : obj(o)
    {
        Py_XINCREF(obj);
    }

    PyObjectWrapper(const PyObjectWrapper&) = delete;
    PyObjectWrapper& operator=(const PyObjectWrapper&) = delete;

    PyObjectWrapper(PyObjectWrapper&& other)
    {
        obj = other.obj;
        other.obj = nullptr;
    }

    PyObjectWrapper& operator=(PyObjectWrapper&& other) {
        if (&other != this) {
            Py_XDECREF(obj);
            obj = other.obj;
            other.obj = nullptr;
      }
      return *this;
    };

    ~PyObjectWrapper() {
        Py_XDECREF(obj);
    }
};

void default_string_deinit(RF_String* string)
{
    free(string->data);
}

template <typename T>
static inline rapidfuzz::basic_string_view<T> no_process(const RF_String& s)
{
    return rapidfuzz::basic_string_view<T>((T*)s.data, s.length);
}

template <typename Func, typename... Args>
auto visit(const RF_String& str, Func&& f, Args&&... args)
{
    switch(str.kind) {
# define X_ENUM(kind, type) case kind: return f(no_process<type>(str), std::forward<Args>(args)...);
    LIST_OF_CASES()
# undef X_ENUM
    default:
        throw std::logic_error("Invalid string type");
    }
}

template <typename Func, typename... Args>
auto visitor(const RF_String& str1, const RF_String& str2, Func&& f, Args&&... args)
{
    return visit(str2,
        [&](auto str) {
            return visit(str1, std::forward<Func>(f), str, std::forward<Args>(args)...);
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

static inline RF_String convert_string(PyObject* py_str)
{
#if PY_VERSION_HEX > PYTHON_VERSION(3, 0, 0)
    if (PyBytes_Check(py_str)) {
        return {
            nullptr,
            RF_UINT8,
            PyBytes_AS_STRING(py_str),
            static_cast<std::size_t>(PyBytes_GET_SIZE(py_str)),
            nullptr
        };
    } else {
        RF_StringType kind;
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
            nullptr,
            kind,
            PyUnicode_DATA(py_str),
            static_cast<std::size_t>(PyUnicode_GET_LENGTH(py_str)),
            nullptr
        };
    }
#else /* Python2 */
    if (PyObject_TypeCheck(py_str, &PyString_Type)) {
        return {
            nullptr,
            RF_CHAR,
            PyString_AS_STRING(py_str),
            static_cast<std::size_t>(PyString_GET_SIZE(py_str)),
            nullptr
        };
    }
    else if (PyObject_TypeCheck(py_str, &PyUnicode_Type)) {
        return {
            nullptr,
            RF_UNICODE,
            PyUnicode_AS_UNICODE(py_str),
            static_cast<std::size_t>(PyUnicode_GET_SIZE(py_str)),
            nullptr
        };
    }
    else {
        throw PythonTypeError("choice must be a String, Unicode or None");
    }
#endif
}