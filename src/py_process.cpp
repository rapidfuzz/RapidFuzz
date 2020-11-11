#include "fuzz.hpp"
#include "py_utils.hpp"
#include "utils.hpp"
#include <string>

namespace rfuzz = rapidfuzz::fuzz;
namespace rutils = rapidfuzz::utils;

static inline bool use_preprocessing(PyObject* processor, bool processor_default)
{
  return processor ? PyObject_IsTrue(processor) != 0 : processor_default;
}

static inline bool non_default_process(PyObject* processor)
{
  return false;
  /*if (processor) {
    if (PyCFunction_Check(processor)) {
        if (PyCFunction_GetFunction(processor) == (PyCFunction)(void (*)(void))default_process) {
          return false;
        }
    }
  }

  return PyCallable_Check(processor);*/
}



static PyMethodDef methods[] = {
    PY_METHOD(extractOne),
/* sentinel */
    {NULL, NULL, 0, NULL}
};

PY_INIT_MOD(test, NULL, methods)