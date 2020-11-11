#pragma once


template <typename MatchingFunc>
inline PyObject* fuzz_call(bool processor_default, PyObject* args, PyObject* keywds)
{
  std::vector<PyObject*> owner_list;
  python_string proc_s1;
  python_string proc_s2;

  PyObject* py_s1;
  PyObject* py_s2;
  PyObject* processor = NULL;
  double score_cutoff = 0;
  static const char* kwlist[] = {"s1", "s2", "processor", "score_cutoff", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|Od", const_cast<char**>(kwlist), &py_s1,
                                   &py_s2, &processor, &score_cutoff))
  {
    return NULL;
  }

  if (py_s1 == Py_None || py_s2 == Py_None) {
    return PyFloat_FromDouble(0);
  }

  if (!valid_str(py_s1, "s1") || !valid_str(py_s2, "s2")) {
    return NULL;
  }

  if (!process_string(py_s1, processor, processor_default, proc_s1, owner_list)) {
    return NULL;
  }

  if (!process_string(py_s2, processor, processor_default, proc_s2, owner_list)) {
    free_owner_list(owner_list);
    return NULL;
  }

  double result = mpark::visit(
        [score_cutoff](auto&& val1, auto&& val2) {
          return MatchingFunc::call(val1, val2, score_cutoff);
        },
        proc_s1, proc_s2);

  free_owner_list(owner_list);
  return PyFloat_FromDouble(result);
}