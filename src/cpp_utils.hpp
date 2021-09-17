#pragma once
#include "cpp_common.hpp"

PyObject* default_process_impl(PyObject* sentence) {
    proc_string c_sentence = convert_string(sentence);

    switch (c_sentence.kind) {
#if PY_VERSION_HEX > PYTHON_VERSION(3, 0, 0)
    case RAPIDFUZZ_UINT8:
    {
        auto proc_str = utils::default_process(
            rapidfuzz::basic_string_view<uint8_t>(static_cast<uint8_t*>(c_sentence.data), c_sentence.length));
        return PyUnicode_FromKindAndData(PyUnicode_1BYTE_KIND, proc_str.data(), (Py_ssize_t)proc_str.size());
    }
    case RAPIDFUZZ_UINT16:
    {
        auto proc_str = utils::default_process(
            rapidfuzz::basic_string_view<uint16_t>(static_cast<uint16_t*>(c_sentence.data), c_sentence.length));
        return PyUnicode_FromKindAndData(PyUnicode_2BYTE_KIND, proc_str.data(), (Py_ssize_t)proc_str.size());
    }
    case RAPIDFUZZ_UINT32:
    {
        auto proc_str = utils::default_process(
            rapidfuzz::basic_string_view<uint32_t>(static_cast<uint32_t*>(c_sentence.data), c_sentence.length));
        return PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, proc_str.data(), (Py_ssize_t)proc_str.size());
    }
#else
    case RAPIDFUZZ_CHAR:
    {
        auto proc_str = utils::default_process(
            rapidfuzz::basic_string_view<char>(static_cast<char*>(c_sentence.data), c_sentence.length));
        return PyString_FromStringAndSize(proc_str.data(), (Py_ssize_t)proc_str.size());
    }
    case RAPIDFUZZ_UNICODE:
    {
        auto proc_str = utils::default_process(
            rapidfuzz::basic_string_view<Py_UNICODE>(static_cast<Py_UNICODE*>(c_sentence.data), c_sentence.length));
        return PyUnicode_FromUnicode(proc_str.data(), (Py_ssize_t)proc_str.size());
    }
#endif
    // ToDo: for now do not process these elements should be done in some way in the future
    default:
        return sentence;
    }
}
