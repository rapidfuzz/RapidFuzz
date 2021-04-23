#include "cpp_common.hpp"

PyObject* default_process_impl(PyObject* sentence) {
    proc_string c_sentence = convert_string(sentence);

    switch (c_sentence.kind) {
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
    // ToDo: for now do not process these elements should be done in some way in the future
    default:
        return sentence;
    }
}
