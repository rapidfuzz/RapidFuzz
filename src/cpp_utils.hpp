#include "cpp_common.hpp"

PyObject* default_process_impl(PyObject* sentence) {
    proc_string c_sentence = convert_string(sentence, "sentence must be a String");

    switch (c_sentence.kind) {
    case PyUnicode_1BYTE_KIND:
    {
        auto proc_str = utils::default_process(
            rapidfuzz::basic_string_view<uint8_t>(static_cast<uint8_t*>(c_sentence.data), c_sentence.length));
        return PyUnicode_FromKindAndData(PyUnicode_1BYTE_KIND, proc_str.data(), proc_str.size());
    }
    case PyUnicode_2BYTE_KIND:
    {
        auto proc_str = utils::default_process(
            rapidfuzz::basic_string_view<uint16_t>(static_cast<uint16_t*>(c_sentence.data), c_sentence.length));
        return PyUnicode_FromKindAndData(PyUnicode_2BYTE_KIND, proc_str.data(), proc_str.size());
    }
    default:
    {
        auto proc_str = utils::default_process(
            rapidfuzz::basic_string_view<uint32_t>(static_cast<uint32_t*>(c_sentence.data), c_sentence.length));
        return PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, proc_str.data(), proc_str.size());
    }
    }
}
