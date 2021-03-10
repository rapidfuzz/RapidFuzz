#include "cpp_common.hpp"

/*
 * Levenshtein
 */

template<typename CharT>
size_t levenshtein_impl_inner(proc_string s1, proc_string s2,
    size_t insertion, size_t deletion, size_t substitution, size_t max)
{
    switch(s2.kind){
    case PyUnicode_1BYTE_KIND:
        return string_metric::levenshtein(
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),
            rapidfuzz::basic_string_view<uint8_t>((uint8_t*)s2.data, s2.length),
            {insertion, deletion, substitution}, max
        );
    case PyUnicode_2BYTE_KIND:
        return string_metric::levenshtein(
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),
            rapidfuzz::basic_string_view<uint16_t>((uint16_t*)s2.data, s2.length),
            {insertion, deletion, substitution}, max
        );
    default:
        return string_metric::levenshtein(
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),
            rapidfuzz::basic_string_view<uint32_t>((uint32_t*)s2.data, s2.length),
            {insertion, deletion, substitution}, max
        );
    }
}

PyObject* levenshtein_impl(PyObject* s1, PyObject* s2,
    size_t insertion, size_t deletion, size_t substitution, size_t max)
{
    size_t result = 0;
    proc_string c_s1 = convert_string(s1, "s1 must be a String");
    proc_string c_s2 = convert_string(s2, "s2 must be a String");

    switch(c_s1.kind){
    case PyUnicode_1BYTE_KIND:
        result = levenshtein_impl_inner<uint8_t>(
            c_s1, c_s2, insertion, deletion, substitution, max);
        break;
    case PyUnicode_2BYTE_KIND:
        result = levenshtein_impl_inner<uint16_t>(
            c_s1, c_s2, insertion, deletion, substitution, max);
        break;
    default:
        result = levenshtein_impl_inner<uint32_t>(
            c_s1, c_s2, insertion, deletion, substitution, max);
        break;
    }

    if (result == (std::size_t)-1) {
        return PyLong_FromLong(-1);
    }
    return PyLong_FromSize_t(result);
}


/*
 *  Normalized Levenshtein
 */

template<typename CharT>
inline double normalized_levenshtein_impl_inner(proc_string s1, proc_string s2,
    size_t insertion, size_t deletion, size_t substitution, double score_cutoff)
{
    switch(s2.kind){
    case PyUnicode_1BYTE_KIND:
        return string_metric::normalized_levenshtein(
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),
            rapidfuzz::basic_string_view<uint8_t>((uint8_t*)s2.data, s2.length),
            {insertion, deletion, substitution}, score_cutoff
        );
    case PyUnicode_2BYTE_KIND:
        return string_metric::normalized_levenshtein(
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),
            rapidfuzz::basic_string_view<uint16_t>((uint16_t*)s2.data, s2.length),
            {insertion, deletion, substitution}, score_cutoff
        );
    default:
        return string_metric::normalized_levenshtein(
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),
            rapidfuzz::basic_string_view<uint32_t>((uint32_t*)s2.data, s2.length),
            {insertion, deletion, substitution}, score_cutoff
        );
    }
}

double normalized_levenshtein_impl(PyObject* s1, PyObject* s2,
    size_t insertion, size_t deletion, size_t substitution, double score_cutoff)
{
    proc_string c_s1 = convert_string(s1, "s1 must be a String");
    proc_string c_s2 = convert_string(s2, "s2 must be a String");

    switch(c_s1.kind){
    case PyUnicode_1BYTE_KIND:
        return normalized_levenshtein_impl_inner<uint8_t>(
            c_s1, c_s2, insertion, deletion, substitution, score_cutoff);
    case PyUnicode_2BYTE_KIND:
        return normalized_levenshtein_impl_inner<uint16_t>(
            c_s1, c_s2, insertion, deletion, substitution, score_cutoff);
    default:
        return normalized_levenshtein_impl_inner<uint32_t>(
            c_s1, c_s2, insertion, deletion, substitution, score_cutoff);
    }
}


template<typename CharT>
inline double normalized_levenshtein_impl_inner_default_process(proc_string s1, proc_string s2,
    size_t insertion, size_t deletion, size_t substitution, double score_cutoff)
{
    switch(s2.kind){
    case PyUnicode_1BYTE_KIND:
        return string_metric::levenshtein(
            utils::default_process(
                rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length)
            ),
            rapidfuzz::basic_string_view<uint8_t>((uint8_t*)s2.data, s2.length),
            {insertion, deletion, substitution}, score_cutoff
        );
    case PyUnicode_2BYTE_KIND:
        return string_metric::levenshtein(
            utils::default_process(
                rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length)
            ),
            utils::default_process(
                rapidfuzz::basic_string_view<uint16_t>((uint16_t*)s1.data, s1.length)
            ),
            {insertion, deletion, substitution}, score_cutoff
        );
    default:
        return string_metric::levenshtein(
            utils::default_process(
                rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length)
            ),
            utils::default_process(
                rapidfuzz::basic_string_view<uint32_t>((uint32_t*)s1.data, s1.length)
            ),
            {insertion, deletion, substitution}, score_cutoff
        );
    }
}

double normalized_levenshtein_impl_default_process(PyObject* s1, PyObject* s2,
    size_t insertion, size_t deletion, size_t substitution, double score_cutoff)
{
    proc_string c_s1 = convert_string(s1, "s1 must be a String");
    proc_string c_s2 = convert_string(s2, "s2 must be a String");

    switch(c_s1.kind){
    case PyUnicode_1BYTE_KIND:
        return normalized_levenshtein_impl_inner_default_process<uint8_t>(
            c_s1, c_s2, insertion, deletion, substitution, score_cutoff);
    case PyUnicode_2BYTE_KIND:
        return normalized_levenshtein_impl_inner_default_process<uint16_t>(
            c_s1, c_s2, insertion, deletion, substitution, score_cutoff);
    default:
        return normalized_levenshtein_impl_inner_default_process<uint32_t>(
            c_s1, c_s2, insertion, deletion, substitution, score_cutoff);
    }
}

/*
 * Hamming
 */

template<typename CharT>
size_t hamming_impl_inner(proc_string s1, proc_string s2, size_t max)
{
    switch(s2.kind){
    case PyUnicode_1BYTE_KIND:
        return string_metric::hamming(
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),
            rapidfuzz::basic_string_view<uint8_t>((uint8_t*)s2.data, s2.length),
            max
        );
    case PyUnicode_2BYTE_KIND:
        return string_metric::hamming(
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),
            rapidfuzz::basic_string_view<uint16_t>((uint16_t*)s2.data, s2.length),
            max
        );
    default:
        return string_metric::hamming(
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),
            rapidfuzz::basic_string_view<uint32_t>((uint32_t*)s2.data, s2.length),
            max
        );
    }
}

PyObject* hamming_impl(PyObject* s1, PyObject* s2, size_t max)
{
    size_t result = 0;
    proc_string c_s1 = convert_string(s1, "s1 must be a String");
    proc_string c_s2 = convert_string(s2, "s2 must be a String");

    switch(c_s1.kind){
    case PyUnicode_1BYTE_KIND:
        result = hamming_impl_inner<uint8_t>(c_s1, c_s2, max);
        break;
    case PyUnicode_2BYTE_KIND:
        result = hamming_impl_inner<uint16_t>(c_s1, c_s2, max);
        break;
    default:
        result = hamming_impl_inner<uint32_t>(c_s1, c_s2, max);
        break;
    }

    if (result == (std::size_t)-1) {
        return PyLong_FromLong(-1);
    }
    return PyLong_FromSize_t(result);
}


/*
 * Normalized Hamming
 */

template<typename CharT>
inline double normalized_hamming_impl_inner(proc_string s1, proc_string s2, double score_cutoff)
{
    switch(s2.kind){
    case PyUnicode_1BYTE_KIND:
        return string_metric::normalized_hamming(
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),
            rapidfuzz::basic_string_view<uint8_t>((uint8_t*)s2.data, s2.length),
            score_cutoff
        );
    case PyUnicode_2BYTE_KIND:
        return string_metric::normalized_hamming(
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),
            rapidfuzz::basic_string_view<uint16_t>((uint16_t*)s2.data, s2.length),
            score_cutoff
        );
    default:
        return string_metric::normalized_hamming(
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),
            rapidfuzz::basic_string_view<uint32_t>((uint32_t*)s2.data, s2.length),
            score_cutoff
        );
    }
}

double normalized_hamming_impl(PyObject* s1, PyObject* s2, double score_cutoff)
{
    proc_string c_s1 = convert_string(s1, "s1 must be a String");
    proc_string c_s2 = convert_string(s2, "s2 must be a String");

    switch(c_s1.kind){
    case PyUnicode_1BYTE_KIND:
        return normalized_hamming_impl_inner<uint8_t>(c_s1, c_s2, score_cutoff);
    case PyUnicode_2BYTE_KIND:
        return normalized_hamming_impl_inner<uint16_t>(c_s1, c_s2, score_cutoff);
    default:
        return normalized_hamming_impl_inner<uint32_t>(c_s1, c_s2, score_cutoff);
    }
}

template<typename CharT>
inline double normalized_hamming_impl_inner_default_process(
    proc_string s1, proc_string s2, double score_cutoff)
{
    switch(s2.kind){
    case PyUnicode_1BYTE_KIND:
        return string_metric::hamming(
            utils::default_process(
                rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length)
            ),
            rapidfuzz::basic_string_view<uint8_t>((uint8_t*)s2.data, s2.length),
            score_cutoff
        );
    case PyUnicode_2BYTE_KIND:
        return string_metric::hamming(
            utils::default_process(
                rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length)
            ),
            utils::default_process(
                rapidfuzz::basic_string_view<uint16_t>((uint16_t*)s1.data, s1.length)
            ),
            score_cutoff
        );
    default:
        return string_metric::hamming(
            utils::default_process(
                rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length)
            ),
            utils::default_process(
                rapidfuzz::basic_string_view<uint32_t>((uint32_t*)s1.data, s1.length)
            ),
            score_cutoff
        );
    }
}

double normalized_hamming_impl_default_process(PyObject* s1, PyObject* s2, double score_cutoff)
{
    proc_string c_s1 = convert_string(s1, "s1 must be a String");
    proc_string c_s2 = convert_string(s2, "s2 must be a String");

    switch(c_s1.kind){
    case PyUnicode_1BYTE_KIND:
        return normalized_hamming_impl_inner_default_process<uint8_t>(c_s1, c_s2, score_cutoff);
    case PyUnicode_2BYTE_KIND:
        return normalized_hamming_impl_inner_default_process<uint16_t>(c_s1, c_s2, score_cutoff);
    default:
        return normalized_hamming_impl_inner_default_process<uint32_t>(c_s1, c_s2, score_cutoff);
    }
}
