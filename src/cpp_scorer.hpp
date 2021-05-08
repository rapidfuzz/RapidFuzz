#include "cpp_common.hpp"

SIMPLE_RATIO_DEF(ratio)
SIMPLE_RATIO_DEF(partial_ratio)
SIMPLE_RATIO_DEF(token_sort_ratio)
SIMPLE_RATIO_DEF(token_set_ratio)
SIMPLE_RATIO_DEF(token_ratio)
SIMPLE_RATIO_DEF(partial_token_sort_ratio)
SIMPLE_RATIO_DEF(partial_token_set_ratio)
SIMPLE_RATIO_DEF(partial_token_ratio)
SIMPLE_RATIO_DEF(WRatio)
SIMPLE_RATIO_DEF(QRatio)

SIMPLE_DISTANCE_DEF(hamming)
SIMPLE_RATIO_DEF(normalized_hamming)

PyObject* levenshtein_no_process(proc_string s1, proc_string s2,
    size_t insertion, size_t deletion, size_t substitution, size_t max)
{
    rapidfuzz::LevenshteinWeightTable weights = {insertion, deletion, substitution};

    size_t result = levenshtein_impl_no_process(s1, s2, weights, max);
    return dist_to_long(result);
}

PyObject* levenshtein_default_process(proc_string s1, proc_string s2,
    size_t insertion, size_t deletion, size_t substitution, size_t max)
{
    rapidfuzz::LevenshteinWeightTable weights = {insertion, deletion, substitution};

    size_t result = levenshtein_impl_default_process(s1, s2, weights, max);
    return dist_to_long(result);
}

double normalized_levenshtein_no_process(proc_string s1, proc_string s2,
    size_t insertion, size_t deletion, size_t substitution, double score_cutoff)
{
    rapidfuzz::LevenshteinWeightTable weights = {insertion, deletion, substitution};

    return normalized_levenshtein_impl_no_process(s1, s2, weights, score_cutoff);
}

double normalized_levenshtein_default_process(proc_string s1, proc_string s2,
    size_t insertion, size_t deletion, size_t substitution, double score_cutoff)
{
    rapidfuzz::LevenshteinWeightTable weights = {insertion, deletion, substitution};

    return normalized_levenshtein_impl_default_process(s1, s2, weights, score_cutoff);
}
