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
SIMPLE_RATIO_DEF(jaro_similarity)

PyObject* levenshtein_no_process(const proc_string& s1, const proc_string& s2,
    size_t insertion, size_t deletion, size_t substitution, size_t max)
{
    rapidfuzz::LevenshteinWeightTable weights = {insertion, deletion, substitution};

    size_t result = levenshtein_impl_no_process(s1, s2, weights, max);
    return dist_to_long(result);
}

PyObject* levenshtein_default_process(const proc_string& s1, const proc_string& s2,
    size_t insertion, size_t deletion, size_t substitution, size_t max)
{
    rapidfuzz::LevenshteinWeightTable weights = {insertion, deletion, substitution};

    size_t result = levenshtein_impl_default_process(s1, s2, weights, max);
    return dist_to_long(result);
}

double normalized_levenshtein_no_process(const proc_string& s1, const proc_string& s2,
    size_t insertion, size_t deletion, size_t substitution, double score_cutoff)
{
    rapidfuzz::LevenshteinWeightTable weights = {insertion, deletion, substitution};

    return normalized_levenshtein_impl_no_process(s1, s2, weights, score_cutoff);
}

double normalized_levenshtein_default_process(const proc_string& s1, const proc_string& s2,
    size_t insertion, size_t deletion, size_t substitution, double score_cutoff)
{
    rapidfuzz::LevenshteinWeightTable weights = {insertion, deletion, substitution};

    return normalized_levenshtein_impl_default_process(s1, s2, weights, score_cutoff);
}

double jaro_winkler_similarity_no_process(const proc_string& s1, const proc_string& s2,
    double prefix_weight, double score_cutoff)
{
    return jaro_winkler_similarity_impl_no_process(s1, s2, prefix_weight, score_cutoff);
}

double jaro_winkler_similarity_default_process(const proc_string& s1, const proc_string& s2,
    double prefix_weight, double score_cutoff)
{
    return jaro_winkler_similarity_impl_default_process(s1, s2, prefix_weight, score_cutoff);
}

# define X_ENUM(KIND, TYPE, MSVC_TUPLE) \
    case KIND: return GET_RATIO_FUNC MSVC_TUPLE  (s2, GET_PROCESSOR MSVC_TUPLE <TYPE>(s1));

template<typename Sentence>
std::vector<rapidfuzz::LevenshteinEditOp>
levenshtein_editops_inner_no_process(const proc_string& s1, const Sentence& s2)
{
    switch(s1.kind){
    LIST_OF_CASES(string_metric::levenshtein_editops, no_process)
    }
    assert(false); /* silence any warnings about missing return value */   
}

std::vector<rapidfuzz::LevenshteinEditOp>
levenshtein_editops_no_process(const proc_string& s1, const proc_string& s2)
{
    switch(s1.kind){
    LIST_OF_CASES(levenshtein_editops_inner_no_process, no_process)
    }
    assert(false); /* silence any warnings about missing return value */
}

template<typename Sentence>
std::vector<rapidfuzz::LevenshteinEditOp>
levenshtein_editops_inner_default_process(const proc_string& s1, const Sentence& s2)
{
    switch(s1.kind){
    LIST_OF_CASES(string_metric::levenshtein_editops, default_process)
    }          
}

std::vector<rapidfuzz::LevenshteinEditOp>
levenshtein_editops_default_process(const proc_string& s1, const proc_string& s2)
{
    switch(s1.kind){
    LIST_OF_CASES(levenshtein_editops_inner_default_process, default_process)
    }          
}

# undef X_ENUM
