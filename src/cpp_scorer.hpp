#pragma once
#include "cpp_common.hpp"

/* ratio */

double ratio_no_process(const proc_string& s1, const proc_string& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::ratio(str1, str2, score_cutoff);
    });
}
double ratio_default_process(const proc_string& s1, const proc_string& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::ratio(str1, str2, score_cutoff);
    });
}

double partial_ratio_no_process(const proc_string& s1, const proc_string& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_ratio(str1, str2, score_cutoff);
    });
}
double partial_ratio_default_process(const proc_string& s1, const proc_string& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_ratio(str1, str2, score_cutoff);
    });
}

double token_sort_ratio_no_process(const proc_string& s1, const proc_string& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::token_sort_ratio(str1, str2, score_cutoff);
    });
}
double token_sort_ratio_default_process(const proc_string& s1, const proc_string& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::token_sort_ratio(str1, str2, score_cutoff);
    });
}

double token_set_ratio_no_process(const proc_string& s1, const proc_string& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::token_set_ratio(str1, str2, score_cutoff);
    });
}
double token_set_ratio_default_process(const proc_string& s1, const proc_string& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::token_set_ratio(str1, str2, score_cutoff);
    });
}

double token_ratio_no_process(const proc_string& s1, const proc_string& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::token_ratio(str1, str2, score_cutoff);
    });
}
double token_ratio_default_process(const proc_string& s1, const proc_string& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::token_ratio(str1, str2, score_cutoff);
    });
}

double partial_token_sort_ratio_no_process(const proc_string& s1, const proc_string& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_token_sort_ratio(str1, str2, score_cutoff);
    });
}
double partial_token_sort_ratio_default_process(const proc_string& s1, const proc_string& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_token_sort_ratio(str1, str2, score_cutoff);
    });
}

double partial_token_set_ratio_no_process(const proc_string& s1, const proc_string& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_token_set_ratio(str1, str2, score_cutoff);
    });
}
double partial_token_set_ratio_default_process(const proc_string& s1, const proc_string& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_token_set_ratio(str1, str2, score_cutoff);
    });
}

double partial_token_ratio_no_process(const proc_string& s1, const proc_string& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_token_ratio(str1, str2, score_cutoff);
    });
}
double partial_token_ratio_default_process(const proc_string& s1, const proc_string& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_token_ratio(str1, str2, score_cutoff);
    });
}

double WRatio_no_process(const proc_string& s1, const proc_string& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::WRatio(str1, str2, score_cutoff);
    });
}
double WRatio_default_process(const proc_string& s1, const proc_string& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::WRatio(str1, str2, score_cutoff);
    });
}

double QRatio_no_process(const proc_string& s1, const proc_string& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::QRatio(str1, str2, score_cutoff);
    });
}
double QRatio_default_process(const proc_string& s1, const proc_string& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::QRatio(str1, str2, score_cutoff);
    });
}

/* string_metric */
PyObject* levenshtein_no_process(const proc_string& s1, const proc_string& s2,
    size_t insertion, size_t deletion, size_t substitution, size_t max)
{
    size_t result = visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::levenshtein(str1, str2, {insertion, deletion, substitution}, max);
    });
    return dist_to_long(result);
}
PyObject* levenshtein_default_process(const proc_string& s1, const proc_string& s2,
    size_t insertion, size_t deletion, size_t substitution, size_t max)
{
    size_t result = visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return string_metric::levenshtein(str1, str2, {insertion, deletion, substitution}, max);
    });
    return dist_to_long(result);
}

double normalized_levenshtein_no_process(const proc_string& s1, const proc_string& s2,
    size_t insertion, size_t deletion, size_t substitution, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::normalized_levenshtein(str1, str2, {insertion, deletion, substitution}, score_cutoff);
    });
}
double normalized_levenshtein_default_process(const proc_string& s1, const proc_string& s2,
    size_t insertion, size_t deletion, size_t substitution, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return string_metric::normalized_levenshtein(str1, str2, {insertion, deletion, substitution}, score_cutoff);
    });
}

PyObject* hamming_no_process(const proc_string& s1, const proc_string& s2, size_t max)
{
    size_t result = visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::hamming(str1, str2, max);
    });
    return dist_to_long(result);
}
PyObject* hamming_default_process(const proc_string& s1, const proc_string& s2, size_t max)
{
    size_t result = visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return string_metric::hamming(str1, str2, max);
    });
    return dist_to_long(result);
}

double normalized_hamming_no_process(const proc_string& s1, const proc_string& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::normalized_hamming(str1, str2, score_cutoff);
    });
}
double normalized_hamming_default_process(const proc_string& s1, const proc_string& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return string_metric::normalized_hamming(str1, str2, score_cutoff);
    });
}

double jaro_similarity_no_process(const proc_string& s1, const proc_string& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::jaro_similarity(str1, str2, score_cutoff);
    });
}
double jaro_similarity_default_process(const proc_string& s1, const proc_string& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return string_metric::jaro_similarity(str1, str2, score_cutoff);
    });
}

double jaro_winkler_similarity_no_process(const proc_string& s1, const proc_string& s2,
    double prefix_weight, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::jaro_winkler_similarity(str1, str2, prefix_weight, score_cutoff);
    });
}
double jaro_winkler_similarity_default_process(const proc_string& s1, const proc_string& s2,
    double prefix_weight, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return string_metric::jaro_winkler_similarity(str1, str2, prefix_weight, score_cutoff);
    });
}

std::vector<rapidfuzz::LevenshteinEditOp> levenshtein_editops_no_process(
    const proc_string& s1, const proc_string& s2)
{
    return visitor(s1, s2, [](auto str1, auto str2) {
        return string_metric::levenshtein_editops(str1, str2);
    });
}

std::vector<rapidfuzz::LevenshteinEditOp> levenshtein_editops_default_process(
    const proc_string& s1, const proc_string& s2)
{
    return visitor_default_process(s1, s2, [](auto str1, auto str2) {
        return string_metric::levenshtein_editops(str1, str2);
    });     
}
