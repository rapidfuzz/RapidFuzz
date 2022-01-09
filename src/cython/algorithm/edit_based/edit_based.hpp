#pragma once
#include "cpp_common.hpp"

/**
 * @brief Edit operations used by the Levenshtein distance
 *
 * This represents an edit operation of type type which is applied to
 * the source string
 *
 * None:    s1[src_begin:src_end] == s1[dest_begin:dest_end]
 * Replace: s1[i1:i2] should be replaced by s2[dest_begin:dest_end]
 * Insert:  s2[dest_begin:dest_end] should be inserted at s1[src_begin:src_begin].
 *          Note that src_begin==src_end in this case.
 * Delete:  s1[src_begin:src_end] should be deleted.
 *          Note that dest_begin==dest_end in this case.
 */
struct LevenshteinOpcodes {
    rapidfuzz::LevenshteinEditType type; /**< type of the edit operation */
    std::size_t src_begin;    /**< index into the source string */
    std::size_t src_end;      /**< index into the source string */
    std::size_t dest_begin;   /**< index into the destination string */
    std::size_t dest_end;     /**< index into the destination string */
};

static inline bool operator ==(const LevenshteinOpcodes& a, const LevenshteinOpcodes& b) {
	return (a.type == b.type)
        && (a.src_begin == b.src_begin)
        && (a.src_end == b.src_end)
        && (a.dest_begin == b.dest_begin)
        && (a.dest_end == b.dest_end);
}

static inline size_t levenshtein_func(const RF_String& s1, const RF_String& s2,
    size_t insertion, size_t deletion, size_t substitution, size_t max)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::levenshtein(str1, str2, {insertion, deletion, substitution}, max);
    });
}
static inline bool LevenshteinInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs, size_t str_count, const RF_String* str)
{
    return scorer_init_u64<string_metric::CachedLevenshtein>(
        self, str_count, str, *(rapidfuzz::LevenshteinWeightTable*)(kwargs->context)
    );
}

static inline double normalized_levenshtein_func(const RF_String& s1, const RF_String& s2,
    size_t insertion, size_t deletion, size_t substitution, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::normalized_levenshtein(
            str1, str2, {insertion, deletion, substitution}, score_cutoff
        );
    });
}
static inline bool NormalizedLevenshteinInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs, size_t str_count, const RF_String* str)
{
    return scorer_init_f64<string_metric::CachedNormalizedLevenshtein>(
        self, str_count, str, *(rapidfuzz::LevenshteinWeightTable*)(kwargs->context)
    );
}

static inline size_t hamming_func(const RF_String& s1, const RF_String& s2, size_t max)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::hamming(str1, str2, max);
    });
}
static inline bool HammingInit(RF_ScorerFunc* self, const RF_Kwargs*, size_t str_count, const RF_String* str)
{
    return scorer_init_u64<string_metric::CachedHamming>(self, str_count, str);
}

static inline double normalized_hamming_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::normalized_hamming(str1, str2, score_cutoff);
    });
}
static inline bool NormalizedHammingInit(RF_ScorerFunc* self, const RF_Kwargs*, size_t str_count, const RF_String* str)
{
    return scorer_init_f64<string_metric::CachedNormalizedHamming>(self, str_count, str);
}

static inline size_t indel_func(const RF_String& s1, const RF_String& s2, size_t max)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::levenshtein(str1, str2, {1, 1, 2}, max);
    });
}
static inline bool IndelInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs, size_t str_count, const RF_String* str)
{
    rapidfuzz::LevenshteinWeightTable weights = {1, 1, 2};
    return scorer_init_u64<string_metric::CachedLevenshtein>(self, str_count, str, weights);
}

static inline double normalized_indel_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::normalized_levenshtein(str1, str2, {1, 1, 2}, score_cutoff);
    });
}
static inline bool NormalizedIndelInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs, size_t str_count, const RF_String* str)
{
    rapidfuzz::LevenshteinWeightTable weights = {1, 1, 2};
    return scorer_init_f64<string_metric::CachedNormalizedLevenshtein>(self, str_count, str, weights);
}

static inline double jaro_similarity_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::jaro_similarity(str1, str2, score_cutoff);
    });
}
static inline bool JaroSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs*, size_t str_count, const RF_String* str)
{
    return scorer_init_f64<string_metric::CachedJaroSimilarity>(self, str_count, str);
}

static inline double jaro_winkler_similarity_func(const RF_String& s1, const RF_String& s2,
    double prefix_weight, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::jaro_winkler_similarity(str1, str2, prefix_weight, score_cutoff);
    });
}
static inline bool JaroWinklerSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs, size_t str_count, const RF_String* str)
{
    return scorer_init_f64<string_metric::CachedJaroWinklerSimilarity>(self, str_count, str, *(double*)(kwargs->context));
}

static inline std::vector<rapidfuzz::LevenshteinEditOp> levenshtein_editops_func(
    const RF_String& s1, const RF_String& s2)
{
    return visitor(s1, s2, [](auto str1, auto str2) {
        return string_metric::levenshtein_editops(str1, str2);
    });
}

std::vector<rapidfuzz::LevenshteinEditOp> opcodes_to_editops(const std::vector<LevenshteinOpcodes>& ops)
{
    std::vector<rapidfuzz::LevenshteinEditOp> result;

    for (const auto& op : ops)
    {
        switch(op.type)
        {
        case rapidfuzz::LevenshteinEditType::None:
            break;

        case rapidfuzz::LevenshteinEditType::Replace:
            for (size_t j = 0; j < op.src_end - op.src_begin; j++) {
                result.push_back({
                    rapidfuzz::LevenshteinEditType::Replace,
                    op.src_begin + j,
                    op.dest_begin + j
                });
            }
            break;

        case rapidfuzz::LevenshteinEditType::Insert:
            for (size_t j = 0; j < op.dest_end - op.dest_begin; j++) {
                result.push_back({
                    rapidfuzz::LevenshteinEditType::Insert,
                    op.src_begin,
                    op.dest_begin + j
                });
            }
            break;

        case rapidfuzz::LevenshteinEditType::Delete:
            for (size_t j = 0; j < op.src_end - op.src_begin; j++) {
                result.push_back({
                    rapidfuzz::LevenshteinEditType::Delete,
                    op.src_begin + j,
                    op.dest_begin
                });
            }
            break;
        }
    }

    return result;
}


std::vector<LevenshteinOpcodes> editops_to_opcodes(const std::vector<rapidfuzz::LevenshteinEditOp>& ops, size_t src_len, size_t dest_len)
{
    std::vector<LevenshteinOpcodes> result;

    size_t src_pos = 0;
    size_t dest_pos = 0;
    for (size_t i = 0; i < ops.size();)
    {
        if (src_pos < ops[i].src_pos || dest_pos < ops[i].dest_pos)
        {
            result.push_back({
                rapidfuzz::LevenshteinEditType::None,
                src_pos, ops[i].src_pos,
                dest_pos, ops[i].dest_pos
            });
            src_pos = ops[i].src_pos;
            dest_pos = ops[i].dest_pos;
        }

        size_t src_begin = src_pos;
        size_t dest_begin = dest_pos;
        rapidfuzz::LevenshteinEditType type = ops[i].type;
        for (; i < ops.size(); ++i)
        {
            if (ops[i].type == type && src_pos < ops[i].src_pos && dest_pos < ops[i].dest_pos)
            {
                break;
            }

            switch(type)
            {
            case rapidfuzz::LevenshteinEditType::None:
                break;

            case rapidfuzz::LevenshteinEditType::Replace:
                src_pos++;
                dest_pos++;
                break;

            case rapidfuzz::LevenshteinEditType::Insert:
                dest_pos++;
                break;

            case rapidfuzz::LevenshteinEditType::Delete:
                src_pos++;
                break;
            }
        }

        result.push_back({type, src_begin, src_pos, dest_begin, dest_pos});
    }

    if (src_pos < src_len || dest_pos < dest_len)
    {
        result.push_back({
            rapidfuzz::LevenshteinEditType::None,
            src_pos, src_len,
            dest_pos, dest_len
        });
    }

    return result;
}
