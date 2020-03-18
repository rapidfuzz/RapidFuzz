#include "fuzz.hpp"
#include "levenshtein.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include <limits>


float partial_string_ratio(std::string a, std::string b, float score_cutoff=0.0) {
  if (a.empty() || b.empty()) {
    return 0.0;
  }

  return 0.0;
}

/*
fn partial_string_ratio(query: &str, choice: &str) -> f32 {
	if query.is_empty() || choice.is_empty() {
		return 0.0;
	}

	let shorter;
	let longer;
	
	if query.len() <= choice.len() {
		shorter = query;
		longer = choice;
	} else {
		longer = query;
		shorter = choice;
	}

	let edit_ops = editops::editops_find(shorter, longer);
	let blocks = editops::editops_matching_blocks(shorter.len(), longer.len(), &edit_ops);

	let mut scores: Vec<f32> = vec![];
	for block in blocks {
		let long_start = if block.second_start > block.first_start {
			block.second_start - block.first_start
		} else {
			0
		};

		let long_end = long_start + shorter.chars().count();
		let long_substr = &longer[long_start..long_end];

		let ls_ratio = normalized_weighted_levenshtein(shorter, long_substr);
	
		if ls_ratio > 0.995 {
			return 1.0;
		} else {
			scores.push(ls_ratio)
		}
			
	}

	scores.iter().fold(0.0f32, |max, &val| max.max(val))
}
*/


static float full_ratio(const std::string &query, const std::string &choice, float score_cutoff=0) {
  float sratio = fuzz::ratio(query, choice, score_cutoff);

  const float UNBASE_SCALE = 95;
  float min_ratio = std::max(score_cutoff, sratio);
  if (min_ratio < UNBASE_SCALE) {
    float unbased_score_cutoff = (score_cutoff > 70) ? score_cutoff/UNBASE_SCALE : 0;
    sratio = std::max(sratio, fuzz::token_ratio(query, choice, unbased_score_cutoff) * UNBASE_SCALE);
  }

  return sratio * 100.0;
}


float fuzz::ratio(const std::string &a, const std::string &b, float score_cutoff) {
  // this needs more thoughts when to start using score cutoff, since it performs slower when it can not exit early
  // has to be tested with some real training examples
  float sratio = (score_cutoff > 70)
    ? levenshtein::normalized_weighted_distance(a, b, score_cutoff / (float)100.0)
    : levenshtein::normalized_weighted_distance(a, b);

  return sratio * 100.0;
}


float fuzz::token_ratio(const std::string &a, const std::string &b, float score_cutoff) {
  std::vector<std::string_view> tokens_a = splitSV(a);
  std::sort(tokens_a.begin(), tokens_a.end());
  std::vector<std::string_view> tokens_b = splitSV(b);
  std::sort(tokens_b.begin(), tokens_b.end());

  auto intersection = intersection_count_sorted_vec(tokens_a, tokens_b);

  size_t ab_len = recursiveIterableSize(intersection.ab, 1);
  size_t ba_len = recursiveIterableSize(intersection.ba, 1);
  size_t double_prefix = 2 * recursiveIterableSize(intersection.sect, 1);

  // fuzzywuzzy joined sect and ab/ba for comparisions
  // this is not done here as an optimisation, so the lengths get incremented by 1
  // since there would be a whitespace between the joined strings
  if (double_prefix) {
    // exit early since this will always result in a ratio of 1
    if (!ab_len || !ba_len) return 1.0;

    ++ab_len;
    ++ba_len;
  }

  float result = levenshtein::normalized_weighted_distance(tokens_a, tokens_b, score_cutoff, " ");
  size_t lensum = ab_len + ba_len + double_prefix;
  size_t sect_distance = levenshtein::weighted_distance(intersection.ab, intersection.ba, score_cutoff, " ");
  if (sect_distance != std::numeric_limits<size_t>::max()) {
    result = std::max(result, (float)1.0 - sect_distance / (float)lensum);
  }

  // exit early since the other ratios are 0
  // (when a or b was empty they would cause a segfault)
  if (!double_prefix) {
    return result;
  }

  return std::max({
    result,
    // levenshtein distances sect+ab <-> sect and sect+ba <-> sect
    // would exit early after removing the prefix sect, so the distance can be directly calculated
    (float)1.0 - (float)ab_len / (float)(ab_len + double_prefix),
    (float)1.0 - (float)ba_len / (float)(ba_len + double_prefix)
  });
}


float partial_token_ratio(const std::string &a, const std::string &b, float score_cutoff=0.0) {
  // probably faster to split the String view already sorted
  std::vector<std::string_view> tokens_a = splitSV(a);
  std::sort(tokens_a.begin(), tokens_a.end());
  std::vector<std::string_view> tokens_b = splitSV(b);
  std::sort(tokens_b.begin(), tokens_b.end());

  auto intersection = intersection_count_sorted_vec(tokens_a, tokens_b);

  // the implementation in fuzzywuzzy would always return 1 here but calculate
  // the levenshtein distance at least 8 times to get to this result
  if (!intersection.sect.empty()) {
    return 1.0;
  }

  // joining the sentences here is probably not the fastest way
  return std::max(
    partial_string_ratio(sentenceJoin(tokens_a), sentenceJoin(tokens_b), score_cutoff),
    partial_string_ratio(sentenceJoin(intersection.ab), sentenceJoin(intersection.ba), score_cutoff)
  );
}


float partial_ratio(const std::string &query, const std::string &choice, float partial_scale, float score_cutoff) {
  const float UNBASE_SCALE = 0.95;
  float normalized_score_cutoff = score_cutoff / (float)100.0;

  // TODO: this needs more thoughts when to start using score cutoff, since it performs slower when it can not exit early
  // has to be tested with some real training examples
  float sratio = (score_cutoff > 70)
    ? levenshtein::normalized_weighted_distance(query, choice, normalized_score_cutoff)
    : levenshtein::normalized_weighted_distance(query, choice);

  float min_ratio = std::max(normalized_score_cutoff, sratio);
  if (min_ratio < partial_scale) {
    sratio = std::max(sratio, partial_string_ratio(query, choice) * partial_scale);
    min_ratio = std::max(sratio, min_ratio);

    if (min_ratio < UNBASE_SCALE * partial_scale) {
      sratio = std::max(sratio, partial_token_ratio(query, choice) * UNBASE_SCALE * partial_scale );
    }
  }
  return sratio * 100.0;
}


float fuzz::QRatio(const std::string &a, const std::string &b, float score_cutoff) {
  if (score_cutoff == 100) {
    return 0;
  }

  return ratio(a, b, score_cutoff);
}


float fuzz::WRatio(const std::string &a, const std::string &b, float score_cutoff) {
  if (score_cutoff == 100) {
    return 0;
  }

  size_t len_a = a.length();
  size_t len_b = b.length();
  float len_ratio = (len_a > len_b) ? (float)len_a / (float)len_b : (float)len_b / (float)len_a;

  if (len_ratio < 1.5) {
    return full_ratio(a, b, score_cutoff);
  // TODO: this is still missing
  } else if (len_ratio < 8.0) {
    return 0.0;
    // return partial_ratio(query, choice, 0.9, score_cutoff);
  } else {
    return 0.0;
    // return partial_ratio(query, choice, 0.6, score_cutoff);
  }
}
