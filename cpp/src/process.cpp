#include "process.hpp"
#include "fuzz.hpp"
#include <algorithm>
#include <utility>
#include "utils.hpp"


std::vector<std::pair<std::wstring, float>>
process::extract(const std::wstring &query, const std::vector<std::wstring> &choices,
                 std::size_t limit, uint8_t score_cutoff, bool preprocess)
{
  std::vector<std::pair<std::wstring, float>> results;
  results.reserve(choices.size());

  std::wstring a;
  if (preprocess) {
    a = utils::default_process(query);
  } else {
    a = query;
  }

  for (const auto &choice : choices) {
    std::wstring b;
    if (preprocess) {
      b = utils::default_process(choice);
    } else {
      b = choice;
    }

    float score = fuzz::WRatio(query, choice, score_cutoff, false);
    if (score >= score_cutoff) {
      results.emplace_back(std::make_pair(choice, score));
    }
  }

  // TODO: possibly could use a similar improvement to extract_one
  // but when using limits close to choices.size() might actually be slower
  std::sort(results.rbegin(), results.rend(), [](auto const &t1, auto const &t2) {
    return std::get<1>(t1) < std::get<1>(t2);
  });

  if (limit < results.size()) {
    results.resize(limit);
  }
  
  return results;
}


std::optional<std::pair<std::wstring, float>>
process::extractOne(const std::wstring &query, const std::vector<std::wstring> &choices,
                    uint8_t score_cutoff, bool preprocess) {
  bool match_found = false;
  std::wstring result_choice;

  std::wstring a;
  if (preprocess) {
    a = utils::default_process(query);
  } else {
    a = query;
  }

  for (const auto &choice : choices) {
    std::wstring b;
    if (preprocess) {
      b = utils::default_process(choice);
    } else {
      b = choice;
    }

    float score = fuzz::WRatio(a, b, score_cutoff, false);
    if (score >= score_cutoff) {
      score_cutoff = score;
      match_found = true;
      result_choice = choice;
    }
  }
  
  if (!match_found) {
    return {};
  }
  return std::make_pair(result_choice, score_cutoff);
}