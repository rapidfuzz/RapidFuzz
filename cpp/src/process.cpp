#include "process.hpp"
#include "fuzz.hpp"
#include <algorithm>

std::vector<std::pair<std::wstring, float>>
process::extract(const std::wstring &query, const std::vector<std::wstring> &choices, std::size_t limit, uint8_t score_cutoff) {
  std::vector<std::pair<std::wstring, float>> results;
  results.reserve(choices.size());

  for (const auto &choice : choices) {
    float score = fuzz::WRatio(query, choice, score_cutoff);
    if (score > score_cutoff) {
      results.emplace_back(std::make_pair(choice, score));
    }
  }

  // TODO: possibly could use a similar improvement to extract_one
  // but when using limits close to choices.size() might actually be slower
  if (limit < choices.size()) {
    std::sort(results.rbegin(), results.rend());
    results.resize(limit);
  }
  
  return results;
}


std::optional<std::pair<std::wstring, float>>
process::extractOne(const std::wstring &query, const std::vector<std::wstring> &choices, uint8_t score_cutoff) {
  bool match_found = false;
  std::wstring result_choice;
  for (const auto &choice : choices) {
    float score = fuzz::WRatio(query, choice, score_cutoff);
    if (score > score_cutoff) {
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