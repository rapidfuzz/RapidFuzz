#include "process.hpp"
#include "fuzz.hpp"

std::vector<std::pair<float, std::string>>
process::extract(std::string query, std::vector<std::string> choices, uint8_t score_cutoff) {
  std::vector<std::pair<float, std::string>> results;
  results.reserve(choices.size());

  for (const auto &choice : choices) {
    float score = fuzz::WRatio(query, choice, score_cutoff);
    if (score > score_cutoff) {
      results.emplace_back(std::make_pair(score, choice));
    }
  }
  
  return results;
}


std::optional<std::pair<float, std::string>>
process::extract_one(std::string query, std::vector<std::string> choices, uint8_t score_cutoff) {
  float max_score = 0;
  std::string result_choice;
  for (const auto &choice : choices) {
    float score = fuzz::WRatio(query, choice, score_cutoff);
    if (score > score_cutoff) {
      score_cutoff = score;
      max_score = score;
      result_choice = choice;
    }
  }
  
  if (!max_score) {
    return {};
  }
  return std::make_pair(max_score, result_choice);
}