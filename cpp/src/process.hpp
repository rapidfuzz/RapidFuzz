#pragma once
#include "fuzz.hpp"
//#include <execution>
#include <optional>


std::optional<std::pair<float, std::string>> extract_one(std::string query, std::vector<std::string> choices,
                    uint8_t score_cutoff = 0)
{
  if (choices.empty()) {
    return {};
  }
  float max_score = 0;
  std::string result_choice;
  for (const auto &choice : choices) {
    float score = fuzz::ratio(query, choice, score_cutoff);
    if (score > score_cutoff) {
      score_cutoff = score;
      max_score = score;
      result_choice = choice;
    }
  }

  return std::make_pair(max_score, result_choice);
}

/*std::transform(std::execution::par,
    b.begin(), b.end(), out.begin(),
    [a](std::string elem) { return ratio(a, elem, 0); }

);*/
