#pragma once
#include "fuzz.hpp"
#include <execution>

uint8_t extract_one(std::string query, std::vector<std::string> choices,
                    uint8_t score_cutoff = 0) {
  uint8_t max_score = 0;
  for (const auto &choice : choices) {
    uint8_t score = ratio(query, choice, score_cutoff);
    if (score > score_cutoff) {
      score_cutoff = score;
      max_score = score;
    }
  }
  return max_score;
}

/*std::transform(std::execution::par,
    b.begin(), b.end(), out.begin(),
    [a](std::string elem) { return ratio(a, elem, 0); }

);*/