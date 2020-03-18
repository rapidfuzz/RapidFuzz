#pragma once
#include <string>

// 0.0% - 100.0%
using percent = float;

// 0.0 - 1.0
using decimal = float;

namespace fuzz {
  float ratio(const std::string &a, const std::string &b, float score_cutoff=0.0);
  float token_ratio(const std::string &a, const std::string &b, float score_cutoff=0.0);

  percent QRatio(const std::string &a, const std::string &b, percent score_cutoff = 0);
  percent WRatio(const std::string &a, const std::string &b, percent score_cutoff = 0);
}
