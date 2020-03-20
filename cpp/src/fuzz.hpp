#pragma once
#include <string>

// 0.0% - 100.0%
using percent = float;

namespace fuzz {
  percent ratio(const std::string &a, const std::string &b, percent score_cutoff=0.0);
  percent partial_ratio(const std::string &a, const std::string &b, percent score_cutoff=0.0);

  percent token_sort_ratio(const std::string &a, const std::string &b, percent score_cutoff=0.0);
  percent partial_token_sort_ratio(const std::string &a, const std::string &b, percent score_cutoff=0.0);

  percent token_set_ratio(const std::string &a, const std::string &b, percent score_cutoff=0.0);
  percent partial_token_set_ratio(const std::string &a, const std::string &b, percent score_cutoff=0.0);

  percent QRatio(const std::string &a, const std::string &b, percent score_cutoff = 0);
  percent WRatio(const std::string &a, const std::string &b, percent score_cutoff = 0);
}
