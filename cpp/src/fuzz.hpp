#pragma once
#include <string>

namespace fuzz {
  float ratio(const std::string &a, const std::string &b, float score_cutoff=0.0);
  float token_ratio(const std::string &a, const std::string &b, float score_cutoff=0.0);

  float QRatio(const std::string &a, const std::string &b, float score_cutoff = 0);
  float WRatio(const std::string &a, const std::string &b, float score_cutoff = 0);
}
