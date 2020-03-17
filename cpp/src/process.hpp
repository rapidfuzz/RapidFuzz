#pragma once
#include <optional>
#include <vector>
#include <string>

namespace process {
  std::vector<std::pair<float, std::string>>
  extract(std::string query, std::vector<std::string> choices, uint8_t score_cutoff = 0);

  std::optional<std::pair<float, std::string>>
  extract_one(std::string query, std::vector<std::string> choices, uint8_t score_cutoff = 0);
}


