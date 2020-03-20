#pragma once
#include <optional>
#include <vector>
#include <string>

namespace process {
  std::vector<std::pair<std::wstring, float>>
  extract(std::wstring query, std::vector<std::wstring> choices, std::size_t limit = 5, uint8_t score_cutoff = 0);

  std::optional<std::pair<std::wstring, float>>
  extract_one(std::wstring query, std::vector<std::wstring> choices, uint8_t score_cutoff = 0);
}


