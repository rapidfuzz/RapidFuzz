#pragma once
#include <optional>
#include <vector>
#include <string>

namespace process {
  std::vector<std::pair<std::wstring, float>>
  extract(const std::wstring &query, const std::vector<std::wstring> &choices, std::size_t limit = 5, uint8_t score_cutoff = 0);

  std::optional<std::pair<std::wstring, float>>
  extractOne(const std::wstring &query, const std::vector<std::wstring> &choices, uint8_t score_cutoff = 0);
}


