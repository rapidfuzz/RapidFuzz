#pragma once
#include <string>

// 0.0% - 100.0%
using percent = float;

namespace fuzz {
  percent ratio(const std::wstring &a, const std::wstring &b, percent score_cutoff=0.0);
  percent partial_ratio(const std::wstring &a, const std::wstring &b, percent score_cutoff=0.0);

  percent token_sort_ratio(const std::wstring &a, const std::wstring &b, percent score_cutoff=0.0);
  percent partial_token_sort_ratio(const std::wstring &a, const std::wstring &b, percent score_cutoff=0.0);

  percent token_set_ratio(std::wstring a, std::wstring b, percent score_cutoff=0.0);
  percent partial_token_set_ratio(std::wstring a, std::wstring b, percent score_cutoff=0.0);

  percent token_ratio(const std::wstring &a, const std::wstring &b, percent score_cutoff=0.0);
  percent partial_token_ratio(const std::wstring &a, const std::wstring &b, percent score_cutoff=0.0);

  percent QRatio(std::wstring a, std::wstring b, percent score_cutoff = 0);
  percent WRatio(std::wstring a, std::wstring b, percent score_cutoff = 0);
}
