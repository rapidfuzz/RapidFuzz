#pragma once
#include <string>

// 0.0% - 100.0%
using percent = float;

namespace fuzz {
  percent ratio(const std::wstring &s1, const std::wstring &s2, percent score_cutoff=0, bool preprocess = true);
  percent partial_ratio(const std::wstring &s1, const std::wstring &s2, percent score_cutoff=0, bool preprocess = true);

  percent token_sort_ratio(const std::wstring &a, const std::wstring &b, percent score_cutoff=0, bool preprocess = true);
  percent partial_token_sort_ratio(const std::wstring &a, const std::wstring &b, percent score_cutoff=0, bool preprocess = true);

  percent token_set_ratio(const std::wstring &s1, const std::wstring &s2, percent score_cutoff=0, bool preprocess = true);
  percent partial_token_set_ratio(const std::wstring &s1, const std::wstring &s2, percent score_cutoff=0, bool preprocess = true);

  percent token_ratio(const std::wstring &s1, const std::wstring &s2, percent score_cutoff=0, bool preprocess = true);
  percent partial_token_ratio(const std::wstring &s1, const std::wstring &s2, percent score_cutoff=0, bool preprocess = true);

  percent WRatio(const std::wstring &s1, const std::wstring &s2, percent score_cutoff = 0, bool preprocess = true);
}
