#pragma once
#include <string>
#include <vector>

/* 0.0% - 100.0% */
using percent = double;

struct DecomposedSet {
    std::vector<std::wstring_view> intersection;
    std::vector<std::wstring_view> difference_ab;
    std::vector<std::wstring_view> difference_ba;
    DecomposedSet(std::vector<std::wstring_view> intersection, std::vector<std::wstring_view> difference_ab, std::vector<std::wstring_view> difference_ba)
        : intersection(std::move(intersection))
        , difference_ab(std::move(difference_ab))
        , difference_ba(std::move(difference_ba))
    {}
};

struct Affix {
    std::size_t prefix_len;
    std::size_t suffix_len;
};

namespace utils {

std::vector<std::wstring_view> splitSV(const std::wstring_view& str);

DecomposedSet set_decomposition(std::vector<std::wstring_view> a, std::vector<std::wstring_view> b);

std::size_t joined_size(const std::wstring_view& x);

std::size_t joined_size(const std::vector<std::wstring_view>& x);

std::wstring join(const std::vector<std::wstring_view>& sentence);

percent result_cutoff(double result, percent score_cutoff);

void trim(std::wstring& s);

void lower_case(std::wstring& s);

std::wstring default_process(std::wstring s);

Affix remove_common_affix(std::wstring_view& a, std::wstring_view& b);

void remove_common_affix(std::vector<std::wstring_view>& a, std::vector<std::wstring_view>& b);
}
