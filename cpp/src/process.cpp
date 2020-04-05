#include "process.hpp"
#include "fuzz.hpp"
#include "utils.hpp"
#include <algorithm>

std::vector<std::pair<std::wstring, double> >
process::extract(const std::wstring& query, const std::vector<std::wstring>& choices,
    std::size_t limit, double score_cutoff, bool preprocess)
{
    std::vector<std::pair<std::wstring, double> > results;
    results.reserve(choices.size());

    std::wstring a = (preprocess) ? utils::default_process(query) : query;

    for (const auto& choice : choices) {
        std::wstring b = (preprocess) ? utils::default_process(choice) : choice;

        double score = fuzz::WRatio(Sentence<wchar_t>(query), Sentence<wchar_t>(choice), score_cutoff);
        if (score >= score_cutoff) {
            results.emplace_back(std::make_pair(choice, score));
        }
    }

    // TODO: possibly could use a similar improvement to extract_one
    // but when using limits close to choices.size() might actually be slower
    std::sort(results.rbegin(), results.rend(), [](auto const& t1, auto const& t2) {
        return std::get<1>(t1) < std::get<1>(t2);
    });

    if (limit < results.size()) {
        results.resize(limit);
    }

    return results;
}

std::optional<std::pair<std::wstring, double> >
process::extractOne(const std::wstring& query, const std::vector<std::wstring>& choices,
    double score_cutoff, bool preprocess)
{
    bool match_found = false;
    std::wstring result_choice;

    std::wstring a = (preprocess) ? utils::default_process(query) : query;

    for (const auto& choice : choices) {
        std::wstring b = (preprocess) ? utils::default_process(choice) : choice;

        double score = fuzz::WRatio(Sentence<wchar_t>(a), Sentence<wchar_t>(b), score_cutoff);
        if (score >= score_cutoff) {
            score_cutoff = score;
            match_found = true;
            result_choice = choice;
        }
    }

    if (!match_found) {
        return {};
    }
    return std::make_pair(result_choice, score_cutoff);
}