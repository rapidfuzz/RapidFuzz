#include <benchmark/benchmark.h>
#include "../src/levenshtein.hpp"
#include <string_view>
#include <vector>

// Define another benchmark
static void BM_LevWeightedDist1(benchmark::State &state) {
  std::string_view a = "aaaaaaaaaa";
  for (auto _ : state) {
    benchmark::DoNotOptimize(levenshtein::weighted_distance(a, a));
  }
  state.SetLabel("Similar Strings");
}

static void BM_LevWeightedDist2(benchmark::State &state) {
  std::string_view a = "aaaaaaaaaa";
  std::string_view b = "bbbbbbbbbb";
  for (auto _ : state) {
    benchmark::DoNotOptimize(levenshtein::weighted_distance(a, b));
  }
  state.SetLabel("Different Strings");
}

static void BM_LevWeightedDist3(benchmark::State &state) {
  std::string_view a = "aaaaaaaaaa";
  std::string_view b = "bbbbbbbbbb";
  for (auto _ : state) {
    benchmark::DoNotOptimize(levenshtein::weighted_distance(a, b, 20));
  }
  state.SetLabel("Different Strings with max distance (no early exit)");
}

static void BM_LevWeightedDist4(benchmark::State &state) {
  std::string_view a = "aaaaaaaaaa";
  std::string_view b = "bbbbbbbbbb";
  for (auto _ : state) {
    benchmark::DoNotOptimize(levenshtein::weighted_distance(a, b, 5));
  }
  state.SetLabel("Different Strings with max distance (early exit)");
}

static void BM_LevNormWeightedDist1(benchmark::State &state) {
  std::string_view a = "aaaaaaaaaa";
  for (auto _ : state) {
    benchmark::DoNotOptimize(levenshtein::normalized_weighted_distance(a, a));
  }
  state.SetLabel("Similar Strings");
}

static void BM_LevNormWeightedDist2(benchmark::State &state) {
  std::string_view a = "aaaaaaaaaa";
  std::string_view b = "bbbbbbbbbb";
  for (auto _ : state) {
    benchmark::DoNotOptimize(levenshtein::normalized_weighted_distance(a, b));
  }
  state.SetLabel("Different Strings");
}


BENCHMARK(BM_LevWeightedDist1);
BENCHMARK(BM_LevWeightedDist2);
BENCHMARK(BM_LevWeightedDist3);
BENCHMARK(BM_LevWeightedDist4);

BENCHMARK(BM_LevNormWeightedDist1);
BENCHMARK(BM_LevNormWeightedDist2);

BENCHMARK_MAIN();
