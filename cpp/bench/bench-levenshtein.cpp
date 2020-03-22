#include <benchmark/benchmark.h>
#include "../src/levenshtein.hpp"
#include <string>
#include <vector>

// Define another benchmark
static void BM_LevWeightedDist1(benchmark::State &state) {
  std::string_view a = "aaaaa aaaaa";
  for (auto _ : state) {
    benchmark::DoNotOptimize(levenshtein::weighted_distance(a, a));
  }
  state.SetLabel("Similar Strings");
}

static void BM_LevWeightedDist2(benchmark::State &state) {
  std::string_view a = "aaaaa aaaaa";
  std::string_view b = "bbbbb bbbbb";
  for (auto _ : state) {
    benchmark::DoNotOptimize(levenshtein::weighted_distance(a, b));
  }
  state.SetLabel("Different Strings");
}

static void BM_LevWeightedDist3(benchmark::State &state) {
  std::string_view a = "aaaaa aaaaa";
  std::string_view b = "bbbbb bbbbb";
  for (auto _ : state) {
    benchmark::DoNotOptimize(levenshtein::weighted_distance(a, b, 30));
  }
  state.SetLabel("Different Strings with max distance (no early exit)");
}

static void BM_LevWeightedDist4(benchmark::State &state) {
  std::string_view a = "aaaaa aaaaa";
  std::string_view b = "bbbbb bbbbb";
  for (auto _ : state) {
    benchmark::DoNotOptimize(levenshtein::weighted_distance(a, b, 5));
  }
  state.SetLabel("Different Strings with max distance (early exit)");
}

static void BM_LevNormWeightedDist1(benchmark::State &state) {
  std::string_view a = "aaaaa aaaaa";
  for (auto _ : state) {
    benchmark::DoNotOptimize(levenshtein::normalized_weighted_distance(a, a));
  }
  state.SetLabel("Similar Strings");
}

static void BM_LevNormWeightedDist2(benchmark::State &state) {
  std::string_view a = "aaaaa aaaaa";
  std::string_view b = "bbbbb bbbbb";
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


static void BM_LevWeightedDistVec1(benchmark::State &state) {
  std::vector<std::string_view> a {"aaaaa", "aaaaa"};
  for (auto _ : state) {
    benchmark::DoNotOptimize(levenshtein::weighted_distance(a, a));
  }
  state.SetLabel("Similar String Vectors");
}

static void BM_LevWeightedDistVec2(benchmark::State &state) {
  std::vector<std::string_view> a {"aaaaa", "aaaaa"};
  std::vector<std::string_view> b {"bbbbb", "bbbbb"};
  for (auto _ : state) {
    benchmark::DoNotOptimize(levenshtein::weighted_distance(a, b));
  }
  state.SetLabel("Different String Vectors");
}

static void BM_LevWeightedDistVec3(benchmark::State &state) {
  std::vector<std::string_view> a {"aaaaa", "aaaaa"};
  std::vector<std::string_view> b {"bbbbb", "bbbbb"};
  for (auto _ : state) {
    benchmark::DoNotOptimize(levenshtein::weighted_distance(a, b, 30));
  }
  state.SetLabel("Different String Vectors with max distance (no early exit)");
}

static void BM_LevWeightedDistVec4(benchmark::State &state) {
  std::vector<std::string_view> a {"aaaaa", "aaaaa"};
  std::vector<std::string_view> b {"bbbbb", "bbbbb"};
  for (auto _ : state) {
    benchmark::DoNotOptimize(levenshtein::weighted_distance(a, b, 5));
  }
  state.SetLabel("Different String Vectors with max distance (early exit)");
}

static void BM_LevNormWeightedDistVec1(benchmark::State &state) {
  std::vector<std::string_view> a {"aaaaa", "aaaaa"};
  for (auto _ : state) {
    benchmark::DoNotOptimize(levenshtein::normalized_weighted_distance(a, a));
  }
  state.SetLabel("Similar String Vectors");
}

static void BM_LevNormWeightedDistVec2(benchmark::State &state) {
  std::vector<std::string_view> a {"aaaaa", "aaaaa"};
  std::vector<std::string_view> b {"bbbbb", "bbbbb"};
  for (auto _ : state) {
    benchmark::DoNotOptimize(levenshtein::normalized_weighted_distance(a, b));
  }
  state.SetLabel("Different String Vectors");
}

BENCHMARK(BM_LevWeightedDistVec1);
BENCHMARK(BM_LevWeightedDistVec2);
BENCHMARK(BM_LevWeightedDistVec3);
BENCHMARK(BM_LevWeightedDistVec4);

BENCHMARK(BM_LevNormWeightedDistVec1);
BENCHMARK(BM_LevNormWeightedDistVec2);


BENCHMARK_MAIN();
