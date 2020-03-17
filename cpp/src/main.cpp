#include <benchmark/benchmark.h>
#include <iostream>
#include "levenshtein.hpp"
#include "fuzz.hpp"
#include "process.hpp"
#include <string>
#include <vector>

// Define another benchmark
static void BM_LevMatrix(benchmark::State &state) {
  std::string a = "please add bananas to my shopping list";
  std::string b = "whats the weather like in Paris";
  for (auto _ : state) {
    benchmark::DoNotOptimize(ratio(a, b, 0.0));
  }
}

static void BM_LevMatrix2(benchmark::State &state) {
  std::string a = "please add bananas to my shopping list peter";
  std::vector<std::string> b(1000, "can you add bananas to my shopping list please");
  for (auto _ : state) {
    //28
    benchmark::DoNotOptimize(extract_one(a, b));
  }
}

BENCHMARK(BM_LevMatrix2);

//BENCHMARK(BM_LevMatrix);
BENCHMARK_MAIN();
