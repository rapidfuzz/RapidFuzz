#include "process.hpp"
#include <benchmark/benchmark.h>
#include <string>
#include <vector>

// Define another benchmark
static void BM_StringCopy(benchmark::State &state) {
  std::string a =
      "please add bananas to my shopping list I am a really reeally cool guy";
  std::vector<std::string> b(
      1000,
      "whats the weather like in Paris I am john Peter the new guy in class");
  for (auto _ : state) {
    benchmark::DoNotOptimize(extract_one(a, b));
  }
}

BENCHMARK(BM_StringCopy);

BENCHMARK_MAIN();
