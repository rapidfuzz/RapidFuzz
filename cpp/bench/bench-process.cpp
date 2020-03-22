#include <benchmark/benchmark.h>
#include "../src/process.hpp"
#include <string>
#include <vector>

static void BM_ProcessExtract1(benchmark::State &state) {
  std::wstring a = L"aaaaa";
  std::vector<std::wstring> b(1000, L"aaaaa");
  for (auto _ : state) {
    benchmark::DoNotOptimize(process::extract(a, b));
  }
  state.SetLabel("Similar Strings");
}

static void BM_ProcessExtract2(benchmark::State &state) {
  std::wstring a = L"aaaaa";
  std::vector<std::wstring> b(1000, L"bbbbb");
  for (auto _ : state) {
    benchmark::DoNotOptimize(process::extract(a, b));
  }
  state.SetLabel("Different Strings");
}

static void BM_ProcessExtract3(benchmark::State &state) {
  std::wstring a = L"aaaaa";
  std::vector<std::wstring> b(1000, L"bbbbbbbbbb");
  for (auto _ : state) {
    benchmark::DoNotOptimize(process::extract(a, b));
  }
  state.SetLabel("Different Strings and different length");
}

static void BM_ProcessExtract4(benchmark::State &state) {
  std::wstring a = L"please add bananas to my shopping list";
  std::vector<std::wstring> b(1000, L"can you add bananas to my shopping list please");
  for (auto _ : state) {
    benchmark::DoNotOptimize(process::extract(a, b));
  }
  state.SetLabel("Full sentence 95% equal");
}

static void BM_ProcessExtract5(benchmark::State &state) {
  std::wstring a = L"please add bananas to my shopping list";
  std::vector<std::wstring> b(1000, L"whats the weather like in Paris");
  for (auto _ : state) {
    benchmark::DoNotOptimize(process::extract(a, b));
  }
  state.SetLabel("Full sentence not very equal");
}

BENCHMARK(BM_ProcessExtract1);
BENCHMARK(BM_ProcessExtract2);
BENCHMARK(BM_ProcessExtract3);
BENCHMARK(BM_ProcessExtract4);
BENCHMARK(BM_ProcessExtract5);


static void BM_ProcessExtractOne1(benchmark::State &state) {
  std::wstring a = L"aaaaa";
  std::vector<std::wstring> b(1000, L"aaaaa");
  for (auto _ : state) {
    benchmark::DoNotOptimize(process::extractOne(a, b));
  }
  state.SetLabel("Similar Strings");
}

static void BM_ProcessExtractOne2(benchmark::State &state) {
  std::wstring a = L"aaaaa";
  std::vector<std::wstring> b(1000, L"bbbbb");
  for (auto _ : state) {
    benchmark::DoNotOptimize(process::extractOne(a, b));
  }
  state.SetLabel("Different Strings");
}

static void BM_ProcessExtractOne3(benchmark::State &state) {
  std::wstring a = L"aaaaa";
  std::vector<std::wstring> b(1000, L"bbbbbbbbbb");
  for (auto _ : state) {
    benchmark::DoNotOptimize(process::extractOne(a, b));
  }
  state.SetLabel("Different Strings and different length");
}

static void BM_ProcessExtractOne4(benchmark::State &state) {
  std::wstring a = L"please add bananas to my shopping list";
  std::vector<std::wstring> b(1000, L"can you add bananas to my shopping list please");
  for (auto _ : state) {
    benchmark::DoNotOptimize(process::extractOne(a, b));
  }
  state.SetLabel("Full sentence 95% equal");
}

static void BM_ProcessExtractOne5(benchmark::State &state) {
  std::wstring a = L"please add bananas to my shopping list";
  std::vector<std::wstring> b(1000, L"whats the weather like in Paris");
  for (auto _ : state) {
    benchmark::DoNotOptimize(process::extractOne(a, b));
  }
  state.SetLabel("Full sentence not very equal");
}

BENCHMARK(BM_ProcessExtractOne1);
BENCHMARK(BM_ProcessExtractOne2);
BENCHMARK(BM_ProcessExtractOne3);
BENCHMARK(BM_ProcessExtractOne4);
BENCHMARK(BM_ProcessExtractOne5);

BENCHMARK_MAIN();
