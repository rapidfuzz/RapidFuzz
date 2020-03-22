#include <benchmark/benchmark.h>
#include "../src/fuzz.hpp"
#include "../src/levenshtein.hpp"
#include <string>
#include <vector>

static void BM_FuzzRatio1(benchmark::State &state) {
  std::wstring a = L"aaaaa aaaaa";
  for (auto _ : state) {
    benchmark::DoNotOptimize(fuzz::ratio(a, a));
  }
  state.SetLabel("Similar Strings");
}

static void BM_FuzzRatio2(benchmark::State &state) {
  std::wstring a = L"aaaaa aaaaa";
  std::wstring b = L"bbbbb bbbbb";
  for (auto _ : state) {
    benchmark::DoNotOptimize(fuzz::ratio(a, b));
  }
  state.SetLabel("Different Strings");
}

BENCHMARK(BM_FuzzRatio1);
BENCHMARK(BM_FuzzRatio2);


static void BM_FuzzPartialRatio1(benchmark::State &state) {
  std::wstring a = L"aaaaa aaaaa";
  for (auto _ : state) {
    benchmark::DoNotOptimize(fuzz::partial_ratio(a, a));
  }
  state.SetLabel("Similar Strings");
}

static void BM_FuzzPartialRatio2(benchmark::State &state) {
  std::wstring a = L"aaaaa aaaaa";
  std::wstring b = L"bbbbb bbbbb";
  for (auto _ : state) {
    benchmark::DoNotOptimize(fuzz::partial_ratio(a, b));
  }
  state.SetLabel("Different Strings");
}

BENCHMARK(BM_FuzzPartialRatio1);
BENCHMARK(BM_FuzzPartialRatio2);

static void BM_FuzzTokenSort1(benchmark::State &state) {
  std::wstring a = L"aaaaa aaaaa";
  for (auto _ : state) {
    benchmark::DoNotOptimize(fuzz::token_sort_ratio(a, a));
  }
  state.SetLabel("Similar Strings");
}

static void BM_FuzzTokenSort2(benchmark::State &state) {
  std::wstring a = L"aaaaa aaaaa";
  std::wstring b = L"bbbbb bbbbb";
  for (auto _ : state) {
    benchmark::DoNotOptimize(fuzz::token_sort_ratio(a, b));
  }
  state.SetLabel("Different Strings");
}

BENCHMARK(BM_FuzzTokenSort1);
BENCHMARK(BM_FuzzTokenSort2);

static void BM_FuzzPartialTokenSort1(benchmark::State &state) {
  std::wstring a = L"aaaaa aaaaa";
  for (auto _ : state) {
    benchmark::DoNotOptimize(fuzz::partial_token_sort_ratio(a, a));
  }
  state.SetLabel("Similar Strings");
}

static void BM_FuzzPartialTokenSort2(benchmark::State &state) {
  std::wstring a = L"aaaaa aaaaa";
  std::wstring b = L"bbbbb bbbbb";
  for (auto _ : state) {
    benchmark::DoNotOptimize(fuzz::partial_token_sort_ratio(a, b));
  }
  state.SetLabel("Different Strings");
}

BENCHMARK(BM_FuzzPartialTokenSort1);
BENCHMARK(BM_FuzzPartialTokenSort2);


static void BM_FuzzTokenSet1(benchmark::State &state) {
  std::wstring a = L"aaaaa aaaaa";
  for (auto _ : state) {
    benchmark::DoNotOptimize(fuzz::token_set_ratio(a, a));
  }
  state.SetLabel("Similar Strings");
}

static void BM_FuzzTokenSet2(benchmark::State &state) {
  std::wstring a = L"aaaaa aaaaa";
  std::wstring b = L"bbbbb bbbbb";
  for (auto _ : state) {
    benchmark::DoNotOptimize(fuzz::token_set_ratio(a, b));
  }
  state.SetLabel("Different Strings");
}

BENCHMARK(BM_FuzzTokenSet1);
BENCHMARK(BM_FuzzTokenSet2);


static void BM_FuzzPartialTokenSet1(benchmark::State &state) {
  std::wstring a = L"aaaaa aaaaa";
  for (auto _ : state) {
    benchmark::DoNotOptimize(fuzz::partial_token_set_ratio(a, a));
  }
  state.SetLabel("Similar Strings");
}

static void BM_FuzzPartialTokenSet2(benchmark::State &state) {
  std::wstring a = L"aaaaa aaaaa";
  std::wstring b = L"bbbbb bbbbb";
  for (auto _ : state) {
    benchmark::DoNotOptimize(fuzz::partial_token_set_ratio(a, b));
  }
  state.SetLabel("Different Strings");
}

BENCHMARK(BM_FuzzPartialTokenSet1);
BENCHMARK(BM_FuzzPartialTokenSet2);


static void BM_FuzzToken1(benchmark::State &state) {
  std::wstring a = L"aaaaa aaaaa";
  for (auto _ : state) {
    benchmark::DoNotOptimize(fuzz::token_ratio(a, a));
  }
  state.SetLabel("Similar Strings");
}

static void BM_FuzzToken2(benchmark::State &state) {
  std::wstring a = L"aaaaa aaaaa";
  std::wstring b = L"bbbbb bbbbb";
  for (auto _ : state) {
    benchmark::DoNotOptimize(fuzz::token_ratio(a, b));
  }
  state.SetLabel("Different Strings");
}

BENCHMARK(BM_FuzzToken1);
BENCHMARK(BM_FuzzToken2);


static void BM_FuzzPartialToken1(benchmark::State &state) {
  std::wstring a = L"aaaaa aaaaa";
  for (auto _ : state) {
    benchmark::DoNotOptimize(fuzz::partial_token_ratio(a, a));
  }
  state.SetLabel("Similar Strings");
}

static void BM_FuzzPartialToken2(benchmark::State &state) {
  std::wstring a = L"aaaaa aaaaa";
  std::wstring b = L"bbbbb bbbbb";
  for (auto _ : state) {
    benchmark::DoNotOptimize(fuzz::partial_token_ratio(a, b));
  }
  state.SetLabel("Different Strings");
}

BENCHMARK(BM_FuzzPartialToken1);
BENCHMARK(BM_FuzzPartialToken2);



static void BM_FuzzQRatio1(benchmark::State &state) {
  std::wstring a = L"aaaaa aaaaa";
  for (auto _ : state) {
    benchmark::DoNotOptimize(fuzz::QRatio(a, a));
  }
  state.SetLabel("Similar Strings");
}

static void BM_FuzzQRatio2(benchmark::State &state) {
  std::wstring a = L"aaaaa aaaaa";
  std::wstring b = L"bbbbb bbbbb";
  for (auto _ : state) {
    benchmark::DoNotOptimize(fuzz::QRatio(a, b));
  }
  state.SetLabel("Different Strings");
}

BENCHMARK(BM_FuzzQRatio1);
BENCHMARK(BM_FuzzQRatio2);

static void BM_FuzzWRatio1(benchmark::State &state) {
  std::wstring a = L"aaaaa aaaaa";
  for (auto _ : state) {
    benchmark::DoNotOptimize(fuzz::WRatio(a, a));
  }
  state.SetLabel("Similar Strings");
}

static void BM_FuzzWRatio3(benchmark::State &state) {
  std::wstring a = L"aaaaa aaaaa";
  std::wstring b = L"bbbbb bbbbb";
  for (auto _ : state) {
    benchmark::DoNotOptimize(fuzz::WRatio(a, b));
  }
  state.SetLabel("Different Strings");
}

static void BM_FuzzWRatio2(benchmark::State &state) {
  std::wstring a = L"aaaaa b";
  std::wstring b = L"bbbbb bbbbbbb";
  for (auto _ : state) {
    benchmark::DoNotOptimize(fuzz::WRatio(a, b));
  }
  state.SetLabel("Different length Strings");
}

BENCHMARK(BM_FuzzWRatio1);
BENCHMARK(BM_FuzzWRatio2);
BENCHMARK(BM_FuzzWRatio3);


BENCHMARK_MAIN();
