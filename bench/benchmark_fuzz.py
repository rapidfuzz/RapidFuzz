# todo combine benchmarks of scorers into common code base
from __future__ import annotations

import timeit

import pandas as pd


def benchmark(name, func, setup, lengths, count):
    print(f"starting {name}")
    start = timeit.default_timer()
    results = []
    for length in lengths:
        test = timeit.Timer(func, setup=setup.format(length, count))
        results.append(min(test.timeit(number=1) for _ in range(7)) / count)
    stop = timeit.default_timer()
    print(f"finished {name}, Runtime: ", stop - start)
    return results


setup = """
from rapidfuzz import fuzz as rfuzz
from fuzzywuzzy import fuzz
import string
import random
random.seed(18)
characters = string.ascii_letters + string.digits + string.whitespace + string.punctuation
a      = ''.join(random.choice(characters) for _ in range({0}))
b_list = [''.join(random.choice(characters) for _ in range({0})) for _ in range({1})]
"""

lengths = list(range(1, 512, 2))
count = 1000


def scorer_benchmark(funcname):
    time_rapidfuzz = benchmark("rapidfuzz", f"[rfuzz.{funcname}(a, b) for b in b_list]", setup, lengths, count)

    time_fuzzywuzzy = benchmark("fuzzywuzzy", f"[fuzz.{funcname}(a, b) for b in b_list]", setup, lengths, count)

    results = pd.DataFrame(
        data={
            "length": lengths,
            "rapidfuzz": time_rapidfuzz,
            "fuzzywuzzy": time_fuzzywuzzy,
        }
    )

    results.to_csv(f"results/{funcname}.csv", sep=",", index=False)


scorer_benchmark("ratio")
scorer_benchmark("partial_ratio")
scorer_benchmark("token_sort_ratio")
scorer_benchmark("token_set_ratio")
scorer_benchmark("partial_token_sort_ratio")
scorer_benchmark("partial_token_set_ratio")
scorer_benchmark("WRatio")

# token_ratio is unique to RapidFuzz
time_token_ratio = benchmark(
    "token_ratio",
    "[rfuzz.token_ratio(a, b, processor=None) for b in b_list]",
    setup,
    lengths,
    count,
)

# this gets very slow, so only benchmark it for smaller values
time_token_ratio_simple = benchmark(
    "fuzzywuzzy",
    "[max(rfuzz.token_sort_ratio(a, b, processor=None), rfuzz.token_set_ratio(a, b, processor=None)) for b in b_list]",
    setup,
    lengths,
    count,
)

results = pd.DataFrame(
    data={
        "length": lengths,
        "token_ratio": time_token_ratio,
        "max(token_sort_ratio, token_set_ratio)": time_token_ratio_simple,
    }
)

results.to_csv("results/token_ratio.csv", sep=",", index=False)

# partial_token_ratio is unique to RapidFuzz
time_partial_token_ratio = benchmark(
    "token_ratio",
    "[rfuzz.partial_token_ratio(a, b, processor=None) for b in b_list]",
    setup,
    lengths,
    count,
)

# this gets very slow, so only benchmark it for smaller values
time_partial_token_ratio_simple = benchmark(
    "fuzzywuzzy",
    (
        "[max(rfuzz.partial_token_sort_ratio(a, b, processor=None), "
        "rfuzz.partial_token_set_ratio(a, b, processor=None)) for b in b_list]"
    ),
    setup,
    lengths,
    count,
)

results = pd.DataFrame(
    data={
        "length": lengths,
        "partial_token_ratio": time_partial_token_ratio,
        "max(partial_token_sort_ratio, partial_token_set_ratio)": time_partial_token_ratio_simple,
    }
)

results.to_csv("results/partial_token_ratio.csv", sep=",", index=False)
