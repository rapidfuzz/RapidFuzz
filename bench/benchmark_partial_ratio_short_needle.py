from __future__ import annotations

import timeit

import numpy as np
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
a      = ''.join(random.choice(characters) for _ in range(64))
b_list = [''.join(random.choice(characters) for _ in range({0})) for _ in range({1})]
"""

lengths = list(range(64, 512, 2))
count = 4000

time_rapidfuzz = benchmark("rapidfuzz", "[rfuzz.partial_ratio(a, b) for b in b_list]", setup, lengths, count)

# this gets very slow, so only benchmark it for smaller values
time_fuzzywuzzy = (
    benchmark(
        "fuzzywuzzy",
        "[fuzz.partial_ratio(a, b) for b in b_list]",
        setup,
        list(range(64, 256, 2)),
        count,
    )
    + [np.nan] * 128
)

results = pd.DataFrame(
    data={
        "length": lengths,
        "rapidfuzz": time_rapidfuzz,
        "fuzzywuzzy": time_fuzzywuzzy,
    }
)

results.to_csv("results/partial_ratio_short_needle.csv", sep=",", index=False)
