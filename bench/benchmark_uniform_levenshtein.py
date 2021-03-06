import timeit
import pandas
import numpy as np

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

setup ="""
from rapidfuzz import string_metric
import Levenshtein
import polyleven
import edlib
import editdistance
import string
import random
random.seed(18)
characters = string.ascii_letters + string.digits + string.whitespace + string.punctuation
a = ''.join(random.choice(characters) for _ in range({0}))
b_list = [''.join(random.choice(characters) for _ in range({0})) for _ in range({1})]
"""

lengths = list(range(1,512,2))
count = 2000

time_rapidfuzz = benchmark("rapidfuzz",
        '[string_metric.levenshtein(a, b) for b in b_list]',
        setup, lengths, count)

time_polyleven = benchmark("polyleven",
        '[polyleven.levenshtein(a, b) for b in b_list]',
        setup, lengths, count)

# this gets very slow, so only benchmark it for smaller values
time_python_levenshtein = benchmark("python-Levenshtein",
        '[Levenshtein.distance(a, b) for b in b_list]',
        setup, list(range(1,256,2)), count) + [np.NaN] * 128

time_edlib = benchmark("edlib",
        '[edlib.align(a, b) for b in b_list]',
        setup, lengths, count)

time_editdistance = benchmark("editdistance",
        '[editdistance.eval(a, b) for b in b_list]',
        setup, lengths, count)

df = pandas.DataFrame(data={
    "length": lengths,
    "rapidfuzz": time_rapidfuzz,
    "polyleven": time_polyleven,
    "python-Levenshtein": time_python_levenshtein,
    "edlib": time_edlib,
    "editdistance": time_editdistance
})

df.to_csv("results/levenshtein_uniform.csv", sep=',',index=False)
