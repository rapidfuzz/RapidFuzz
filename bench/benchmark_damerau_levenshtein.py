# todo combine benchmarks of scorers into common code base
import timeit
import pandas

def benchmark(name, func, setup, lengths, count):
    print(f"starting {name}")
    start = timeit.default_timer()
    results = []
    from tqdm import tqdm
    for length in tqdm(lengths):
        #for length in lengths:
        test = timeit.Timer(func, setup=setup.format(length, count))
        results.append(min(test.timeit(number=1) for _ in range(7)) / count)
    stop = timeit.default_timer()
    print(f"finished {name}, Runtime: ", stop - start)
    return results

setup ="""
from rapidfuzz.distance.DamerauLevenshtein import distance
from jellyfish import damerau_levenshtein_distance
import string
import random
random.seed(18)
characters = string.ascii_letters + string.digits + string.whitespace + string.punctuation
a      = ''.join(random.choice(characters) for _ in range({0}))
b_list = [''.join(random.choice(characters) for _ in range({0})) for _ in range({1})]
"""

lengths = list(range(1,256,2))
count = 1000

time_rapidfuzz = benchmark("rapidfuzz",
        '[distance(a, b) for b in b_list]',
        setup, lengths, count)

time_jellyfish = benchmark("jellyfish",
        '[damerau_levenshtein_distance(a, b) for b in b_list]',
        setup, lengths, count)

df = pandas.DataFrame(data={
    "length": lengths,
    "rapidfuzz": time_rapidfuzz,
    "jellyfish": time_jellyfish
})

df.to_csv("results/levenshtein_damerau.csv", sep=',',index=False)
