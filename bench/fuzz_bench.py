from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict
import timeit
import os

class BenchmarkRunner:
  def __init__(self, functions, setups, names, lengths):
    self.functions = functions
    self.setups = setups
    self.names = names
    self.lengths = lengths
    self._result = None


  def run(self):
    self._result = defaultdict(list)

    for func, setup, name in zip(self.functions, self.setups, self.names):
        print(f"testing {func}...")
        for value in self.lengths:
            test = timeit.Timer(func.format(value), setup=setup.format(value))
            result = test.timeit(number=100000)
            self._result[name].append([value, result*10])


  def store(self, xlabel, title, file_name):
    c = iter(plt.get_cmap('rainbow')(np.linspace(0, 1, len(self._result))))  
    dataT = {k: list(zip(*v)) for k, v in self._result.items() }

    fig, ax = plt.subplots()
    for k, v in dataT.items():
        ax.plot(v[0], v[-1], c=next(c), lw=2)
    ax.legend(list(dataT), loc=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('time (µs)')
    ax.set_title(title)
    ax.grid(True)

    fig.set_size_inches(12.8, 9.6)
    fig.savefig(file_name, bbox_inches='tight', dpi=100)  


def fuzz_bench(name):
  setup ="""
from {0} import fuzz
import string
import random
random.seed(18)
characters = string.ascii_letters + string.digits + string.whitespace + string.punctuation
a = ''.join(random.choice(characters) for _ in range({1}))
b = (a + '.')[:-1] # make sure it is not the same object
no_affix_a = "a" + a + "a"
no_affix_b = "b" + a + "b"
no_suffix_a = a + "a"
no_suffix_b = a + "b"
"""

  bench_runner = BenchmarkRunner(
    functions=[
      f'fuzz.{name}(a, b)',
      f'fuzz.{name}(a, b, score_cutoff=80)',
      f'fuzz.{name}(a, b)'
    ],
    setups=[
      setup.format("rapidfuzz", "{0}"),
      setup.format("rapidfuzz", "{0}"),
      setup.format("fuzzywuzzy", "{0}")],
    names=["rapidfuzz", "rapidfuzz with score_cutoff=80", "fuzzywuzzy"],
    lengths=[int(1.4**x) for x in range(3,25)]
  )
  bench_runner.run()
  bench_runner.store('string length (in characters)', f'fuzz.{name} with similar strings', f'bench_results/{name}_similar.svg')

  bench_runner.functions = [
    f'fuzz.{name}(no_suffix_a, no_suffix_b)',
    f'fuzz.{name}(no_suffix_a, no_suffix_b, score_cutoff=80)',
    f'fuzz.{name}(no_suffix_a, no_suffix_b)'
  ]
  bench_runner.run()
  bench_runner.store('string length (in characters)', f'fuzz.{name} with mostly similar strings', f'bench_results/{name}_mostly_similar.svg')

  bench_runner.functions = [f'fuzz.{name}("a"*{"{0}"} , "b"*{"{0}"})']*2
  bench_runner.functions = [
    f'fuzz.{name}(no_affix_a, no_affix_b)',
    f'fuzz.{name}(no_affix_a, no_affix_b, score_cutoff=80)',
    f'fuzz.{name}(no_affix_a, no_affix_b)'
  ]
  bench_runner.lengths=[int(1.4**x) for x in range(3,20)]
  bench_runner.run()
  bench_runner.store('string length (in characters)', f'fuzz.{name} with different strings', f'bench_results/{name}_different.svg')

try:
  os.mkdir("bench_results")
except FileExistsError:
  pass

fuzz_bench("ratio")
fuzz_bench("partial_ratio")
fuzz_bench("token_sort_ratio")
fuzz_bench("partial_token_sort_ratio")
fuzz_bench("token_sort_ratio")
fuzz_bench("partial_token_sort_ratio")
fuzz_bench("token_sort_ratio")
fuzz_bench("partial_token_sort_ratio")
fuzz_bench("QRatio")
fuzz_bench("WRatio")
