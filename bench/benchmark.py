"""
This benchmark suite has the following goals:

- Help to detect performance regression in a new version
- Validate that an optimization change makes RapidFuzz faster and doesn't lead to major performance regressions
- Compare multiple implementations of the same algorithms in different libraries
- Showcase performance which ideally would be representative of performances of applications running
  on production
"""

from __future__ import annotations

import json
import math
import os
import random
import string
import subprocess
import sys
import timeit
from multiprocessing import Process, Value
from pathlib import Path
from urllib.request import urlopen

import click
import matplotlib.pyplot as plt
import pandas as pd
from packaging.version import Version
from tqdm import tqdm


def benchmark(result, setup, func, queries, choices):
    timer = timeit.Timer(func, setup=setup, globals={"queries": queries, "choices": choices})

    number = 1
    while True:
        timings = timer.repeat(repeat=7, number=number)
        min_timing = min(timings)
        # hopefully this reduces derivations in the benchmarks a bit
        if min_timing < 0.05:
            multiplicator = max(2, 0.05 / min_timing)
            number = math.ceil(number * multiplicator)
        else:
            result.value = min_timing / number
            return


# todo not sure how to handle ranges optimally here
# e.g. someone might want to
# - benchmark against all versions compatible
# - only the latest patch of each major
# - a custom range
# - only the installed version
def get_rapidfuzz_versions(version):
    url = "https://pypi.org/pypi/rapidfuzz/json"
    data = json.load(urlopen(url))
    versions = list(data["releases"].keys())
    versions.sort(key=Version, reverse=True)

    if version == "latest_patches":
        filtered_versions = []
        last_ver = None
        for version in versions:
            ver = Version(version)
            if last_ver is not None and last_ver.major == ver.major and last_ver.minor == ver.minor:
                continue

            last_ver = ver
            filtered_versions.append(ver)

        versions = filtered_versions

    return versions


def run_benchmark(datasets, setup, func) -> bool:
    results = []
    for dataset in tqdm(datasets["data"]):
        queries = [dataset[0]]
        choices = dataset[1:]

        result = Value("f", 0.0)
        # running this in a different process ensures, that imports do not "leak" out
        # this is relevant, since for regressions tests we install different versions of the module
        # during runtime
        process = Process(target=benchmark, args=(result, setup, func, queries, choices))
        process.start()
        process.join()
        # something went wrong
        if result.value == 0.0:
            return []
        results.append(result.value / len(choices))

    return results


def run_benchmarks_rapidfuzz(rapidfuzz_version, func_name, dataset, result_df):
    SCORERS = {
        "Jaro": "from rapidfuzz.distance.metrics_cpp import jaro_similarity as scorer",
        "JaroWinkler": "from rapidfuzz.distance.metrics_cpp import jaro_winkler_similarity as scorer",
        "OSA": "from rapidfuzz.distance.metrics_cpp import osa_distance as scorer",
        "Levenshtein": "from rapidfuzz.distance.metrics_cpp import levenshtein_distance as scorer",
        "Indel": "from rapidfuzz.distance.metrics_cpp import indel_distance as scorer",
        "DamerauLevenshtein": "from rapidfuzz.distance.metrics_cpp import damerau_levenshtein_distance as scorer",
        "Hamming": "from rapidfuzz.distance.metrics_cpp import hamming_distance as scorer",
    }

    setup = SCORERS[func_name]
    func = "[[scorer(a, b) for a in queries] for b in choices]"

    if rapidfuzz_version == "current":
        print("Benchmarking rapidfuzz")
        result_df["rapidfuzz"] = run_benchmark(dataset, setup, func)
    else:
        rapidfuzz_versions = get_rapidfuzz_versions(rapidfuzz_version)
        os.environ["RAPIDFUZZ_BUILD_EXTENSION"] = "1"

        for version in rapidfuzz_versions:
            print(f"Benchmarking rapidfuzz=={version}")
            res = subprocess.run(
                [sys.executable, "-m", "pip", "install", f"rapidfuzz=={version}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            # older versions are likely not installable
            if res.returncode:
                break

            # likely broken install on old version
            results = run_benchmark(dataset, setup, func)
            if not results:
                break

            result_df[f"rapidfuzz=={version}"] = results


def run_benchmarks_jellyfish(func_name, result_df, dataset):
    setup = "import jellyfish"
    JELLYFISH_SCORERS = {
        "Jaro": "jaro_similarity",
        "JaroWinkler": "jaro_winkler_similarity",
        "Levenshtein": "levenshtein_distance",
        "DamerauLevenshtein": "damerau_levenshtein_distance",
        "Hamming": "hamming_distance",
    }

    if func_name not in JELLYFISH_SCORERS:
        return

    print("Benchmarking jellyfish")
    func = f"[[jellyfish.{JELLYFISH_SCORERS[func_name]}(a, b) for a in queries] for b in choices]"
    result_df["jellyfish"] = run_benchmark(dataset, setup, func)


def run_benchmarks_polyleven(func_name, result_df, dataset):
    if func_name != "Levenshtein":
        return

    print("Benchmarking polyleven")
    setup = "import polyleven"
    func = "[[polyleven.levenshtein(a, b) for a in queries] for b in choices]"
    result_df["polyleven"] = run_benchmark(dataset, setup, func)


def run_benchmarks_edlib(func_name, result_df, dataset):
    if func_name != "Levenshtein":
        return

    print("Benchmarking edlib")
    setup = "import edlib"
    func = "[[edlib.align(a, b) for a in queries] for b in choices]"
    result_df["edlib"] = run_benchmark(dataset, setup, func)

    print("Benchmarking edlib(k=max)")
    func = "[[edlib.align(a, b, k=max(len(a), len(b))) for a in queries] for b in choices]"
    result_df["edlib(k=max)"] = run_benchmark(dataset, setup, func)


def run_benchmarks_editdistance(func_name, result_df, dataset):
    if func_name != "Levenshtein":
        return

    print("Benchmarking editdistance")
    setup = "import editdistance"
    func = "[[editdistance.eval(a, b) for a in queries] for b in choices]"
    result_df["editdistance"] = run_benchmark(dataset, setup, func)


def run_benchmarks_pyxdameraulevenshtein(func_name, result_df, dataset):
    if func_name != "OSA":
        return

    print("Benchmarking pyxdameraulevenshtein")
    setup = "import pyxdameraulevenshtein"
    func = "[[pyxdameraulevenshtein.damerau_levenshtein_distance(a, b) for a in queries] for b in choices]"
    result_df["pyxdameraulevenshtein"] = run_benchmark(dataset, setup, func)


AVAILABLE_BENCHMARKS = [
    "Jaro",
    "JaroWinkler",
    "OSA",
    "Levenshtein",
    "Indel",
    "DamerauLevenshtein",
]


@click.group
def cli():
    pass


@cli.command()
@click.option("--rapidfuzz_version", default="current", help="versions of rapidfuzz to benchmark")
@click.option(
    "--func",
    default="Levenshtein",
    type=click.Choice(AVAILABLE_BENCHMARKS, case_sensitive=False),
    help="function to benchmark",
)
def bench(rapidfuzz_version, func):
    temp_dir = Path(__file__).parent / "temp"
    temp_dir.mkdir(exist_ok=True)

    with open(temp_dir / "dataset.json") as f:
        dataset = json.load(f)

    result_df = pd.DataFrame(data={"x_axis": dataset["x_axis"]})

    run_benchmarks_rapidfuzz(rapidfuzz_version, func, dataset, result_df)

    # third party libs
    # run_benchmarks_jellyfish(func, result_df, dataset)
    run_benchmarks_polyleven(func, result_df, dataset)
    run_benchmarks_edlib(func, result_df, dataset)
    # run_benchmarks_editdistance(func, result_df, dataset)
    run_benchmarks_pyxdameraulevenshtein(func, result_df, dataset)

    result_df.to_csv(temp_dir / "result.csv", sep=",", index=False)


@cli.group()
def generate_dataset():
    pass


@generate_dataset.command()
@click.option("--start", default=0)
@click.option("--end", default=512)
@click.option("--step", default=4)
@click.option("--count", default=100)
def length_based(start, end, step, count):
    temp_dir = Path(__file__).parent / "temp"
    temp_dir.mkdir(exist_ok=True)

    random.seed(18)
    characters = string.ascii_letters + string.digits + string.whitespace + string.punctuation
    dataset = {"x_label": "string length [in characters]", "x_axis": [], "data": []}
    for i in range(start, end, step):
        data = ["".join(random.choice(characters) for _ in range(i)) for _ in range(count + 1)]
        dataset["data"].append(data)
        dataset["x_axis"].append(i)

    with open(temp_dir / "dataset.json", "w") as f:
        json.dump(dataset, f)


@cli.command()
def show():
    temp_dir = Path(__file__).parent / "temp"
    temp_dir.mkdir(exist_ok=True)

    with open(temp_dir / "dataset.json") as f:
        dataset = json.load(f)

    results = pd.read_csv("temp/result.csv")

    results.loc[:, (results.columns != "x_axis")] *= 1000 * 1000
    ax = results.plot(x="x_axis")

    # plt.xticks(list(range(0, 64*20+1, 64)))

    plt.title("Performance comparison of the \nDamerauLevenshtein similarity in different libraries")
    plt.xlabel(dataset["x_label"])
    plt.ylabel("runtime [μs]")
    ax.set_xlim(xmin=0)
    # ax.set_ylim(bottom=0)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    cli()
